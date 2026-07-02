"""模型初始化端点。

提供 ``POST /api/v1/initialize`` 用于在运行时热加载模型权重与配置，
避免只能通过环境变量在启动时加载。该端点对应启动日志中的提示：
"Load via /api/v1/initialize"。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ...utils.io import safe_torch_load
from ...utils.logging import setup_logger
from ...utils.checkpoint_utils import resolve_inference_config
from .state import (
    STATE,
    initialize_model,
    initialize_pathway_mapper,
    initialize_variant_workflow,
    record_model_provenance,
)

logger = setup_logger(__name__)

router = APIRouter()


def _detect_checkpoint_format(checkpoint: Any) -> str:
    """识别 checkpoint 的存储格式，用于响应诊断与日志。"""
    import torch

    if not isinstance(checkpoint, dict):
        return "unknown"
    if "pytorch-lightning_version" in checkpoint or (
        "state_dict" in checkpoint and any(
            k.startswith("model.") for k in checkpoint["state_dict"]
        )
    ):
        return "lightning"
    if "model_state_dict" in checkpoint:
        return "legacy"
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return "state_dict"
    return "unknown"


class InitializeRequest(BaseModel):
    """热加载模型的请求体。"""

    checkpoint_path: str = Field(..., description="模型权重文件路径")
    config_path: Optional[str] = Field(
        None,
        description=(
            "配置文件路径。若留空，则自动查找 checkpoint 同目录的 "
            "<name>.config.yaml"
        ),
    )
    cell_states: Optional[List[str]] = Field(
        None,
        description="细胞状态标签列表。若留空，则使用配置文件中的 data.cell_states。",
    )
    device: Optional[str] = Field(
        None, description="加载设备（cuda/cpu）。留空时自动选择。"
    )


class InitializeResponse(BaseModel):
    """热加载结果。"""

    status: str = Field(..., description="加载状态: success/failed")
    checkpoint_path: str = Field(..., description="实际加载的权重路径")
    config_path: Optional[str] = Field(None, description="实际使用的配置路径")
    cell_states: List[str] = Field(default_factory=list, description="加载的细胞状态标签")
    device: str = Field(..., description="模型所在设备")
    message: Optional[str] = Field(None, description="附加信息（如警告）")
    loaded_format: Optional[str] = Field(
        None, description="识别到的 checkpoint 格式：state_dict/lightning/legacy"
    )
    missing_keys_count: int = Field(0, description="加载时缺失的权重键数量")
    unexpected_keys_count: int = Field(0, description="加载时多余的权重键数量")
    loaded_parameter_ratio: Optional[float] = Field(
        None, description="已加载参数占模型总参数的比例 (0-1)"
    )


@router.post("/initialize", response_model=InitializeResponse)
async def initialize_endpoint(request: InitializeRequest) -> InitializeResponse:
    """热加载模型权重与配置。

    步骤：
        1. 通过 ``resolve_inference_config`` 解析 checkpoint 与 config。
        2. 构建模型并加载 state_dict（strict=False，允许部分加载）。
        3. 调用 ``initialize_model`` 更新全局 STATE。
    """
    ckpt_path = Path(request.checkpoint_path)
    if not ckpt_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"checkpoint 不存在: {request.checkpoint_path}",
        )

    try:
        config_obj, config_source = resolve_inference_config(
            request.checkpoint_path, request.config_path
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    config: Dict[str, Any] = config_obj.to_dict()

    cell_states = request.cell_states or config_obj.get("data.cell_states") or []
    if not cell_states:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未提供 cell_states，且配置文件中也未定义 data.cell_states。",
        )

    device = request.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 延迟导入，避免在 API 模块加载阶段强制依赖 torch 模型构建
        from ...models.architectures import PTM2CellNet
        from ...utils.checkpoint_utils import extract_model_state_dict

        config_obj.set("model.num_classes", len(cell_states))
        model = PTM2CellNet.from_config(config_obj.to_dict())

        raw_checkpoint = safe_torch_load(str(ckpt_path), map_location=device)
        loaded_format = _detect_checkpoint_format(raw_checkpoint)
        state_dict = extract_model_state_dict(raw_checkpoint)

        # P0-2: 默认严格加载——权重不匹配时直接失败，避免用随机/部分初始化模型推理。
        # 统一 checkpoint 合约已能处理 Lightning .ckpt，因此裸加载应无 missing/unexpected。
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as shape_exc:
            # 形状不匹配（如 embed_dim/num_classes 不一致）会直接抛 RuntimeError，
            # 而非通过 missing/unexpected 报告。统一转为 400 可操作错误。
            logger.error(
                "initialize: checkpoint 权重形状不匹配 (format=%s): %s",
                loaded_format, shape_exc,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"checkpoint 权重与模型结构不匹配 (format={loaded_format})，"
                    f"通常因 config（如 embed_dim/num_classes/vocab_size）与训练时不一致："
                    f"{shape_exc}"
                ),
            ) from shape_exc

        if missing or unexpected:
            total_params = sum(p.numel() for p in model.parameters())
            matched_params = sum(
                p.numel()
                for name, p in model.named_parameters()
                if name not in missing and name in state_dict
            )
            ratio = (matched_params / total_params) if total_params else 0.0
            logger.error(
                "initialize: checkpoint 权重不匹配 (format=%s) missing=%d unexpected=%d "
                "loaded_ratio=%.4f",
                loaded_format, len(missing), len(unexpected), ratio,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"checkpoint 权重与模型不匹配 (format={loaded_format})："
                    f"缺失 {len(missing)} 个键，多余 {len(unexpected)} 个键，"
                    f"参数覆盖率 {ratio:.1%}。请使用与当前 config 匹配的 checkpoint，"
                    f"或 Lightning 训练导出的 best_model.pt。"
                    f"missing[:5]={missing[:5]}, unexpected[:5]={unexpected[:5]}"
                ),
            )

        initialize_model(model, cell_states, model_device=device, config=config)
        record_model_provenance(str(ckpt_path), config_source, config_obj.to_dict())
        initialize_pathway_mapper()
        try:
            initialize_variant_workflow(str(ckpt_path))
        except Exception as exc:
            logger.warning("initialize: variant workflow 初始化失败: %s", exc)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "模型热加载完成: checkpoint=%s, config=%s, device=%s, format=%s",
            ckpt_path, config_source, device, loaded_format,
        )

        return InitializeResponse(
            status="success",
            checkpoint_path=str(ckpt_path),
            config_path=config_source,
            cell_states=STATE.cell_states,
            device=device,
            message=None,
            loaded_format=loaded_format,
            missing_keys_count=0,
            unexpected_keys_count=0,
            loaded_parameter_ratio=1.0 if total_params else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("模型热加载失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型加载失败: {exc}",
        ) from exc


__all__ = ["router", "initialize_endpoint", "InitializeRequest", "InitializeResponse"]
