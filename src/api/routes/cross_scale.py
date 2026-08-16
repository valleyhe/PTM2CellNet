"""跨尺度模型（cross-scale）API 端点。

提供 ``POST /api/v1/cross-scale/initialize``、``/cross-scale/predict`` 与
``/cross-scale/batch_predict`` 三个端点，把离线跨尺度 artifact
（``scripts/train_cross_scale.py`` 产物）接入在线服务（F-01，接口契约见
``project_analysis_20260816.md`` §4.2 缺失接口表）。

设计要点：

* **独立状态槽**：跨尺度模型与标准 ``PTM2CellNet`` 的输入/输出/生命周期
  契约完全不同（NPZ embedding 或原始序列 vs aa-index 张量），因此使用
  独立的 ``CROSS_SCALE_STATE`` 而不复用 ``STATE``，两类服务可并存且互不
  破坏对方的前置组件（标准 ``initialize_model`` 会重建 FeatureExtractor
  并清除 variant workflow）。
* **严格加载**：复用 ``load_cross_scale_artifact`` 的全部校验
  （schema / label 词表 / 路径穿越 / ``weights_only=True`` / strict 权重）；
  ``strict_assets=False`` 仅放宽 best→last checkpoint 的回退，其余校验不变。
* **显式 fallback 报告**：契约要求"禁止隐式随机 fallback"——当推理依赖
  pLM fallback 向量（未加载真实 backbone）或图近似（请求未携带 signal
  graph）时，响应通过 ``fallback_flags`` 显式标记，绝不静默。
* **门禁复用**：与 ``/initialize`` 同级的 API key 门禁与路径安全校验。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...data.features import DEFAULT_PTM_TYPES
from ...inference.cross_scale_predictor import (
    ARTIFACT_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    CrossScaleArtifactError,
    load_cross_scale_artifact,
)
from ...models.cross_scale import CrossScaleContractError, CrossScalePTM2CellNet
from ...utils.logging import setup_logger
from ..schemas import PTMSite
from .initialize import _require_api_key, _validate_path_within_allowed

logger = setup_logger(__name__)

# 跨尺度 NPZ 输入支持的 schema 版本（数据集与 embedding 预计算产物）。
_CROSS_SCALE_NPZ_SCHEMA = "ptm2cellnet.cross-scale.npz.v1"
_EMBEDDING_CACHE_SCHEMA = "ptm2cellnet.embedding-cache.v1"

# 单请求样本数上限，与 /batch_predict 的 1000 上限相比更保守：
# 跨尺度单样本推理包含 pLM 编码或大 embedding 加载，成本显著更高。
_MAX_BATCH_SAMPLES = 256


@dataclass
class _CrossScaleState:
    """跨尺度模型服务状态（与标准 :data:`STATE` 相互独立）。"""

    model: Optional[CrossScalePTM2CellNet] = None
    manifest: Optional[Dict[str, Any]] = None
    labels: List[str] = field(default_factory=list)
    device: str = "cpu"
    artifact_path: Optional[str] = None
    manifest_digest: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    max_batch_size: int = 8
    # 服务端默认图（cell_edge_index / signal_edge_index / signal_gene_map 等）。
    # 模型契约要求 cell graph 显式提供（"不可静默替换"），在线请求无法内嵌
    # 大图，因此由 initialize 时的 graph_ref 登记为默认图；单样本 embedding
    # NPZ 自带图数组时优先生效。
    default_graph: Optional[Dict[str, torch.Tensor]] = None
    graph_ref: Optional[str] = None
    initialization_failed: bool = False


CROSS_SCALE_STATE = _CrossScaleState()

# API 层 PTM 类型字符串 → 数值 ID 的单源词表（1-based，0 保留给 padding，
# 与 PTMTokenAdapter 及标准 API initialize_model 的约定一致）。
PTM_TYPE_TO_IDX: Dict[str, int] = {ptm_type: index + 1 for index, ptm_type in enumerate(DEFAULT_PTM_TYPES)}


def reset_cross_scale_state() -> None:
    """Reset the cross-scale service state to its uninitialized defaults.

    Mirrors :func:`src.api.routes.state.reset_state` so tests and operational
    tooling can deterministically clear the cross-scale model slot.
    """
    fresh = _CrossScaleState()
    for field_name in ("model", "manifest", "labels", "device", "artifact_path",
                       "manifest_digest", "model_info", "max_batch_size",
                       "default_graph", "graph_ref", "initialization_failed"):
        setattr(CROSS_SCALE_STATE, field_name, getattr(fresh, field_name))


# 图数组键：embedding NPZ 或 graph_ref NPZ 中可携带的结构输入。模型契约
# 要求 cell graph 显式提供（"不可静默替换"），这些键从 NPZ 透传进 batch。
_GRAPH_ARRAY_KEYS = (
    "cell_edge_index",
    "cell_edge_weight",
    "signal_edge_index",
    "signal_edge_weight",
    "signal_gene_map",
)


def _read_npz_graph_arrays(payload: "np.lib.npyio.NpzFile") -> Dict[str, torch.Tensor]:
    """Extract graph structural arrays (if present) from an opened NPZ payload."""
    # edge_index / signal_gene_map 为二维；edge_weight 每边一个标量为 [E] 一维。
    two_dim_keys = ("cell_edge_index", "signal_edge_index", "signal_gene_map")
    graphs: Dict[str, torch.Tensor] = {}
    for key in _GRAPH_ARRAY_KEYS:
        if key not in payload:
            continue
        array = np.asarray(payload[key])
        expected_ndim = 2 if key in two_dim_keys else 1
        if array.ndim != expected_ndim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"图数组 {key} 必须是 {expected_ndim} 维数组，实际 ndim={array.ndim}",
            )
        dtype = torch.long if key in ("cell_edge_index", "signal_edge_index") else torch.float32
        graphs[key] = torch.from_numpy(np.ascontiguousarray(array)).to(dtype)
    return graphs


def _load_default_graph(graph_ref: str, model: CrossScalePTM2CellNet) -> Dict[str, torch.Tensor]:
    """加载 initialize 时的服务端默认图 NPZ 并做最小契约校验。"""
    path = _validate_path_within_allowed(Path(graph_ref))
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"graph_ref 文件不存在: {graph_ref}",
        )
    try:
        payload = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无法解析 graph_ref NPZ {graph_ref}: {exc}",
        ) from exc
    with payload:
        graphs = _read_npz_graph_arrays(payload)
    if "cell_edge_index" not in graphs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"graph_ref NPZ 缺少 cell_edge_index 数组: {graph_ref}",
        )
    num_genes = model.cross_scale_config.num_cell_genes
    cell_edges = graphs["cell_edge_index"]
    if cell_edges.numel() and (
        int(cell_edges.max().item()) >= num_genes or int(cell_edges.min().item()) < 0
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"graph_ref cell_edge_index 节点越界（模型 num_cell_genes={num_genes}）",
        )
    if "signal_gene_map" in graphs and graphs["signal_gene_map"].shape[-1] != num_genes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"graph_ref signal_gene_map 最后一维 {graphs['signal_gene_map'].shape[-1]} "
                f"与 num_cell_genes={num_genes} 不一致"
            ),
        )
    return graphs


def _manifest_digest(artifact_dir: Path) -> str:
    """Compute the sha256 digest of the artifact manifest file for provenance."""
    payload = (artifact_dir / "artifact_manifest.json").read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _require_model() -> None:
    if CROSS_SCALE_STATE.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="跨尺度模型未初始化，请先调用 POST /api/v1/cross-scale/initialize",
        )


router = APIRouter()


class CrossScaleInitializeRequest(BaseModel):
    """跨尺度 artifact 加载请求（F-01 契约）。"""

    artifact_path: str = Field(..., description="跨尺度 artifact 目录（train_cross_scale.py --output 产物）")
    device: Optional[str] = Field(None, description="加载设备（cpu/cuda），留空自动选择")
    strict_assets: bool = Field(
        True,
        description="严格模式：仅接受 best_checkpoint。False 时 best 缺失回退 last_checkpoint，"
        "其余校验（schema/词表/权重 strict 加载）保持不变。",
    )
    max_batch_size: int = Field(8, ge=1, le=512, description="服务端推理批次上限")
    graph_ref: Optional[str] = Field(
        None,
        description="服务端默认图 NPZ 路径（含 cell_edge_index，可选 signal_edge_index/"
        "signal_gene_map/cell_edge_weight）。模型契约要求 cell graph 显式提供；"
        "在线请求无图数组时使用此默认图。",
    )


class CrossScaleModelSummary(BaseModel):
    protein_backbones: List[str] = Field(default_factory=list)
    loaded_plm_model_names: Dict[str, str] = Field(default_factory=dict)
    num_cell_genes: int = 0
    num_cell_states: int = 0
    num_ptm_types: int = 0
    allow_plm_fallback: bool = False
    allow_graph_approximation: bool = False
    signal_graph_version: str = "unversioned"
    data_manifest_digest: Optional[str] = None


class CrossScaleInitializeResponse(BaseModel):
    status: str
    model_type: str = "cross_scale"
    checkpoint_schema: str = CHECKPOINT_SCHEMA_VERSION
    artifact_schema: str = ARTIFACT_SCHEMA_VERSION
    artifact_path: str
    manifest_digest: str
    label_vocabulary: List[str] = Field(default_factory=list)
    device: str
    max_batch_size: int
    readiness: str
    graph_registered: bool = False
    model_summary: CrossScaleModelSummary
    message: Optional[str] = None


@router.post("/cross-scale/initialize", response_model=CrossScaleInitializeResponse)
async def cross_scale_initialize(
    request: CrossScaleInitializeRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> CrossScaleInitializeResponse:
    """加载跨尺度 artifact 并登记服务状态。

    加载本身（权重反序列化 + 模型构建）是阻塞 CPU 工作，卸载到线程池
    （N01 惯例）。失败时置 ``initialization_failed`` 以便运维排障。
    """
    _require_api_key(x_api_key)

    artifact_dir = _validate_path_within_allowed(Path(request.artifact_path))
    if not artifact_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"artifact 目录不存在: {request.artifact_path}",
        )

    device = request.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load() -> Tuple[CrossScalePTM2CellNet, Dict[str, Any], str, Optional[Dict[str, torch.Tensor]]]:
        model, manifest, _checkpoint = load_cross_scale_artifact(
            artifact_dir,
            device=device,
            allow_last_checkpoint_fallback=not request.strict_assets,
        )
        default_graph = (
            _load_default_graph(request.graph_ref, model) if request.graph_ref else None
        )
        return model, manifest, _manifest_digest(artifact_dir), default_graph

    try:
        model, manifest, digest, default_graph = await run_in_threadpool(_load)
    except HTTPException:
        # graph_ref 校验失败等可操作错误直接透传（在线程池内抛出）
        raise
    except CrossScaleArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"跨尺度 artifact 校验失败: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - 转换为可操作的 5xx 响应
        CROSS_SCALE_STATE.initialization_failed = True
        logger.error("cross-scale initialize 失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"跨尺度模型加载失败: {exc}",
        ) from exc

    model_info = model.get_model_info()
    CROSS_SCALE_STATE.model = model
    CROSS_SCALE_STATE.manifest = manifest
    CROSS_SCALE_STATE.labels = list(manifest["label_vocabulary"])
    CROSS_SCALE_STATE.device = device
    CROSS_SCALE_STATE.artifact_path = str(artifact_dir)
    CROSS_SCALE_STATE.manifest_digest = digest
    CROSS_SCALE_STATE.model_info = model_info
    CROSS_SCALE_STATE.max_batch_size = request.max_batch_size
    CROSS_SCALE_STATE.default_graph = default_graph
    CROSS_SCALE_STATE.graph_ref = request.graph_ref
    CROSS_SCALE_STATE.initialization_failed = False

    cfg = model.cross_scale_config
    summary = CrossScaleModelSummary(
        protein_backbones=list(model.protein_encoder.backbone_names),
        loaded_plm_model_names=dict(model_info.get("loaded_plm_model_names", {})),
        num_cell_genes=cfg.num_cell_genes,
        num_cell_states=cfg.num_cell_states,
        num_ptm_types=cfg.num_ptm_types,
        allow_plm_fallback=model.allow_plm_fallback,
        allow_graph_approximation=model.allow_graph_approximation,
        signal_graph_version=model.signal_graph_version,
        data_manifest_digest=model.data_manifest_digest,
    )
    logger.info(
        "cross-scale 模型已加载: artifact=%s device=%s labels=%s digest=%s",
        artifact_dir, device, CROSS_SCALE_STATE.labels, digest[:12],
    )
    return CrossScaleInitializeResponse(
        status="success",
        artifact_path=str(artifact_dir),
        manifest_digest=digest,
        label_vocabulary=CROSS_SCALE_STATE.labels,
        device=device,
        max_batch_size=request.max_batch_size,
        readiness="ready",
        graph_registered=default_graph is not None,
        model_summary=summary,
    )


class CrossScalePredictRequest(BaseModel):
    """单样本跨尺度预测请求（F-01 契约：sequence / embedding_ref 二选一）。"""

    sequence: Optional[str] = Field(None, min_length=1, description="原始氨基酸序列")
    embedding_ref: Optional[str] = Field(
        None,
        description="预计算 embedding NPZ 的服务端路径（prepare_cross_scale_embeddings.py 产物或"
        "跨尺度 NPZ），需含 {backbone}_embeddings 数组",
    )
    sample_index: int = Field(0, ge=0, description="embedding_ref NPZ 中的样本行号")
    ptm_sites: List[PTMSite] = Field(default_factory=list)
    sample_id: Optional[str] = None
    include_delta_expression: bool = Field(
        False, description="是否返回 num_cell_genes 维 delta_expression 向量（默认关闭，响应更小）"
    )
    allow_uniform_signal_map: bool = Field(
        False,
        description="显式工程选择：缺少可用的 signal_gene_map 时，按当前请求的蛋白质节点数构造"
        "均匀映射（signal→gene 无信息退化），并在 fallback_flags.signal_map_uniform 标记。"
        "默认关闭，缺少映射时显式 400。",
    )

    @property
    def effective_sample_id(self) -> str:
        return self.sample_id or (
            f"seq:{len(self.sequence)}" if self.sequence else f"emb:{self.sample_index}"
        )


class CrossScalePrediction(BaseModel):
    sample_id: str
    cell_state: str
    confidence: float
    probabilities: Dict[str, float]
    cell_state_logits: List[float]
    delta_expression: Optional[List[float]] = None
    num_cell_genes: int
    ptm_sites_applied: int
    fallback_flags: Dict[str, bool]
    provenance: Dict[str, Any]


class CrossScalePredictResponse(CrossScalePrediction):
    processing_time_ms: Optional[float] = None


class CrossScaleBatchPredictRequest(BaseModel):
    samples: List[CrossScalePredictRequest] = Field(..., min_length=1, max_length=_MAX_BATCH_SAMPLES)
    batch_size: int = Field(8, ge=1, le=64)
    fail_fast: bool = Field(
        False, description="True 时首个失败样本立即中止并返回 400；False 时逐样本收集 errors"
    )


class CrossScaleBatchError(BaseModel):
    index: int
    sample_id: str
    detail: str


class CrossScaleBatchSummary(BaseModel):
    total: int
    succeeded: int
    failed: int


class CrossScaleBatchPredictResponse(BaseModel):
    predictions: List[CrossScalePrediction]
    errors: List[CrossScaleBatchError]
    summary: CrossScaleBatchSummary
    provenance: Dict[str, Any]


def _fallback_flags(
    model: CrossScalePTM2CellNet, *, graph_supplied: bool, sequence_mode: bool
) -> Dict[str, bool]:
    """显式报告本次推理依赖的工程 fallback（契约：禁止隐式随机 fallback）。

    ``plm_fallback_used`` 仅在原始序列路径下有意义：embedding 路径消费的
    是预计算真实表示，不触及 fallback 向量。
    """
    loaded_backbones = dict(getattr(model.protein_encoder, "backbone_model_names", {}) or {})
    return {
        "plm_fallback_used": sequence_mode and model.allow_plm_fallback and not loaded_backbones,
        "graph_approximation_used": model.allow_graph_approximation and not graph_supplied,
    }


def _ptm_tensors(
    ptm_sites: List[PTMSite], num_ptm_types: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 API PTM 位点映射为模型期望的 1-based 类型/位置张量。"""
    type_ids: List[int] = []
    positions: List[int] = []
    for site in ptm_sites:
        type_id = PTM_TYPE_TO_IDX.get(site.type.strip().lower())
        if type_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"未知 PTM 类型: {site.type!r}。支持的类型词表: {sorted(PTM_TYPE_TO_IDX)}",
            )
        if type_id > num_ptm_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"PTM 类型 {site.type!r} 的 ID {type_id} 超出模型 num_ptm_types={num_ptm_types}，"
                    "该 artifact 训练词表不覆盖此类型"
                ),
            )
        type_ids.append(type_id)
        positions.append(int(site.position))
    return (
        torch.tensor([type_ids], dtype=torch.long),
        torch.tensor([positions], dtype=torch.long),
    )


def _validate_single_input(sample: CrossScalePredictRequest) -> None:
    if (sample.sequence is None) == (sample.embedding_ref is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sequence 与 embedding_ref 必须二选一（恰好提供一个）",
        )


def _prepare_batch(
    sample: CrossScalePredictRequest, model: CrossScalePTM2CellNet
) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    """把单个请求样本转成模型 batch dict（含契约校验，显式失败）。"""
    _validate_single_input(sample)
    batch: Dict[str, Any] = {}
    if sample.sequence is not None:
        batch["sequence"] = [sample.sequence]
        protein_node_count = len(sample.sequence)
    else:
        batch.update(_load_embedding_inputs(sample.embedding_ref, sample.sample_index, model))
        first_embedding = next(iter(batch.values()))
        protein_node_count = int(first_embedding.shape[1])
    if sample.ptm_sites:
        ptm_types, ptm_positions = _ptm_tensors(sample.ptm_sites, model.cross_scale_config.num_ptm_types)
        batch["ptm_types"] = ptm_types
        batch["ptm_positions"] = ptm_positions
    # cell graph 契约：模型要求显式提供（不可静默替换）。优先级为
    # 单样本 NPZ 自带图 > initialize 登记的服务端默认图；两者皆无时
    # 显式 400，绝不静默近似。
    _apply_default_graph(batch, protein_node_count)
    extra_flags = _ensure_signal_gene_map(
        batch,
        model,
        allow_uniform=sample.allow_uniform_signal_map,
        protein_node_count=protein_node_count,
        batch_size=1,
    )
    return batch, extra_flags


def _apply_default_graph(batch: Dict[str, Any], protein_node_count: int) -> None:
    """Fill in the server-side default graph when the batch lacks a cell graph.

    ``signal_gene_map`` 的 N_signal 维与请求数据的蛋白质节点数耦合（序列路径
    = 序列长度，embedding 路径 = token 长度），静态默认图形状不匹配时跳过
    该键（其余图结构仍生效），由后续 ``_ensure_signal_gene_map`` 处理。
    """
    if "cell_edge_index" not in batch and CROSS_SCALE_STATE.default_graph:
        for key, value in CROSS_SCALE_STATE.default_graph.items():
            if key not in batch:
                if key == "signal_gene_map" and value.shape[0] != protein_node_count:
                    continue
                batch[key] = value
    if "cell_edge_index" not in batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "缺少 cell graph：请求未携带图数组，且服务端未在 initialize 时登记默认图"
                "（graph_ref）。cell graph 不可静默替换，请提供 graph_ref 或使用自带图数组"
                "的跨尺度 NPZ。"
            ),
        )


def _ensure_signal_gene_map(
    batch: Dict[str, Any],
    model: CrossScalePTM2CellNet,
    *,
    allow_uniform: bool,
    protein_node_count: int,
    batch_size: int,
    valid_lengths: Optional[List[int]] = None,
) -> Dict[str, bool]:
    """保证 batch 携带模型要求的 signal_gene_map，返回额外 fallback 标记。

    优先级：NPZ/默认图自带映射 > 显式请求的均匀映射（``allow_uniform``，
    标记 ``signal_map_uniform=True``）> 显式 400。模型 config 以
    ``require_signal_gene_map=False`` 显式关闭该要求时不做任何构造。
    均匀映射对 padding 节点赋零权重（``valid_lengths``），等价于有效残基均值。
    """
    if batch.get("signal_gene_map") is not None or not model.cell_head.require_signal_gene_map:
        return {}
    if not allow_uniform:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "缺少 signal_gene_map（signal→gene 映射，N_signal 需匹配本次请求的蛋白质"
                "节点数）：请在 embedding_ref 使用自带映射的跨尺度 NPZ，或显式设置 "
                "allow_uniform_signal_map=true（均匀映射，无信息退化，将在 fallback_flags 标记）。"
            ),
        )
    num_genes = model.cross_scale_config.num_cell_genes
    lengths = valid_lengths or [protein_node_count] * batch_size
    mapping = torch.zeros((batch_size, protein_node_count, num_genes), dtype=torch.float32)
    for row, length in enumerate(lengths):
        mapping[row, :length, :] = 1.0 / float(length)
    batch["signal_gene_map"] = mapping
    return {"signal_map_uniform": True}


def _load_embedding_inputs(
    embedding_ref: str, sample_index: int, model: CrossScalePTM2CellNet
) -> Dict[str, Any]:
    """从 embedding NPZ 提取单样本输入（含自带图数组）并校验 backbone 契约。"""
    path = _validate_path_within_allowed(Path(embedding_ref))
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"embedding 文件不存在: {embedding_ref}",
        )
    try:
        payload = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无法解析 embedding NPZ {embedding_ref}: {exc}",
        ) from exc

    with payload:
        schema = str(payload["schema_version"]) if "schema_version" in payload else ""
        if schema not in ("", _CROSS_SCALE_NPZ_SCHEMA, _EMBEDDING_CACHE_SCHEMA):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"embedding NPZ schema 不受支持: {schema}",
            )
        inputs: Dict[str, Any] = {}
        attention: Optional[torch.Tensor] = None
        for name in model.protein_encoder.backbone_names:
            key = f"{name}_embeddings"
            if key not in payload:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"embedding NPZ 缺少模型必需的 {key} 数组（backbone={name}）",
                )
            array = np.asarray(payload[key])
            if array.ndim != 3:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{key} 必须是 [N, L, D] 三维数组，实际 ndim={array.ndim}",
                )
            if sample_index >= array.shape[0]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"sample_index={sample_index} 超出 {key} 样本数 {array.shape[0]}",
                )
            expected_dim = model.protein_encoder.backbone_dims.get(name)
            if expected_dim is not None and array.shape[-1] != expected_dim:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"{key} 最后一维 {array.shape[-1]} 与模型 backbone dim {expected_dim} 不一致"
                    ),
                )
            inputs[key] = torch.from_numpy(array[sample_index: sample_index + 1].copy()).float()
            mask_key = f"{name}_attention_mask"
            if mask_key in payload:
                mask = np.asarray(payload[mask_key])
                if mask.ndim == 2 and sample_index < mask.shape[0]:
                    attention = torch.from_numpy(mask[sample_index: sample_index + 1].copy()).float()
        if attention is not None:
            inputs["protein_attention_mask"] = attention
        # 跨尺度 NPZ（区别于 embedding cache）可自带图结构，单样本优先使用。
        if schema == _CROSS_SCALE_NPZ_SCHEMA:
            inputs.update(_read_npz_graph_arrays(payload))
    return inputs


def _build_prediction(
    sample: CrossScalePredictRequest,
    outputs: Dict[str, torch.Tensor],
    row: int,
    model: CrossScalePTM2CellNet,
    *,
    graph_supplied: bool,
    extra_flags: Optional[Dict[str, bool]] = None,
) -> CrossScalePrediction:
    """把模型输出行拆包为 API 响应（单样本与组批共享）。"""
    logits = outputs["cell_state_logits"].detach().cpu().float()[row]
    probabilities = torch.softmax(logits, dim=-1)
    index = int(probabilities.argmax())
    fallback_flags = _fallback_flags(
        model,
        graph_supplied=graph_supplied,
        sequence_mode=sample.sequence is not None,
    )
    fallback_flags.update(extra_flags or {})
    return CrossScalePrediction(
        sample_id=sample.effective_sample_id,
        cell_state=CROSS_SCALE_STATE.labels[index],
        confidence=float(probabilities[index]),
        probabilities={
            CROSS_SCALE_STATE.labels[i]: float(probabilities[i])
            for i in range(len(CROSS_SCALE_STATE.labels))
        },
        cell_state_logits=[float(value) for value in logits.tolist()],
        delta_expression=(
            [float(value) for value in outputs["delta_expression"].detach().cpu().float()[row].tolist()]
            if sample.include_delta_expression
            else None
        ),
        num_cell_genes=model.cross_scale_config.num_cell_genes,
        ptm_sites_applied=len(sample.ptm_sites),
        fallback_flags=fallback_flags,
        provenance={
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_type": "cross_scale",
            "manifest_digest": CROSS_SCALE_STATE.manifest_digest,
            "artifact_path": CROSS_SCALE_STATE.artifact_path,
            "signal_graph_version": model.signal_graph_version,
            "data_manifest_digest": model.data_manifest_digest,
            "ptm_type_vocabulary": list(DEFAULT_PTM_TYPES),
            "input_mode": "sequence" if sample.sequence is not None else "embedding_ref",
        },
    )


def _predict_single_sync(sample: CrossScalePredictRequest) -> CrossScalePrediction:
    """同步单样本推理（由线程池调用）。任何契约违规显式失败，绝不静默降级。"""
    model = CROSS_SCALE_STATE.model
    assert model is not None  # 由端点层 _require_model 保证
    batch, extra_flags = _prepare_batch(sample, model)
    try:
        with torch.inference_mode():
            outputs = model(batch)
    except CrossScaleContractError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"跨尺度推理契约违规: {exc}",
        ) from exc
    return _build_prediction(
        sample,
        outputs,
        0,
        model,
        graph_supplied="signal_edge_index" in batch,
        extra_flags=extra_flags,
    )


@router.post("/cross-scale/predict", response_model=CrossScalePredictResponse)
async def cross_scale_predict(request: CrossScalePredictRequest) -> CrossScalePredictResponse:
    """单样本跨尺度预测。"""
    _require_model()
    started = time.perf_counter()
    prediction = await run_in_threadpool(_predict_single_sync, request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response = CrossScalePredictResponse(**prediction.model_dump())
    response.processing_time_ms = round(elapsed_ms, 3)
    return response


def _predict_plain_sequence_chunk_sync(
    chunk: List[CrossScalePredictRequest],
) -> List[CrossScalePrediction]:
    """组批推理一组"纯序列 + 无 PTM"样本（模型原生支持序列列表）。"""
    model = CROSS_SCALE_STATE.model
    assert model is not None
    sequences = [sample.sequence for sample in chunk]
    batch: Dict[str, Any] = {"sequence": sequences}
    node_count = max(len(sequence) for sequence in sequences)
    _apply_default_graph(batch, node_count)
    extra_flags = _ensure_signal_gene_map(
        batch,
        model,
        allow_uniform=any(sample.allow_uniform_signal_map for sample in chunk),
        protein_node_count=node_count,
        batch_size=len(chunk),
        valid_lengths=[len(sequence) for sequence in sequences],
    )
    with torch.inference_mode():
        outputs = model(batch)
    return [
        _build_prediction(
            sample,
            outputs,
            row,
            model,
            graph_supplied="signal_edge_index" in batch,
            extra_flags=extra_flags,
        )
        for row, sample in enumerate(chunk)
    ]


@router.post("/cross-scale/batch_predict", response_model=CrossScaleBatchPredictResponse)
async def cross_scale_batch_predict(
    request: CrossScaleBatchPredictRequest,
) -> CrossScaleBatchPredictResponse:
    """批量跨尺度预测。

    "纯序列且无 PTM 位点"的样本按 ``batch_size`` 组批共享一次 forward；
    其余样本（带 PTM 位点或 embedding 引用）逐个推理。失败策略由
    ``fail_fast`` 控制：False（默认）逐样本收集 errors 并返回部分成功
    结果，保证可审计（契约）；True 时首个错误即中止并返回 400。
    """
    _require_model()
    model = CROSS_SCALE_STATE.model
    assert model is not None
    batch_size = min(request.batch_size, CROSS_SCALE_STATE.max_batch_size)

    plain_positions: List[int] = []
    other_positions: List[int] = []
    for position, sample in enumerate(request.samples):
        try:
            _validate_single_input(sample)
        except HTTPException as exc:
            if request.fail_fast:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"样本 {position} ({sample.effective_sample_id}): {exc.detail}",
                ) from exc
            other_positions.append(position)  # 交由单样本路径记录错误
            continue
        if sample.sequence is not None and not sample.ptm_sites:
            plain_positions.append(position)
        else:
            other_positions.append(position)

    results: Dict[int, CrossScalePrediction] = {}
    errors: List[CrossScaleBatchError] = []

    # 纯序列组批路径：chunk 级失败（如序列编码契约违规）影响整个 chunk。
    for start in range(0, len(plain_positions), batch_size):
        chunk_positions = plain_positions[start: start + batch_size]
        chunk = [request.samples[position] for position in chunk_positions]
        try:
            chunk_predictions = await run_in_threadpool(
                _predict_plain_sequence_chunk_sync, chunk
            )
        except CrossScaleContractError as exc:
            detail = f"跨尺度推理契约违规: {exc}"
        except HTTPException as exc:
            detail = str(exc.detail)
        else:
            for position, prediction in zip(chunk_positions, chunk_predictions, strict=True):
                results[position] = prediction
            continue
        if request.fail_fast:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"样本组 {chunk_positions}: {detail}",
            )
        for position in chunk_positions:
            sample = request.samples[position]
            errors.append(
                CrossScaleBatchError(
                    index=position, sample_id=sample.effective_sample_id, detail=detail
                )
            )

    # 其余样本（embedding 引用 / 带 PTM / 输入非法）逐个推理并隔离失败。
    for position in other_positions:
        sample = request.samples[position]
        try:
            results[position] = await run_in_threadpool(_predict_single_sync, sample)
        except HTTPException as exc:
            if request.fail_fast:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"样本 {position} ({sample.effective_sample_id}): {exc.detail}",
                ) from exc
            errors.append(
                CrossScaleBatchError(
                    index=position, sample_id=sample.effective_sample_id, detail=str(exc.detail)
                )
            )

    predictions = [results[position] for position in sorted(results)]
    return CrossScaleBatchPredictResponse(
        predictions=predictions,
        errors=errors,
        summary=CrossScaleBatchSummary(
            total=len(request.samples),
            succeeded=len(predictions),
            failed=len(errors),
        ),
        provenance={
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_type": "cross_scale",
            "manifest_digest": CROSS_SCALE_STATE.manifest_digest,
            "artifact_path": CROSS_SCALE_STATE.artifact_path,
            "batch_size": batch_size,
            "fail_fast": request.fail_fast,
            "batched_sequence_chunks": (len(plain_positions) + batch_size - 1) // max(batch_size, 1),
        },
    )


__all__ = [
    "router",
    "CROSS_SCALE_STATE",
    "reset_cross_scale_state",
    "PTM_TYPE_TO_IDX",
    "CrossScaleInitializeRequest",
    "CrossScaleInitializeResponse",
    "CrossScalePredictRequest",
    "CrossScalePredictResponse",
    "CrossScaleBatchPredictRequest",
    "CrossScaleBatchPredictResponse",
]
