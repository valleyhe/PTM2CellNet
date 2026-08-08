"""Local protein-language-model asset discovery and integrity checks.

The cross-scale path is intentionally offline-first.  A directory is only
considered loadable when it contains a parseable Hugging Face config, tokenizer
assets, and a complete weight file (or every shard named by an index).  Partial
``*.part`` downloads are reported explicitly and are never passed to
``transformers`` as if they were valid checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_LOCAL_PLM_DIRS: Dict[str, str] = {
    "ankh39": "ankh-base",
    "esm2": "esm2_t30_150M_UR50D",
    "prott5": "prot_t5_xl_half_uniref50-enc",
}

_TOKENIZER_FILES = (
    "tokenizer.json",
    "spiece.model",
    "vocab.txt",
    "vocab.json",
)
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_WEIGHT_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


class PLMAssetError(ValueError):
    """Raised when a required local pLM asset is missing or incomplete."""


def _json_mapping(path: Path, errors: list[str]) -> Optional[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"无法解析 JSON {path.name}: {exc}")
        return None
    if not isinstance(payload, Mapping):
        errors.append(f"{path.name} 顶层必须是对象")
        return None
    return payload


def inspect_plm_asset(path: str | Path) -> Dict[str, Any]:
    """Inspect one local Hugging Face model directory without loading weights."""

    candidate = Path(path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    weight_files: list[str] = []
    partial_files: list[str] = []

    if not candidate.is_dir():
        errors.append(f"模型目录不存在: {candidate}")
        return {
            "ok": False,
            "path": str(candidate),
            "errors": errors,
            "warnings": warnings,
            "weight_files": weight_files,
            "partial_files": partial_files,
        }

    config_path = candidate / "config.json"
    config = None
    if not config_path.is_file():
        errors.append("缺少 config.json")
    else:
        config = _json_mapping(config_path, errors)

    tokenizer_files = [name for name in _TOKENIZER_FILES if (candidate / name).is_file()]
    if not tokenizer_files:
        errors.append("缺少 tokenizer.json/spiece.model/vocab.txt 等 tokenizer 资产")

    for name in _WEIGHT_FILES:
        weight_path = candidate / name
        if weight_path.is_file() and weight_path.stat().st_size > 0:
            weight_files.append(name)

    for index_name in _WEIGHT_INDEX_FILES:
        index_path = candidate / index_name
        if not index_path.is_file():
            continue
        index = _json_mapping(index_path, errors)
        if index is None:
            continue
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            errors.append(f"{index_name}.weight_map 必须是非空对象")
            continue
        shards = sorted({str(value) for value in weight_map.values()})
        missing_shards = [name for name in shards if not (candidate / name).is_file()]
        empty_shards = [
            name
            for name in shards
            if (candidate / name).is_file() and (candidate / name).stat().st_size <= 0
        ]
        if missing_shards:
            errors.append(f"{index_name} 引用缺失分片: {missing_shards}")
        if empty_shards:
            errors.append(f"{index_name} 引用空分片: {empty_shards}")
        if not missing_shards and not empty_shards:
            weight_files.extend(shards)

    partial_files = sorted(
        item.name
        for item in candidate.iterdir()
        if item.is_file() and (item.name.endswith(".part") or item.name.endswith(".tmp"))
    )
    if not weight_files:
        errors.append("缺少完整 model.safetensors/pytorch_model.bin 或完整分片权重")
    if partial_files:
        message = f"发现未完成下载文件: {partial_files}"
        if weight_files:
            warnings.append(message)
        else:
            errors.append(message)

    return {
        "ok": not errors,
        "path": str(candidate),
        "model_type": config.get("model_type") if config is not None else None,
        "architectures": list(config.get("architectures", []) or []) if config is not None else [],
        "tokenizer_files": tokenizer_files,
        "weight_files": sorted(set(weight_files)),
        "partial_files": partial_files,
        "errors": errors,
        "warnings": warnings,
    }


def resolve_local_plm_assets(
    asset_root: str | Path,
    *,
    model_dirs: Optional[Mapping[str, str]] = None,
    required_backbones: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Resolve configured backbone names to validated absolute local paths."""

    root = Path(asset_root).expanduser().resolve()
    selected = {str(key).lower(): str(value) for key, value in (model_dirs or DEFAULT_LOCAL_PLM_DIRS).items()}
    required = tuple(str(name).lower() for name in (required_backbones or tuple(selected)))
    unknown = sorted(set(required) - set(selected))
    errors = [f"必需 backbone 未配置目录: {unknown}"] if unknown else []
    assets: Dict[str, Dict[str, Any]] = {}
    resolved_model_names: Dict[str, str] = {}
    for name, directory in selected.items():
        path = Path(directory)
        if not path.is_absolute():
            path = root / path
        report = inspect_plm_asset(path)
        report["required"] = name in required
        assets[name] = report
        if report["ok"]:
            resolved_model_names[name] = report["path"]
        elif name in required:
            errors.extend(f"{name}: {message}" for message in report["errors"])
    return {
        "ok": not errors,
        "asset_root": str(root),
        "required_backbones": list(required),
        "model_names": resolved_model_names,
        "assets": assets,
        "errors": errors,
    }


def require_local_plm_assets(
    asset_root: str | Path,
    *,
    model_dirs: Optional[Mapping[str, str]] = None,
    required_backbones: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Return validated model paths or raise one actionable aggregate error."""

    report = resolve_local_plm_assets(
        asset_root,
        model_dirs=model_dirs,
        required_backbones=required_backbones,
    )
    if not report["ok"]:
        raise PLMAssetError("本地 pLM 资产校验失败: " + "; ".join(report["errors"]))
    return dict(report["model_names"])


__all__ = [
    "DEFAULT_LOCAL_PLM_DIRS",
    "PLMAssetError",
    "inspect_plm_asset",
    "resolve_local_plm_assets",
    "require_local_plm_assets",
]
