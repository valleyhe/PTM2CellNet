"""Tests for the offline pLM asset gate."""

import json
from pathlib import Path

import pytest

from src.models.plm_assets import (
    PLMAssetError,
    inspect_plm_asset,
    require_local_plm_assets,
    resolve_local_plm_assets,
)


def _asset(root: Path, name: str, *, complete: bool = True) -> Path:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "esm", "hidden_size": 8}), encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    weight_name = "model.safetensors" if complete else "model.safetensors.part"
    (path / weight_name).write_bytes(b"weights")
    return path


def test_inspect_plm_asset_accepts_complete_directory(tmp_path):
    report = inspect_plm_asset(_asset(tmp_path, "complete"))

    assert report["ok"] is True
    assert report["model_type"] == "esm"
    assert report["weight_files"] == ["model.safetensors"]


def test_inspect_plm_asset_rejects_partial_download(tmp_path):
    report = inspect_plm_asset(_asset(tmp_path, "partial", complete=False))

    assert report["ok"] is False
    assert report["partial_files"] == ["model.safetensors.part"]
    assert any("未完成下载" in error for error in report["errors"])


def test_resolve_and_require_local_assets_are_fail_fast(tmp_path):
    _asset(tmp_path, "ankh")
    _asset(tmp_path, "esm", complete=False)
    report = resolve_local_plm_assets(
        tmp_path,
        model_dirs={"ankh39": "ankh", "esm2": "esm"},
        required_backbones=("ankh39", "esm2"),
    )

    assert report["ok"] is False
    assert set(report["model_names"]) == {"ankh39"}
    with pytest.raises(PLMAssetError, match="esm2"):
        require_local_plm_assets(
            tmp_path,
            model_dirs={"ankh39": "ankh", "esm2": "esm"},
            required_backbones=("ankh39", "esm2"),
        )
