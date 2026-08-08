"""CLI coverage for the local pLM asset gate."""

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_asset(root: Path, name: str, *, partial: bool = False) -> None:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type":"esm"}', encoding="utf-8")
    (path / "vocab.txt").write_text("A\n", encoding="utf-8")
    (path / ("pytorch_model.bin.part" if partial else "pytorch_model.bin")).write_bytes(b"x")


def test_validate_plm_assets_cli_accepts_complete_required_set(tmp_path):
    for name in ("ankh", "esm", "t5"):
        _write_asset(tmp_path, name)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_plm_assets.py",
            "--asset-root",
            str(tmp_path),
            "--model-dir",
            "ankh39=ankh",
            "--model-dir",
            "esm2=esm",
            "--model-dir",
            "prott5=t5",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["ok"] is True


def test_validate_plm_assets_cli_rejects_partial_required_weight(tmp_path):
    _write_asset(tmp_path, "ankh")
    _write_asset(tmp_path, "esm", partial=True)
    _write_asset(tmp_path, "t5")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_plm_assets.py",
            "--asset-root",
            str(tmp_path),
            "--model-dir",
            "ankh39=ankh",
            "--model-dir",
            "esm2=esm",
            "--model-dir",
            "prott5=t5",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["ok"] is False
    assert any("esm2" in error for error in report["errors"])
