"""CLI checks for registering a local data snapshot without copying it."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_update_and_validate_manifest_cli(tmp_path):
    manifest = tmp_path / "datasets.yaml"
    shutil.copy(PROJECT_ROOT / "data/manifests/datasets.yaml", manifest)
    snapshot = tmp_path / "fixture.csv"
    snapshot.write_text("sequence,label\nACDE,1\n", encoding="utf-8")

    updated = subprocess.run(
        [
            sys.executable,
            "scripts/update_data_manifest.py",
            "--manifest",
            str(manifest),
            "--dataset",
            "pmads",
            "--path",
            str(snapshot),
            "--format",
            "csv",
            "--required",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert updated.returncode == 0, updated.stderr
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    pmads = next(item for item in payload["datasets"] if item["id"] == "pmads")
    assert pmads["status"] == "local"
    assert pmads["files"][0]["sha256"]

    # The working copy carries required local snapshots (e.g. the 26 scperturb
    # h5ad) whose relative paths resolve against the real repo, not the tmp
    # copy. Downgrade every non-pmads required file so --check-files only
    # validates the file this test registers (absolute path in tmp).
    for dataset in payload["datasets"]:
        if dataset["id"] == "pmads":
            continue
        for file_entry in dataset.get("files", []):
            if isinstance(file_entry, dict):
                file_entry["required"] = False
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    checked = subprocess.run(
        [
            sys.executable,
            "scripts/validate_data_manifest.py",
            "--manifest",
            str(manifest),
            "--check-files",
            "--verify-hashes",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    report = json.loads(checked.stdout)
    assert report["ok"] is True


def test_update_manifest_preserves_distinct_snapshot_entries(tmp_path):
    manifest = tmp_path / "datasets.yaml"
    shutil.copy(PROJECT_ROOT / "data/manifests/datasets.yaml", manifest)
    first_snapshot = tmp_path / "first.csv"
    second_snapshot = tmp_path / "second.csv"
    first_snapshot.write_text("sequence,label\nACDE,1\n", encoding="utf-8")
    second_snapshot.write_text("sequence,label\nFGHI,0\n", encoding="utf-8")

    for snapshot in (first_snapshot, second_snapshot):
        updated = subprocess.run(
            [
                sys.executable,
                "scripts/update_data_manifest.py",
                "--manifest",
                str(manifest),
                "--dataset",
                "pmads",
                "--path",
                str(snapshot),
                "--format",
                "csv",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert updated.returncode == 0, updated.stderr

    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    pmads = next(item for item in payload["datasets"] if item["id"] == "pmads")
    registered_paths = {entry["path"] for entry in pmads["files"]}
    assert {str(first_snapshot.resolve()), str(second_snapshot.resolve())}.issubset(registered_paths)


@pytest.mark.slow
def test_validate_manifest_cli_profile_standard_training_passes():
    """--profile standard_training 应通过（pmads 有本地快照）"""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_data_manifest.py",
            "--profile",
            "standard_training",
            "--check-files",
            "--verify-hashes",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    assert report["profile"] == "standard_training"
    assert "pmads" in report["profile_required_datasets"]


def test_validate_manifest_cli_profile_cross_scale_fails_fast():
    """--profile cross_scale_training 在关键图资产缺失时应 fail-fast"""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_data_manifest.py",
            "--profile",
            "cross_scale_training",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["ok"] is False
    assert any("required by profile" in error for error in report["errors"])


def test_validate_manifest_cli_unknown_profile_fails():
    """未知 profile 名必须显式失败"""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_data_manifest.py",
            "--profile",
            "not_a_profile",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["ok"] is False
    assert any("is not declared" in error for error in report["errors"])


@pytest.mark.slow
def test_validate_manifest_cli_without_profile_keeps_legacy_pass():
    """不带 --profile 时默认校验仍通过（向后兼容）"""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_data_manifest.py",
            "--check-files",
            "--strict",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    assert "profile" not in report
