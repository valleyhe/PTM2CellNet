"""CLI checks for registering a local data snapshot without copying it."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


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
