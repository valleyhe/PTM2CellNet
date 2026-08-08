"""Offline CLI smoke test for the PMADS Ridge baseline."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_baseline_cli_writes_reproducible_artifacts(tmp_path):
    rows = []
    for index in range(30):
        rows.append(
            {
                "sequence": "ACDEFGHIKLMNPQRSTVWY"[: 8 + index % 5],
                "ptm_sites": "[]" if index % 2 else '[{"position": 2, "type": "phosphorylation"}]',
                "label": index % 2,
                "protein_accession": f"P{index // 3:05d}",
            }
        )
    input_path = tmp_path / "pmads_fixture.csv"
    output_dir = tmp_path / "artifact"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/baseline_pmads_ridge.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--manifest",
            "data/manifests/datasets.yaml",
            "--group-col",
            "protein_accession",
            "--demo-data",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output_dir / "baseline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_digest"]
    assert manifest["split"]["strategy"] == "group_shuffle_split"
    assert (output_dir / "ridge_model.joblib").exists()
