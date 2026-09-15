from __future__ import annotations

import json

from scripts.merge_davf_perturbgen_reports import main


def _report(route: str, path):
    path.write_text(
        json.dumps(
            {
                "intervention_type": route,
                "davf_config": f"configs/davf_{route.lower()}.yaml",
                "context_h5ad": f"{route.lower()}.h5ad",
                "candidates": [
                    {
                        "invocation": {
                            "intervention_type": route,
                            "gene_symbol": "STAT3",
                            "ensembl_id": "ENSG00000168610",
                        }
                    }
                ],
                "perturbgen_runs": [{"ensembl_id": "ENSG00000168610", "output_root": f"{route.lower()}-out"}],
            }
        ),
        encoding="utf-8",
    )


def test_merge_report_cli_writes_route_separated_ensembl_record(tmp_path):
    ko = tmp_path / "ko.json"
    kd = tmp_path / "kd.json"
    output = tmp_path / "merged.json"
    _report("KO", ko)
    _report("KD", kd)

    assert main(["--reports", str(ko), str(kd), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "davf_perturbgen_e2e/merged/v1"
    assert payload["candidates"][0]["ensembl_id"] == "ENSG00000168610"
    assert set(payload["candidates"][0]["routes"]) == {"KO", "KD"}
