"""Minimal contract tests for the external KSTAR boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.analysis.kstar_adapter import (
    KSTARAdapterError,
    build_kstar_input,
    convert_kstar_outputs,
    map_gene_symbols_to_ensembl,
    read_kstar_result,
)
from src.analysis.ptm_activity import ACTIVITY_REQUIRED_COLUMNS
from scripts.run_kstar_activity import _normalize_kstar_handover


def _manifest(tmp_path):
    path = tmp_path / "ptm_input_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ptm2cellnet.ptm-input-manifest/v1",
                "source": {"file": "ptm_site_quantification.tsv", "sha256": "0" * 64},
                "standardization": {"n_rows_input": 13},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _standardized_frame():
    rows = []
    for donor, suffix in (("D1", "1"), ("D2", "2")):
        rows.extend(
            [
                {
                    "sample_id": f"N_S9_{suffix}",
                    "donor_id": donor,
                    "condition": "normal",
                    "protein_id": "P49841",
                    "residue": "S9",
                    "ptm_type": "phosphorylation",
                    "ptm_value_normalized": 1.0,
                    "value_scale": "linear",
                    "source_dataset": "PTM_TEST",
                },
                {
                    "sample_id": f"D_S9_{suffix}",
                    "donor_id": donor,
                    "condition": "disease",
                    "protein_id": "P49841",
                    "residue": "S9",
                    "ptm_type": "phosphorylation",
                    "ptm_value_normalized": 2.0,
                    "value_scale": "linear",
                    "source_dataset": "PTM_TEST",
                },
                {
                    "sample_id": f"N_T205_{suffix}",
                    "donor_id": donor,
                    "condition": "normal",
                    "protein_id": "P10636",
                    "residue": "T205",
                    "ptm_type": "phosphorylation",
                    "ptm_value_normalized": 4.0,
                    "value_scale": "linear",
                    "source_dataset": "PTM_TEST",
                },
                {
                    "sample_id": f"D_T205_{suffix}",
                    "donor_id": donor,
                    "condition": "disease",
                    "protein_id": "P10636",
                    "residue": "T205",
                    "ptm_type": "phosphorylation",
                    "ptm_value_normalized": 1.0,
                    "value_scale": "linear",
                    "source_dataset": "PTM_TEST",
                },
                {
                    "sample_id": f"N_Y123_{suffix}",
                    "donor_id": donor,
                    "condition": "normal",
                    "protein_id": "P99999",
                    "residue": "Y123",
                    "ptm_type": "phosphorylation",
                    "ptm_value_normalized": 1.0,
                    "value_scale": "linear",
                    "source_dataset": "PTM_TEST",
                },
                {
                    "sample_id": f"D_Y123_{suffix}",
                    "donor_id": donor,
                    "condition": "disease",
                    "protein_id": "P99999",
                    "residue": "Y123",
                    "ptm_type": "phosphorylation",
                    "ptm_value_normalized": 1.05,
                    "value_scale": "linear",
                    "source_dataset": "PTM_TEST",
                },
            ]
        )
    rows.append(
        {
            "sample_id": "N_NO_DONOR",
            "donor_id": "",
            "condition": "normal",
            "protein_id": "P12345",
            "residue": "S1",
            "ptm_type": "phosphorylation",
            "ptm_value_normalized": 1.0,
            "value_scale": "linear",
            "source_dataset": "PTM_TEST",
        }
    )
    return pd.DataFrame(rows)


def test_build_kstar_input_freezes_direction_and_provenance(tmp_path):
    evidence, audit = build_kstar_input(
        _standardized_frame(),
        input_manifest=_manifest(tmp_path),
        case_condition="disease",
        reference_condition="normal",
        contrast="disease-minus-normal",
    )

    increased = "data:disease-minus-normal:increased"
    decreased = "data:disease-minus-normal:decreased"
    assert dict(zip(evidence["site"], evidence["evidence_class"], strict=True)) == {
        "S9": "increased",
        "T205": "decreased",
        "Y123": "unchanged",
    }
    assert evidence.loc[evidence["site"] == "S9", increased].item() == 1
    assert evidence.loc[evidence["site"] == "S9", decreased].item() == 0
    assert evidence.loc[evidence["site"] == "T205", decreased].item() == 1
    assert evidence["input_manifest"].eq(str(_manifest(tmp_path).resolve())).all()
    assert audit.excluded_no_donor_rows == 1
    assert audit.case_donors == ("D1", "D2")


def test_non_phosphorylation_is_rejected(tmp_path):
    frame = _standardized_frame()
    frame.loc[0, "ptm_type"] = "acetylation"
    with pytest.raises(KSTARAdapterError, match="only ptm_type=phosphorylation"):
        build_kstar_input(
            frame,
            input_manifest=_manifest(tmp_path),
            case_condition="disease",
            reference_condition="normal",
            contrast="disease-minus-normal",
        )


def test_paired_kstar_outputs_produce_nonzero_signed_scores(tmp_path):
    manifest = _manifest(tmp_path)
    increased = pd.DataFrame(
        {
            "KSTAR_KINASE": ["GSK3B", "CDK5", "AKT1"],
            "data:disease-minus-normal:increased": [0.01, 0.8, 1.0],
        }
    )
    decreased = pd.DataFrame(
        {
            "KSTAR_KINASE": ["GSK3B", "CDK5", "AKT1"],
            "data:disease-minus-normal:decreased": [0.2, 0.02, 1.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "regulator_id": ["GSK3B", "CDK5", "AKT1"],
            "n_substrates": [3, 2, 1],
            "network_coverage": [0.75, 0.5, 0.2],
        }
    )

    activity = convert_kstar_outputs(
        increased,
        decreased,
        contrast="disease-minus-normal",
        method_version="1.2.0",
        input_manifest=manifest,
        increased_metrics=metrics,
        decreased_metrics=metrics,
    )
    assert tuple(activity.columns) == ACTIVITY_REQUIRED_COLUMNS
    assert set(activity["regulator_id"]) == {"GSK3B", "CDK5"}
    assert activity.set_index("regulator_id").loc["GSK3B", "activity_score"] > 0
    assert activity.set_index("regulator_id").loc["CDK5", "activity_score"] < 0
    assert (activity["activity_score"] != 0).all()
    assert activity["activity_qvalue"].between(0, 1).all()


def test_winning_direction_binds_its_own_metrics(tmp_path):
    increased = pd.DataFrame(
        {
            "KSTAR_KINASE": ["GSK3B", "CDK5"],
            "data:disease-minus-normal:increased": [0.01, 0.8],
        }
    )
    decreased = pd.DataFrame(
        {
            "KSTAR_KINASE": ["GSK3B", "CDK5"],
            "data:disease-minus-normal:decreased": [0.2, 0.02],
        }
    )
    # Direction-specific evidence sets legitimately differ (TD-01): GSK3B wins
    # increased, CDK5 wins decreased, and each row must bind its own metrics.
    increased_metrics = pd.DataFrame(
        {
            "regulator_id": ["GSK3B", "CDK5"],
            "n_substrates": [7, 1],
            "network_coverage": [0.7, 0.1],
        }
    )
    decreased_metrics = pd.DataFrame(
        {
            "regulator_id": ["GSK3B", "CDK5"],
            "n_substrates": [3, 5],
            "network_coverage": [0.3, 0.5],
        }
    )
    activity = convert_kstar_outputs(
        increased,
        decreased,
        contrast="disease-minus-normal",
        method_version="1.2.0",
        input_manifest=_manifest(tmp_path),
        increased_metrics=increased_metrics,
        decreased_metrics=decreased_metrics,
    )
    indexed = activity.set_index("regulator_id")
    assert indexed.loc["GSK3B", "activity_direction"] == "up"
    assert indexed.loc["GSK3B", "n_substrates"] == 7
    assert indexed.loc["GSK3B", "network_coverage"] == pytest.approx(0.7)
    assert indexed.loc["CDK5", "activity_direction"] == "down"
    assert indexed.loc["CDK5", "n_substrates"] == 5
    assert indexed.loc["CDK5", "network_coverage"] == pytest.approx(0.5)


def test_directional_metrics_must_be_supplied_together(tmp_path):
    increased = pd.DataFrame({"KSTAR_KINASE": ["GSK3B"], "data:disease-minus-normal:increased": [0.01]})
    decreased = pd.DataFrame({"KSTAR_KINASE": ["GSK3B"], "data:disease-minus-normal:decreased": [0.2]})
    metrics = pd.DataFrame({"regulator_id": ["GSK3B"], "n_substrates": [3], "network_coverage": [0.7]})
    with pytest.raises(KSTARAdapterError, match="together"):
        convert_kstar_outputs(
            increased,
            decreased,
            contrast="disease-minus-normal",
            method_version="1.2.0",
            input_manifest=_manifest(tmp_path),
            increased_metrics=metrics,
        )
    mismatched = pd.DataFrame({"regulator_id": ["OTHER"], "n_substrates": [1], "network_coverage": [0.1]})
    with pytest.raises(KSTARAdapterError, match="different kinase sets"):
        convert_kstar_outputs(
            increased,
            decreased,
            contrast="disease-minus-normal",
            method_version="1.2.0",
            input_manifest=_manifest(tmp_path),
            increased_metrics=metrics,
            decreased_metrics=mismatched,
        )


def test_unsigned_or_stub_result_is_rejected(tmp_path):
    unsigned = pd.DataFrame({"KSTAR_KINASE": ["GSK3B"], "p_value": [0.01]})
    with pytest.raises(KSTARAdapterError, match="direction is required"):
        read_kstar_result(unsigned, direction=None, contrast="disease-minus-normal")
    with pytest.raises(KSTARAdapterError, match="real KSTAR 1.2.x"):
        convert_kstar_outputs(
            pd.DataFrame({"KSTAR_KINASE": ["GSK3B"], "data:disease-minus-normal:increased": [0.01]}),
            pd.DataFrame({"KSTAR_KINASE": ["GSK3B"], "data:disease-minus-normal:decreased": [0.2]}),
            contrast="disease-minus-normal",
            method_version="smoke-stub-20260920",
            input_manifest=_manifest(tmp_path),
            increased_metrics=pd.DataFrame({"regulator_id": ["GSK3B"], "n_substrates": [1], "network_coverage": [1.0]}),
            decreased_metrics=pd.DataFrame({"regulator_id": ["GSK3B"], "n_substrates": [1], "network_coverage": [1.0]}),
        )


def test_unnamed_kstar_index_is_normalized_before_from_kstar_read(tmp_path):
    contrast = "disease-minus-normal"
    data_column = f"data:{contrast}:increased"
    malformed = tmp_path / "malformed.tsv"
    malformed.write_text(f"\t{data_column}\nGSK3B\t0.01\n", encoding="utf-8")
    with pytest.raises(KSTARAdapterError, match="missing KSTAR_KINASE"):
        read_kstar_result(malformed, direction="increased", contrast=contrast)

    activity = pd.DataFrame({data_column: [0.01]}, index=pd.Index(["GSK3B"]))
    fpr = pd.DataFrame({data_column: [0.02]}, index=pd.Index(["GSK3B"]))
    kinact = SimpleNamespace(activities_mann_whitney=activity, fpr_mann_whitney=fpr)
    saved: dict[str, object] = {}

    class FakeConfig:
        KSTAR_KINASE = "KSTAR_KINASE"

    class FakeCalculate:
        @staticmethod
        def save_kstar(kinact_dict, name, odir, **kwargs):
            saved["args"] = (name, odir, kwargs)
            result_dir = Path(odir) / "RESULTS" / "ST"
            result_dir.mkdir(parents=True)
            obj = kinact_dict["ST"]
            obj.activities_mann_whitney.to_csv(
                result_dir / f"{name}_ST_mann_whitney_activities.tsv", sep="\t", index=True
            )
            obj.fpr_mann_whitney.to_csv(result_dir / f"{name}_ST_mann_whitney_fpr.tsv", sep="\t", index=True)

    _normalize_kstar_handover(
        {"ST": kinact},
        phospho_type="ST",
        run_name="smoke_increased",
        odir=tmp_path,
        data_column=data_column,
        kstar_calculate=FakeCalculate,
        kstar_config=FakeConfig,
    )
    assert activity.index.name == "KSTAR_KINASE"
    assert fpr.index.name == "KSTAR_KINASE"
    assert saved["args"][0] == "smoke_increased"
    result = read_kstar_result(
        tmp_path / "RESULTS" / "ST" / "smoke_increased_ST_mann_whitney_activities.tsv",
        direction="increased",
        contrast=contrast,
    )
    assert result.to_dict("records") == [{"regulator_id": "GSK3B", "activity_pvalue": 0.01}]


def _ensembl_map(tmp_path: Path) -> Path:
    path = tmp_path / "gene_symbol_ensembl_map.tsv"
    path.write_text("ensembl_id\tgene_symbol\nENSG00000082701\tGSK3B\nENSG00000164885\tCDK5\n", encoding="utf-8")
    return path


def test_map_gene_symbols_to_ensembl_rejects_missing_and_conflicts(tmp_path):
    mapping = _ensembl_map(tmp_path)
    assert map_gene_symbols_to_ensembl(["GSK3B"], mapping)["GSK3B"] == "ENSG00000082701"
    with pytest.raises(KSTARAdapterError, match="missing"):
        map_gene_symbols_to_ensembl(["NOTAKINASE"], mapping)
    conflict = tmp_path / "conflict.tsv"
    conflict.write_text(
        "ensembl_id\tgene_symbol\nENSG00000082701\tGSK3B\nENSG00000082702\tGSK3B\n",
        encoding="utf-8",
    )
    with pytest.raises(KSTARAdapterError, match="conflicts"):
        map_gene_symbols_to_ensembl(["GSK3B"], conflict)


def test_convert_kstar_outputs_maps_regulators_to_ensembl(tmp_path):
    activity = convert_kstar_outputs(
        pd.DataFrame({"KSTAR_KINASE": ["GSK3B", "CDK5"], "data:disease-minus-normal:increased": [0.01, 0.8]}),
        pd.DataFrame({"KSTAR_KINASE": ["GSK3B", "CDK5"], "data:disease-minus-normal:decreased": [0.2, 0.02]}),
        contrast="disease-minus-normal",
        method_version="1.2.0",
        input_manifest=_manifest(tmp_path),
        increased_metrics=pd.DataFrame(
            {"regulator_id": ["GSK3B", "CDK5"], "n_substrates": [3, 2], "network_coverage": [0.75, 0.5]}
        ),
        decreased_metrics=pd.DataFrame(
            {"regulator_id": ["GSK3B", "CDK5"], "n_substrates": [3, 2], "network_coverage": [0.75, 0.5]}
        ),
        ensembl_mapping=_ensembl_map(tmp_path),
    )
    assert set(activity["regulator_id"]) == {"ENSG00000082701", "ENSG00000164885"}
    assert activity.set_index("regulator_id").loc["ENSG00000082701", "activity_score"] > 0
    assert activity.set_index("regulator_id").loc["ENSG00000164885", "activity_score"] < 0


def test_structured_input_manifest_is_required(tmp_path):
    empty_manifest = tmp_path / "empty_manifest.json"
    empty_manifest.write_text(json.dumps({}) + "\n", encoding="utf-8")
    with pytest.raises(KSTARAdapterError, match="schema_version"):
        build_kstar_input(
            _standardized_frame(),
            input_manifest=empty_manifest,
            case_condition="disease",
            reference_condition="normal",
            contrast="disease-minus-normal",
        )


def test_min_donors_per_state_gate(tmp_path):
    frame = _standardized_frame()
    # The fixture has two donors (D1, D2) per condition; raising the floor to 3
    # must hard-fail before any KSTAR evidence is produced.
    with pytest.raises(KSTARAdapterError, match="donors with usable evidence"):
        build_kstar_input(
            frame,
            input_manifest=_manifest(tmp_path),
            case_condition="disease",
            reference_condition="normal",
            contrast="disease-minus-normal",
            min_donors_per_state=3,
        )
    evidence, audit = build_kstar_input(
        frame,
        input_manifest=_manifest(tmp_path),
        case_condition="disease",
        reference_condition="normal",
        contrast="disease-minus-normal",
        min_donors_per_state=2,
    )
    assert not evidence.empty
    assert audit.case_donors == ("D1", "D2")
    with pytest.raises(KSTARAdapterError, match="min_donors_per_state"):
        build_kstar_input(
            frame,
            input_manifest=_manifest(tmp_path),
            case_condition="disease",
            reference_condition="normal",
            contrast="disease-minus-normal",
            min_donors_per_state=0,
        )
