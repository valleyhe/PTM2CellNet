"""N-1/N-2 candidate-spec builder tests (PTM CSV → proposal + evidence join)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.integration.perturbgen.candidate_spec import (
    CandidateSpecError,
    build_candidate_spec_payload,
    build_spec_candidates,
    load_direction_evidence,
    load_direction_map,
    load_gene_map,
    load_ptm_site_predictions,
    write_candidate_spec,
)
from src.integration.perturbgen.contracts import PTMSiteDirectionProposal


GENE = "STAT3"
ENSEMBL = "ENSG00000168610"
PROTEIN = "P40789"


def _predictions_frame(**overrides) -> pd.DataFrame:
    base = {
        "protein_id": [PROTEIN, PROTEIN, "P12345"],
        "position": [12, 705, 3],
        "aa": ["S", "S", "K"],
        "ptm_type": ["phosphorylation", "phosphorylation", "acetylation"],
        "probability": [0.95, 0.30, 0.99],
        "sequence_window": ["x", "y", "z"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _gene_map() -> dict[str, tuple[str, str]]:
    return {PROTEIN: (GENE, ENSEMBL)}


def _evidence_frame(rows=None) -> pd.DataFrame:
    if rows is None:
        rows = [
            {
                "cell_type": "K562",
                "ensembl_id": ENSEMBL,
                "gene_symbol": GENE,
                "log2fc": -1.2,
                "p_value": 0.001,
                "fdr": 0.01,
                "observed_direction": "down",
                "n_normal_donors": 4,
                "n_disease_donors": 4,
            }
        ]
    return pd.DataFrame(rows)


class TestLoaders:
    def test_load_predictions_rejects_missing_columns(self, tmp_path):
        path = tmp_path / "pred.csv"
        path.write_text("protein_id,position\nP1,10\n", encoding="utf-8")
        with pytest.raises(CandidateSpecError, match="missing required columns"):
            load_ptm_site_predictions(path)

    def test_load_predictions_rejects_bad_probability(self, tmp_path):
        path = tmp_path / "pred.csv"
        path.write_text(
            "protein_id,position,ptm_type,probability\nP1,10,phosphorylation,1.5\n",
            encoding="utf-8",
        )
        with pytest.raises(CandidateSpecError, match="within \\[0, 1\\]"):
            load_ptm_site_predictions(path)

    def test_load_gene_map_rejects_blank_mapping(self, tmp_path):
        path = tmp_path / "map.csv"
        path.write_text("protein_id,gene_symbol,ensembl_id\nP1,,\n", encoding="utf-8")
        with pytest.raises(CandidateSpecError, match="both"):
            load_gene_map(path)

    def test_load_gene_map_rejects_conflicting_rows(self, tmp_path):
        path = tmp_path / "map.csv"
        path.write_text(
            f"protein_id,gene_symbol,ensembl_id\n{PROTEIN},{GENE},{ENSEMBL}\n{PROTEIN},OTHER,ENSG00000000001\n",
            encoding="utf-8",
        )
        with pytest.raises(CandidateSpecError, match="conflicting"):
            load_gene_map(path)

    def test_load_evidence_rejects_direction_sign_mismatch(self, tmp_path):
        path = tmp_path / "evidence.csv"
        path.write_text(
            f"cell_type,ensembl_id,log2fc,fdr,observed_direction\nK562,{ENSEMBL},-1.0,0.01,up\n",
            encoding="utf-8",
        )
        with pytest.raises(CandidateSpecError, match="log2fc sign"):
            load_direction_evidence(path)

    def test_load_evidence_rejects_duplicate_rows(self, tmp_path):
        path = tmp_path / "evidence.csv"
        path.write_text(
            "cell_type,ensembl_id,log2fc,fdr,observed_direction\n"
            f"K562,{ENSEMBL},-1.0,0.01,down\n"
            f"K562,{ENSEMBL},-1.1,0.02,down\n",
            encoding="utf-8",
        )
        with pytest.raises(CandidateSpecError, match="duplicates"):
            load_direction_evidence(path)

    def test_load_direction_map_validates_direction(self, tmp_path):
        path = tmp_path / "dirs.csv"
        path.write_text(
            f"protein_id,position,ptm_type,proposed_direction\n{PROTEIN},12,phosphorylation,sideways\n",
            encoding="utf-8",
        )
        with pytest.raises(CandidateSpecError, match="'up' or 'down'"):
            load_direction_map(path)


class TestBuildSpecCandidates:
    def test_joined_candidate_matches_e2e_contract(self):
        candidates, summary = build_spec_candidates(
            _predictions_frame(),
            _gene_map(),
            _evidence_frame(),
            provenance="ptm-site-model/run-1",
            proposed_direction="down",
        )
        # Site probability 0.30 is filtered; P12345 is unmapped.
        assert summary.n_predictions == 3
        assert summary.n_below_probability == 1
        assert summary.n_unmapped_proteins == 1
        assert summary.n_candidates == 1
        row = candidates[0]
        assert row["gene_symbol"] == GENE
        assert row["ensembl_id"] == ENSEMBL
        assert row["position"] == 12
        assert row["proposed_direction"] == "down"
        assert row["site_probability"] == 0.95
        assert row["provenance"] == "ptm-site-model/run-1"
        assert row["cell_type"] == "K562"
        assert row["ptm_context"] == "STAT3:S12"
        assert row["observed_log2fc"] == -1.2
        assert row["observed_fdr"] == 0.01
        assert row["observed_direction"] == "down"
        # The row must satisfy the strict proposal contract used downstream.
        PTMSiteDirectionProposal(
            gene_symbol=row["gene_symbol"],
            ensembl_id=row["ensembl_id"],
            position=row["position"],
            ptm_type=row["ptm_type"],
            proposed_direction=row["proposed_direction"],
            site_probability=row["site_probability"],
            provenance=row["provenance"],
        )

    def test_neutral_evidence_rows_are_skipped(self):
        evidence = _evidence_frame(
            [
                {
                    "cell_type": "K562",
                    "ensembl_id": ENSEMBL,
                    "log2fc": 0.0,
                    "fdr": 0.9,
                    "observed_direction": "neutral",
                }
            ]
        )
        candidates, summary = build_spec_candidates(
            _predictions_frame(),
            _gene_map(),
            evidence,
            provenance="p/run-1",
            proposed_direction="down",
        )
        assert candidates == []
        # One site survives the probability and mapping filters and reaches
        # the neutral-direction check; the others short-circuit earlier.
        assert summary.n_neutral_direction == 1
        assert summary.n_without_evidence == 0

    def test_missing_evidence_rows_are_skipped_not_fabricated(self):
        candidates, summary = build_spec_candidates(
            _predictions_frame(),
            _gene_map(),
            _evidence_frame(
                [
                    {
                        "cell_type": "K562",
                        "ensembl_id": "ENSG00000000099",
                        "log2fc": 1.0,
                        "fdr": 0.01,
                        "observed_direction": "up",
                    }
                ]
            ),
            provenance="p/run-1",
            proposed_direction="up",
        )
        assert candidates == []
        # Only the mapped, above-threshold STAT3 site reaches the evidence
        # lookup; the unmapped P12345 protein short-circuits before it.
        assert summary.n_without_evidence == 1

    def test_multi_cell_type_evidence_requires_disambiguation(self):
        evidence = _evidence_frame(
            [
                {
                    "cell_type": "K562",
                    "ensembl_id": ENSEMBL,
                    "log2fc": -1.0,
                    "fdr": 0.01,
                    "observed_direction": "down",
                },
                {
                    "cell_type": "Monocyte",
                    "ensembl_id": ENSEMBL,
                    "log2fc": -1.5,
                    "fdr": 0.02,
                    "observed_direction": "down",
                },
            ]
        )
        with pytest.raises(CandidateSpecError, match="multiple cell types"):
            build_spec_candidates(
                _predictions_frame(),
                _gene_map(),
                evidence,
                provenance="p/run-1",
                proposed_direction="down",
            )
        candidates, _ = build_spec_candidates(
            _predictions_frame(),
            _gene_map(),
            evidence,
            provenance="p/run-1",
            proposed_direction="down",
            cell_types=["Monocyte"],
        )
        assert len(candidates) == 1
        assert candidates[0]["cell_type"] == "Monocyte"

    def test_low_donor_support_is_flagged_not_dropped(self):
        evidence = _evidence_frame(
            [
                {
                    "cell_type": "K562",
                    "ensembl_id": ENSEMBL,
                    "log2fc": -1.0,
                    "fdr": 0.01,
                    "observed_direction": "down",
                    "n_normal_donors": 2,
                    "n_disease_donors": 5,
                }
            ]
        )
        candidates, summary = build_spec_candidates(
            _predictions_frame(),
            _gene_map(),
            evidence,
            provenance="p/run-1",
            proposed_direction="down",
        )
        assert len(candidates) == 1
        assert candidates[0]["low_donor_support"] is True
        assert candidates[0]["direction_evidence_donor_counts"] == {"normal": 2, "disease": 5}
        assert summary.n_low_donor_support == 1

    def test_site_level_direction_override(self):
        overrides = {(PROTEIN, 12, "phosphorylation"): "up"}
        candidates, _ = build_spec_candidates(
            _predictions_frame(),
            _gene_map(),
            _evidence_frame(
                [
                    {
                        "cell_type": "K562",
                        "ensembl_id": ENSEMBL,
                        "log2fc": 1.0,
                        "fdr": 0.01,
                        "observed_direction": "up",
                    }
                ]
            ),
            provenance="p/run-1",
            proposed_direction="down",
            direction_overrides=overrides,
        )
        assert candidates[0]["proposed_direction"] == "up"


class TestPayload:
    def test_payload_round_trip_and_e2e_consumability(self, tmp_path):
        candidates, summary = build_spec_candidates(
            _predictions_frame(),
            _gene_map(),
            _evidence_frame(),
            provenance="ptm-site-model/run-1",
            proposed_direction="down",
        )
        payload = build_candidate_spec_payload(
            context_h5ad="data/processed/context.h5ad",
            candidates=candidates,
            summary=summary,
            sources={"ptm_site_predictions": "pred.csv"},
        )
        output = write_candidate_spec(payload, tmp_path / "spec.json")
        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded["schema_version"] == "ptm2cellnet.candidate-spec/v1"
        assert loaded["context_h5ad"].endswith("context.h5ad")
        assert loaded["build_summary"]["n_candidates"] == 1
        # The e2e CLI reads exactly these keys.
        assert isinstance(loaded["candidates"], list) and loaded["candidates"]
        row = loaded["candidates"][0]
        for key in (
            "context_cell_index",
            "gene_symbol",
            "ensembl_id",
            "position",
            "ptm_type",
            "proposed_direction",
            "site_probability",
            "provenance",
            "cell_type",
            "ptm_context",
            "observed_log2fc",
            "observed_fdr",
            "observed_direction",
        ):
            assert key in row

    def test_empty_candidate_list_is_rejected(self, tmp_path):
        from src.integration.perturbgen.candidate_spec import CandidateSpecSummary

        summary = CandidateSpecSummary(
            n_predictions=0,
            n_below_probability=0,
            n_unmapped_proteins=0,
            n_neutral_direction=0,
            n_without_evidence=0,
            n_low_donor_support=0,
            n_candidates=0,
        )
        with pytest.raises(CandidateSpecError, match="no candidates"):
            build_candidate_spec_payload(
                context_h5ad="ctx.h5ad",
                candidates=[],
                summary=summary,
                sources={},
            )

    def test_cli_end_to_end(self, tmp_path):
        predictions = tmp_path / "pred.csv"
        predictions.write_text(
            f"protein_id,position,aa,ptm_type,probability,sequence_window\n{PROTEIN},12,S,phosphorylation,0.95,xxx\n",
            encoding="utf-8",
        )
        gene_map = tmp_path / "map.csv"
        gene_map.write_text(
            f"protein_id,gene_symbol,ensembl_id\n{PROTEIN},{GENE},{ENSEMBL}\n",
            encoding="utf-8",
        )
        evidence = tmp_path / "evidence.csv"
        evidence.write_text(
            "cell_type,ensembl_id,gene_symbol,log2fc,p_value,fdr,observed_direction\n"
            f"K562,{ENSEMBL},{GENE},-1.2,0.001,0.01,down\n",
            encoding="utf-8",
        )
        context = tmp_path / "context.h5ad"
        context.write_text("placeholder", encoding="utf-8")
        output = tmp_path / "spec.json"

        from scripts.build_candidate_spec import main

        code = main(
            [
                "--ptm-site-predictions",
                str(predictions),
                "--gene-map",
                str(gene_map),
                "--direction-evidence",
                str(evidence),
                "--context-h5ad",
                str(context),
                "--output",
                str(output),
                "--provenance",
                "ptm-site-model/run-1",
                "--proposed-direction",
                "down",
                "--cell-type",
                "K562",
            ]
        )
        assert code == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["candidates"][0]["gene_symbol"] == GENE
