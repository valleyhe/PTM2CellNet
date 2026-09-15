"""End-to-end CLI chain for the PTM-activity mainline (方案 §7 阶段 1→5).

Runs the four stage CLIs against small synthetic tables and a synthetic
cohort h5ad, then checks that the generated candidate spec carries every
field ``run_davf_perturbgen_e2e.py`` requires (proposal fields, direction-gate
fields, semantic context) and that the sidecar carries the frozen target
set. Synthetic data proves the *contracts and plumbing* only — it is not a
biological result (方案 §2.3).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from scripts.build_celltype_candidate_specs import main as stage5_main
from scripts.build_ptm_ad_intersections import main as stage4_main
from scripts.build_ptm_global_gene_scores import main as stage3_main
from scripts.run_ptm_activity import main as stage1_main

MAPT_ENSG = "ENSG00000186868"
GSK3B_ENSG = "ENSG00000082701"


@pytest.fixture()
def pipeline_inputs(tmp_path):
    work = tmp_path / "ptm"
    work.mkdir()

    config_path = work / "ptm_research_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: ptm2cellnet.ptm-research-config/v1",
                "research_objective: association",
                "reference_axis: disease_minus_normal",
                'contrast: "disease-minus-normal"',
                "primary_activity_method: KSTAR",
                "sensitivity_activity_method: PhosR",
                'network_release: "2026-08"',
                "cell_types: [EX, IN]",
                "cohort_h5ad: unused-placeholder.h5ad",
                "cohort_pairing: between_donor",
                'species: "9606"',
                "ptm_cohort: CPTAC_TEST",
                "deg_max_fdr: 0.05",
                "min_donors_per_state: 3",
                "replicate_policy: mean",
                "propagation:",
                "  max_depth: 3",
                "  decay: 0.5",
                "  gene_edge_types: [tf_regulation]",
                "semantic_context:",
                '  context: "GSE174367 {cell_type} cells"',
                '  intervention: "KO"',
                '  comparison_baseline: "donor-level disease vs normal"',
                '  reference_axis: "disease_minus_normal"',
                "  research_objective: association",
                '  evidence_source: "PTM activity network concordance + donor DEG"',
                '  cohort: "GSE174367"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    ptm_input = work / "ptm_site_quantification.tsv"
    pd.DataFrame(
        [
            {
                "sample_id": "S1",
                "donor_id": "D1",
                "condition": "disease",
                "protein_id": "P12345",
                "gene_symbol": "GSK3B",
                "residue": "S9",
                "ptm_type": "phosphorylation",
                "ptm_value": 2.0,
                "value_scale": "linear",
                "ptm_qvalue": 0.01,
                "total_protein_value": 4.0,
                "species": "9606",
                "source_dataset": "CPTAC_TEST",
            },
            {
                "sample_id": "S2",
                "donor_id": "D2",
                "condition": "normal",
                "protein_id": "P12345",
                "gene_symbol": "GSK3B",
                "residue": "S9",
                "ptm_type": "phosphorylation",
                "ptm_value": 1.0,
                "value_scale": "linear",
                "ptm_qvalue": 0.02,
                "total_protein_value": 4.0,
                "species": "9606",
                "source_dataset": "CPTAC_TEST",
            },
        ]
    ).to_csv(ptm_input, sep="\t", index=False)

    gene_map = work / "gene_map.tsv"
    gene_map.write_text(
        f"protein_id\tgene_symbol\tensembl_id\nP12345\tGSK3B\t{GSK3B_ENSG}\n",
        encoding="utf-8",
    )

    activity = work / "ptm_activity.tsv"
    pd.DataFrame(
        [
            {
                "activity_unit": "zscore",
                "regulator_id": "GSK3B",
                "regulator_type": "kinase",
                "condition_or_contrast": "disease-minus-normal",
                "activity_score": 2.0,
                "activity_direction": "up",
                "activity_pvalue": 0.01,
                "activity_qvalue": 0.05,
                "n_substrates": 12,
                "network_coverage": 0.8,
                "method": "KSTAR",
                "method_version": "1.0",
                "input_manifest": "ptm_input_manifest.json",
            }
        ]
    ).to_csv(activity, sep="\t", index=False)

    network = work / "signed_network.tsv"
    pd.DataFrame(
        {
            "source_id": ["GSK3B", "CDK5"],
            "target_id": ["CDK5", "MAPT"],
            "edge_type": ["protein_protein", "tf_regulation"],
            "effect_sign": ["+1", "-1"],
            "site": ["", "s1"],
            "species": ["9606", "9606"],
            "evidence": ["test", "test"],
            "confidence": [1.0, 1.0],
            "release": ["2026-08", "2026-08"],
        }
    ).to_csv(network, sep="\t", index=False)

    id_map = work / "network_id_map.tsv"
    id_map.write_text(
        f"network_id\tgene_symbol\tensembl_id\nMAPT\tMAPT\t{MAPT_ENSG}\n",
        encoding="utf-8",
    )

    deg = work / "ad_deg.tsv"
    pd.DataFrame(
        [
            # source gene evidence in EX (three-way gate third side)
            ["EX", GSK3B_ENSG, "GSK3B", -0.6, 0.01, "down", 4, 8],
            # concordant target in EX
            ["EX", MAPT_ENSG, "MAPT", -0.7, 0.01, "down", 4, 8],
            # discordant + weak FDR in IN
            ["IN", MAPT_ENSG, "MAPT", 0.5, 0.2, "up", 3, 3],
            ["IN", GSK3B_ENSG, "GSK3B", 0.4, 0.2, "up", 3, 3],
        ],
        columns=[
            "cell_type",
            "ensembl_id",
            "gene_symbol",
            "log2fc",
            "fdr",
            "observed_direction",
            "n_normal_donors",
            "n_disease_donors",
        ],
    ).to_csv(deg, sep="\t", index=False)

    proposals = work / "source_proposals.tsv"
    proposals.write_text(
        "protein_id\tposition\tptm_type\tgene_symbol\tensembl_id\tsource_activity_id\t"
        "proposed_direction\tsite_probability\tprovenance\n"
        f"P12345\t9\tphosphorylation\tGSK3B\t{GSK3B_ENSG}\tGSK3B\t"
        f"down\t0.9\tliterature:test\n",
        encoding="utf-8",
    )

    cohort = work / "cohort.h5ad"
    obs = pd.DataFrame(
        {"cell_type": ["IN", "EX", "EX"], "donor": ["D1", "D1", "D2"]},
        index=["cell0", "cell1", "cell2"],
    )
    var = pd.DataFrame(index=[MAPT_ENSG, GSK3B_ENSG])
    cohort_data = anndata.AnnData(X=np.ones((3, 2), dtype="float32"), obs=obs, var=var)
    cohort_data.write_h5ad(cohort)

    vocab = work / "embedding_vocab.json"
    vocab.write_text(json.dumps({GSK3B_ENSG: 101}), encoding="utf-8")

    return {
        "work": work,
        "config": config_path,
        "ptm_input": ptm_input,
        "gene_map": gene_map,
        "activity": activity,
        "network": network,
        "id_map": id_map,
        "deg": deg,
        "proposals": proposals,
        "cohort": cohort,
        "vocab": vocab,
    }


class TestPTMActivityPipeline:
    def test_full_chain_produces_formal_spec_and_sidecar(self, pipeline_inputs, capsys):
        work = pipeline_inputs["work"]
        standardized = work / "standardized_ptm.tsv"
        input_manifest = work / "ptm_input_manifest.json"
        assert (
            stage1_main(
                [
                    "--config",
                    str(pipeline_inputs["config"]),
                    "--input-tsv",
                    str(pipeline_inputs["ptm_input"]),
                    "--gene-map",
                    str(pipeline_inputs["gene_map"]),
                    "--output-tsv",
                    str(standardized),
                    "--manifest-output",
                    str(input_manifest),
                ]
            )
            == 0
        )
        assert json.loads(input_manifest.read_text(encoding="utf-8"))["standardization"]["n_donors"] == 2

        scores = work / "ptm_global_gene_scores.tsv"
        score_manifest = work / "gene_score_manifest.json"
        assert (
            stage3_main(
                [
                    "--config",
                    str(pipeline_inputs["config"]),
                    "--activity-tsv",
                    str(pipeline_inputs["activity"]),
                    "--network-tsv",
                    str(pipeline_inputs["network"]),
                    "--network-id-map",
                    str(pipeline_inputs["id_map"]),
                    "--output-tsv",
                    str(scores),
                    "--manifest-output",
                    str(score_manifest),
                ]
            )
            == 0
        )
        score_frame = pd.read_csv(scores, sep="\t")
        assert len(score_frame) == 1
        assert score_frame.iloc[0]["target_ensembl_id"] == MAPT_ENSG
        assert score_frame.iloc[0]["predicted_gene_direction"] == "down"
        assert score_frame.iloc[0]["prediction_status"] == "direction_only"

        intersection_dir = work / "intersections"
        assert (
            stage4_main(
                [
                    "--config",
                    str(pipeline_inputs["config"]),
                    "--gene-scores-tsv",
                    str(scores),
                    "--deg-table",
                    str(pipeline_inputs["deg"]),
                    "--output-dir",
                    str(intersection_dir),
                ]
            )
            == 0
        )
        manifest = json.loads((intersection_dir / "target_set_manifest.json").read_text(encoding="utf-8"))
        assert manifest["cell_types"]["EX"]["sources"]["GSK3B"]["targets"][0]["target_ensembl_id"] == MAPT_ENSG
        assert manifest["cell_types"]["IN"]["sources"] == {}

        spec_dir = work / "specs"
        assert (
            stage5_main(
                [
                    "--config",
                    str(pipeline_inputs["config"]),
                    "--source-proposals-tsv",
                    str(pipeline_inputs["proposals"]),
                    "--deg-table",
                    str(pipeline_inputs["deg"]),
                    "--target-set-manifest",
                    str(intersection_dir / "target_set_manifest.json"),
                    "--context-h5ad",
                    str(pipeline_inputs["cohort"]),
                    "--embedding-vocab",
                    str(pipeline_inputs["vocab"]),
                    "--output-dir",
                    str(spec_dir),
                ]
            )
            == 0
        )
        spec_payload = json.loads((spec_dir / "candidate_spec_EX.json").read_text(encoding="utf-8"))
        candidate = spec_payload["candidates"][0]
        for field in (
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
            "semantic_context",
        ):
            assert field in candidate, f"candidate missing {field}"
        # EX first cell is row index 1 in the synthetic cohort (real binding, 方案 §4.6)
        assert candidate["context_cell_index"] == 1
        assert candidate["semantic_context"]["context"] == "GSE174367 EX cells"
        assert candidate["semantic_context"]["intervention"] == "KO"
        assert candidate["observed_direction"] == "down"

        sidecar = json.loads((spec_dir / "downstream_targets_EX.json").read_text(encoding="utf-8"))
        assert sidecar["schema_version"] == "ptm2cellnet.downstream-target-sidecar/v1"
        targets = sidecar["sources"][GSK3B_ENSG]["targets"]
        assert [t["target_ensembl_id"] for t in targets] == [MAPT_ENSG]

        build_summary = json.loads((spec_dir / "build_summary.json").read_text(encoding="utf-8"))
        assert build_summary["cell_types"]["EX"]["n_candidates"] == 1
        assert build_summary["cell_types"]["IN"]["n_candidates"] == 0

    def test_source_without_verified_token_fails_hard(self, pipeline_inputs):
        work = pipeline_inputs["work"]
        bad_vocab = work / "bad_vocab.json"
        bad_vocab.write_text(json.dumps({"ENSG00000000009": 1}), encoding="utf-8")
        assert (
            stage5_main(
                [
                    "--config",
                    str(pipeline_inputs["config"]),
                    "--source-proposals-tsv",
                    str(pipeline_inputs["proposals"]),
                    "--deg-table",
                    str(pipeline_inputs["deg"]),
                    "--target-set-manifest",
                    str(self._intersection_dir(work, pipeline_inputs)),
                    "--context-h5ad",
                    str(pipeline_inputs["cohort"]),
                    "--embedding-vocab",
                    str(bad_vocab),
                    "--output-dir",
                    str(work / "specs_bad"),
                ]
            )
            == 1
        )

    def test_stage1_missing_input_fails_cleanly(self, pipeline_inputs):
        assert (
            stage1_main(
                [
                    "--config",
                    str(pipeline_inputs["config"]),
                    "--input-tsv",
                    str(pipeline_inputs["work"] / "missing.tsv"),
                    "--gene-map",
                    str(pipeline_inputs["gene_map"]),
                    "--output-tsv",
                    str(pipeline_inputs["work"] / "out.tsv"),
                    "--manifest-output",
                    str(pipeline_inputs["work"] / "m.json"),
                ]
            )
            == 1
        )

    @staticmethod
    def _intersection_dir(work, pipeline_inputs):
        scores = work / "ptm_global_gene_scores.tsv"
        if not scores.is_file():
            stage1_main(
                [
                    "--config",
                    str(pipeline_inputs["config"]),
                    "--input-tsv",
                    str(pipeline_inputs["ptm_input"]),
                    "--gene-map",
                    str(pipeline_inputs["gene_map"]),
                    "--output-tsv",
                    str(work / "standardized_ptm.tsv"),
                    "--manifest-output",
                    str(work / "ptm_input_manifest.json"),
                ]
            )
            stage3_main(
                [
                    "--config",
                    str(pipeline_inputs["config"]),
                    "--activity-tsv",
                    str(pipeline_inputs["activity"]),
                    "--network-tsv",
                    str(pipeline_inputs["network"]),
                    "--network-id-map",
                    str(pipeline_inputs["id_map"]),
                    "--output-tsv",
                    str(scores),
                    "--manifest-output",
                    str(work / "gene_score_manifest.json"),
                ]
            )
            stage4_main(
                [
                    "--config",
                    str(pipeline_inputs["config"]),
                    "--gene-scores-tsv",
                    str(scores),
                    "--deg-table",
                    str(pipeline_inputs["deg"]),
                    "--output-dir",
                    str(work / "intersections"),
                ]
            )
        return work / "intersections" / "target_set_manifest.json"
