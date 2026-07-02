import pytest

import pandas as pd
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from src.evaluation.explainers import TwoStageExplanationPipeline
from src.integration.contracts import CandidateRecord, PerturbationResult


class FakeScorer:
    def rank_row(self, model, row):
        return [
            CandidateRecord(
                sample_id="S1",
                protein_id="P04637",
                ptm_type="phosphorylation",
                ptm_position=15,
                baseline_label="activated",
                baseline_probability=0.91,
                perturbed_probability=0.34,
                delta_probability=0.57,
            )
        ]


class FakeMapper:
    def map_candidate(self, candidate):
        return "TP53"


class FakeGenKI:
    def __init__(self) -> None:
        self.requests = []

    def build_request(self, gene_symbol: str, mode: str, magnitude: float):
        from src.integration.contracts import GenePerturbationRequest

        return GenePerturbationRequest(
            gene_symbol=gene_symbol,
            source_protein_id="",
            source_ptm_type="",
            source_ptm_position=-1,
            magnitude=magnitude,
            mode=mode,
        )

    def run(self, request):
        self.requests.append(request)
        return PerturbationResult(
            gene_symbol=request.gene_symbol,
            mode=request.mode,
            distance_score=1.23,
            ranked_genes=["BAX", "MDM2"],
            metadata={"pathways": ["apoptosis"]},
        )


def test_two_stage_pipeline_writes_markdown_summary(tmp_path) -> None:
    pipeline = TwoStageExplanationPipeline(FakeScorer(), FakeMapper(), FakeGenKI())
    df = pd.DataFrame([{"sample_id": "S1", "protein_id": "P04637", "sequence": "ACD", "ptm_sites": "[]"}])

    outputs = pipeline.run(model=object(), df=df, output_dir=tmp_path)

    assert outputs[0].gene_symbol == "TP53"
    assert (tmp_path / "two_stage_summary.md").exists()
    assert (tmp_path / "candidate_scores.csv").exists()
    assert (tmp_path / "genki_explanations.json").exists()
    assert (tmp_path / "gene_rankings.csv").exists()
    assert (tmp_path / "significant_gene_rankings.csv").exists()
    assert (tmp_path / "gsea_rankings.tsv").exists()
    assert (tmp_path / "two_stage_summary.json").exists()


def test_two_stage_pipeline_respects_top_k_candidates_and_mode(tmp_path) -> None:
    class MultiCandidateScorer:
        def rank_row(self, model, row):
            return [
                CandidateRecord("S1", "P04637", "phosphorylation", 15, "activated", 0.91, 0.34, 0.57),
                CandidateRecord("S1", "P04637", "acetylation", 21, "activated", 0.91, 0.55, 0.36),
            ]

    adapter = FakeGenKI()
    pipeline = TwoStageExplanationPipeline(
        MultiCandidateScorer(),
        FakeMapper(),
        adapter,
        top_k_candidates=2,
        perturbation_mode="soft_ptm",
    )
    df = pd.DataFrame([{"sample_id": "S1", "protein_id": "P04637", "sequence": "ACD", "ptm_sites": "[]"}])

    outputs = pipeline.run(model=object(), df=df, output_dir=tmp_path)

    assert len(outputs) == 2
    assert [request.mode for request in adapter.requests] == ["soft_ptm", "soft_ptm"]


def test_two_stage_pipeline_writes_unmapped_candidates(tmp_path) -> None:
    class MissingMapper:
        def map_candidate(self, candidate):
            raise KeyError(candidate.protein_id)

    pipeline = TwoStageExplanationPipeline(FakeScorer(), MissingMapper(), FakeGenKI())
    df = pd.DataFrame([{"sample_id": "S1", "protein_id": "P04637", "sequence": "ACD", "ptm_sites": "[]"}])

    outputs = pipeline.run(model=object(), df=df, output_dir=tmp_path)

    assert outputs == []
    assert (tmp_path / "unmapped_candidates.csv").exists()


def test_two_stage_cli_uses_top_k_and_perturbation_mode(tmp_path) -> None:
    config = {
        "integration": {
            "genki_ref_root": "tests/fixtures/genki",
            "mapping_file": "tests/fixtures/genki/mock_mapping.csv",
            "target_class_index": 1,
            "top_k_candidates": 2,
            "perturbation_mode": "soft_ptm",
            "label_names": ["inactive", "active"],
            "amino_acids": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_type_to_index": {
                "phosphorylation": 1,
                "acetylation": 2,
            },
        }
    }
    config_path = tmp_path / "two_stage.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    output_dir = tmp_path / "outputs"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_two_stage_explanation.py",
            "--config",
            str(config_path),
            "--input",
            "data/processed/example_candidates.csv",
            "--output",
            str(output_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    explanations = pd.read_json(output_dir / "genki_explanations.json")
    assert len(explanations) == 3
    assert set(explanations["mode"].tolist()) == {"soft_ptm"}
    backend_info = pd.read_json(output_dir / "backend_info.json", typ="series")
    assert backend_info["backend"] == "array_files"


def test_two_stage_cli_accepts_genki_source_backend(tmp_path) -> None:
    pytest.importorskip("torch_geometric")
    import anndata as ad
    import scipy.sparse as sp

    adata_path = tmp_path / "mini.h5ad"
    grn_dir = tmp_path / "GRNs"
    grn_dir.mkdir()
    mapping_file = tmp_path / "mapping.csv"
    input_file = tmp_path / "input.csv"

    adata = ad.AnnData(sp.csr_matrix(np.array([[-1.0, 0.5, 1.0], [0.2, -0.3, 0.1]], dtype=float)))
    adata.var_names = ["EGFR", "TP53", "BAX"]
    adata.layers["norm"] = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=float)
    adata.write_h5ad(adata_path)
    sp.save_npz(grn_dir / "pcNet.npz", sp.csr_matrix(np.array([[0.0, 0.9, 0.2], [0.9, 0.0, 0.8], [0.2, 0.8, 0.0]], dtype=float)))
    mapping_file.write_text("protein_id,gene_symbol\nP04637,TP53\n", encoding="utf-8")
    input_file.write_text(
        'sample_id,protein_id,sequence,ptm_sites\n'
        'S1,P04637,ACDEFG,"[{""position"": 2, ""type"": ""phosphorylation""}]"\n',
        encoding="utf-8",
    )

    config = {
        "integration": {
            "genki_ref_root": "ref/GenKI-master-src/GenKI-master",
            "mapping_file": str(mapping_file),
            "adata_file": str(adata_path),
            "grn_file_dir": str(grn_dir),
            "pcnet_name": "pcNet",
            "cutoff": 0,
            "target_class_index": 1,
            "top_k_candidates": 1,
            "perturbation_mode": "hard_ko",
            "label_names": ["inactive", "active"],
            "amino_acids": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_type_to_index": {
                "phosphorylation": 1,
            },
        }
    }
    config_path = tmp_path / "two_stage_source.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    output_dir = tmp_path / "outputs_source"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_two_stage_explanation.py",
            "--config",
            str(config_path),
            "--input",
            str(input_file),
            "--output",
            str(output_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    backend_info = pd.read_json(output_dir / "backend_info.json", typ="series")
    assert backend_info["backend"] == "genki_source"
