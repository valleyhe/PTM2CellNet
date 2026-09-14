from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_davf_perturbgen_e2e as e2e
from scripts.run_davf_perturbgen_e2e import (
    _build_proposal,
    _load_candidate_spec,
    _load_davf_config,
    _preflight_perturbgen_context,
    _validate_perturbgen_tokenise_input,
)
from src.integration.perturbgen.contracts import PerturbGenDataSpec


def _perturbgen_config(context_path: Path) -> dict:
    return {
        "pipeline": {"random_seed": 0},
        "stages": {
            "tokenise": {
                "args": {
                    "h5ad_path": str(context_path),
                    "var_list": ["cell_type", "state", "donor"],
                    "main_pairing_obs": "cell_type",
                    "time_obs": "state",
                    "reference_time": "normal",
                    "time_point_order": ["normal", "disease"],
                }
            }
        },
    }


def test_load_davf_config_resolves_project_relative_assets(tmp_path):
    config_path = tmp_path / "davf.yaml"
    config_path.write_text(
        "davf:\n"
        "  checkpoint_path: checkpoints/model.pt\n"
        "  scvi_model_path: checkpoints/scvi\n"
        "  embedding_asset_path: outputs/asset\n"
        "  gene_names_path: data/aliases.tsv\n"
        "  intervention_type: KO\n"
        "  latent_dim: 64\n"
        "  num_genes: 4018\n",
        encoding="utf-8",
    )

    config = _load_davf_config(config_path)

    assert config.intervention_type == "KO"
    assert config.latent_dim == 64
    assert config.num_genes == 4018
    assert config.checkpoint_path.endswith("/checkpoints/model.pt")
    assert config.scvi_model_path.endswith("/checkpoints/scvi")


def test_load_candidate_spec_requires_explicit_context_cell_index(tmp_path):
    context_path = tmp_path / "context.h5ad"
    context_path.write_bytes(b"placeholder")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "context_h5ad": str(context_path),
                "candidates": [{"gene_symbol": "STAT3"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="context_cell_index"):
        _load_candidate_spec(spec_path)


def test_build_proposal_reports_missing_schema_fields():
    with pytest.raises(ValueError, match="ensembl_id"):
        _build_proposal({"gene_symbol": "STAT3"}, 0)


def test_perturbgen_tokenise_input_must_be_the_candidate_context(tmp_path):
    context_path = tmp_path / "context.h5ad"
    other_path = tmp_path / "other.h5ad"
    context_path.write_bytes(b"context")
    other_path.write_bytes(b"other")

    with pytest.raises(ValueError, match="exact candidate context_h5ad"):
        _validate_perturbgen_tokenise_input(_perturbgen_config(other_path), context_path.resolve())


def test_perturbgen_tokenise_input_binds_cohort_pairing_into_spec(tmp_path):
    context_path = tmp_path / "context.h5ad"
    context_path.write_bytes(b"context")
    config = _perturbgen_config(context_path)
    default_spec, _ = _validate_perturbgen_tokenise_input(config, context_path.resolve())
    assert default_spec.pairing == "within_donor"
    case_control_spec, _ = _validate_perturbgen_tokenise_input(
        config, context_path.resolve(), cohort_pairing="between_donor"
    )
    assert case_control_spec.pairing == "between_donor"
    with pytest.raises(ValueError, match="pairing must be one of"):
        _validate_perturbgen_tokenise_input(config, context_path.resolve(), cohort_pairing="cross_over")


def test_gate0_preflight_uses_the_same_context_and_explicit_cell_type(monkeypatch, tmp_path):
    context_path = tmp_path / "context.h5ad"
    context_path.write_bytes(b"context")
    spec, tokenise_path = _validate_perturbgen_tokenise_input(_perturbgen_config(context_path), context_path.resolve())
    context = SimpleNamespace(obs=pd.DataFrame({"cell_type": ["Mono"]}))
    calls = []

    def fake_prepare(adata, *, cell_type, spec):
        calls.append((adata, cell_type, spec))
        return SimpleNamespace(report=SimpleNamespace(cell_type=cell_type))

    monkeypatch.setattr(
        "scripts.run_davf_perturbgen_e2e.prepare_perturbgen_anndata",
        fake_prepare,
    )

    result = _preflight_perturbgen_context(
        context,
        context_path.resolve(),
        [{"context_cell_index": 0, "cell_type": "Mono"}],
        spec=spec,
        tokenise_path=tokenise_path,
    )

    assert result["status"] == "pass"
    assert result["context_h5ad"] == result["tokenise_h5ad"]
    assert calls == [(context, "Mono", spec)]


def test_gate0_preflight_rejects_context_cell_type_mismatch(tmp_path, monkeypatch):
    context_path = tmp_path / "context.h5ad"
    context_path.write_bytes(b"context")
    spec = PerturbGenDataSpec()
    context = SimpleNamespace(obs=pd.DataFrame({"cell_type": ["Mono"]}))
    monkeypatch.setattr(
        "scripts.run_davf_perturbgen_e2e.prepare_perturbgen_anndata",
        lambda *_args, **_kwargs: SimpleNamespace(report=SimpleNamespace(cell_type="Mono")),
    )

    with pytest.raises(ValueError, match="has cell_type 'Mono', not 'T cell'"):
        _preflight_perturbgen_context(
            context,
            context_path.resolve(),
            [{"context_cell_index": 0, "cell_type": "T cell"}],
            spec=spec,
            tokenise_path=context_path.resolve(),
        )


def test_run_perturbgen_formal_path_executes_gate0_before_runner(tmp_path, monkeypatch):
    context_path = tmp_path / "context.h5ad"
    context_path.write_bytes(b"context")
    spec_path = tmp_path / "candidate.json"
    spec_path.write_text(
        json.dumps(
            {
                "context_h5ad": str(context_path),
                "candidates": [
                    {
                        "context_cell_index": 0,
                        "gene_symbol": "STAT3",
                        "ensembl_id": "ENSG00000168610",
                        "position": 12,
                        "ptm_type": "phosphorylation",
                        "proposed_direction": "down",
                        "site_probability": 0.95,
                        "provenance": "unit-test",
                        "cell_type": "Mono",
                        "ptm_context": "STAT3:S12",
                        "observed_log2fc": -1.0,
                        "observed_fdr": 0.01,
                        "observed_direction": "down",
                        "semantic_context": {
                            "context": "disease",
                            "intervention": "KO",
                            "comparison_baseline": "normal",
                            "reference_axis": "disease-minus-normal",
                            "research_objective": "replication",
                            "evidence_source": "donor_expression+davf_decode",
                            "cohort": "formal",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeContext:
        n_obs = 1
        obs = pd.DataFrame({"cell_type": ["Mono"]})

        def __getitem__(self, _index):
            return self

        def copy(self):
            return self

    context = FakeContext()
    prepare_calls = []

    def fake_prepare(adata, *, cell_type, spec):
        prepare_calls.append((adata, cell_type, spec))
        return SimpleNamespace(report={"cell_type": cell_type, "evaluable_donors": ["d1", "d2", "d3"]})

    class FakeDAVF:
        def __init__(self, _config):
            pass

        def load_scvi_adapter(self, _adata):
            return SimpleNamespace(encode=lambda _selected: "z0")

    class FakeOrchestrator:
        def __init__(self, *, davf_module):
            pass

        def prepare_candidates(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(e2e, "_load_davf_config", lambda _path: SimpleNamespace(intervention_type="KO"))
    monkeypatch.setattr(e2e, "load_pipeline_config", lambda _path: _perturbgen_config(context_path))
    monkeypatch.setattr(e2e, "prepare_perturbgen_anndata", fake_prepare)
    monkeypatch.setattr(e2e, "DAVFInferenceModule", FakeDAVF)
    monkeypatch.setattr(e2e, "DAVFPerturbGenOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(e2e, "merge_route_preparations", lambda _preparations: [])
    monkeypatch.setattr(e2e, "PerturbGenRunner", lambda **_kwargs: object())
    monkeypatch.setitem(sys.modules, "anndata", SimpleNamespace(read_h5ad=lambda _path: context))

    args = SimpleNamespace(
        davf_config=tmp_path / "davf.yaml",
        candidate_spec=spec_path,
        output=tmp_path / "report.json",
        perturbgen_config=tmp_path / "perturbgen.yaml",
        perturbgen_cohort_pairing="within_donor",
        perturbgen_output_root=None,
        gpu_lock_file=tmp_path / "gpu.lock",
        run_perturbgen=True,
        resume=False,
        dry_run=True,
        seeds="0",
        sensitivity_modes="",
        train_donors=None,
        held_out_donors=None,
        frozen_cohort_manifest=None,
        require_donor_split=False,
        assemble_statistical_evidence=False,
        deg_table=None,
        null_distribution_manifest=None,
        statistical_output_dir=None,
    )

    payload = e2e._run(args)

    assert len(prepare_calls) == 1
    assert prepare_calls[0][0] is context
    assert prepare_calls[0][1] == "Mono"
    assert payload["perturbgen_gate0"]["status"] == "pass"
    assert payload["statistical_evidence"] == {
        "status": "inconclusive",
        "scientific_acceptance": False,
        "reason": "statistical_evidence_not_assembled",
        "interfaces": [
            "src.integration.perturbgen.null_generation",
            "src.integration.perturbgen.reports",
            "src.integration.perturbgen.empirical_pvalue",
            "src.integration.perturbgen.dual_path",
        ],
    }

    default_args = SimpleNamespace(**vars(args))
    default_args.run_perturbgen = False
    default_args.perturbgen_config = None
    default_args.dry_run = False
    default_payload = e2e._run(default_args)

    assert default_payload["statistical_evidence"] == {
        "status": "inconclusive",
        "scientific_acceptance": False,
        "reason": "perturbgen_not_requested",
    }


@pytest.mark.parametrize(
    "route,scvi_name",
    [("KO", "davf_ko_dixit"), ("KD", "davf_kd_nadig")],
)
def test_dated_formal_davf_configs_declare_current_route_contract(route, scvi_name):
    config_path = Path(f"configs/davf_{route.lower()}_cell_baseline_formal_20260904.yaml")

    config = _load_davf_config(config_path)

    assert config.intervention_type == route
    assert config.latent_dim == 64
    assert config.num_genes == 4018
    assert config.num_steps == 1
    assert config.checkpoint_path.endswith(
        f"checkpoints/davf/davf_{route.lower()}_cell_baseline_formal_20260904/best_model.pt"
    )
    assert config.scvi_model_path.endswith(f"checkpoints/scvi/{scvi_name}")
    assert config.embedding_asset_path.endswith("outputs/perturbgen/embedding_asset_20260822")


def _write_stat_h5ad(path: Path, *, donor_column: str = "donor_id") -> None:
    import anndata as ad

    baseline = np.asarray([[8.0, 1.0, 100.0], [7.5, 1.2, 110.0], [9.0, 0.8, 120.0]], dtype=float)
    perturbed = np.asarray([[2.0, 2.0, 0.0], [2.5, 2.2, 0.0], [1.5, 2.1, 0.0]], dtype=float)
    adata = ad.AnnData(
        X=perturbed,
        obs=pd.DataFrame({donor_column: ["D1", "D2", "D3"]}, index=["c1", "c2", "c3"]),
        var=pd.DataFrame(index=["UP_A", "DOWN_A", "TARGET"]),
    )
    adata.layers["true_counts"] = np.ones_like(perturbed)
    adata.layers["pred_counts"] = baseline
    adata.obsm["true_cls"] = np.ones((3, 2))
    adata.obsm["perturbed_cls"] = np.ones((3, 2))
    adata.obsm["mean_cos_similarity"] = np.ones((3, 1))
    adata.varm["gene_cos_similarity"] = np.ones((3, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def _write_stat_stage(
    run_root: Path,
    *,
    path_kind: str,
    seed: int,
    tokenise_artifacts: dict,
) -> Path:
    import hashlib

    sequence = "src" if path_kind == "source_intervention" else "tgt"
    stage_dir = run_root / "perturb" / path_kind / f"mask_seed{seed}"
    h5ad = stage_dir / "results" / f"20260910_minference_adata_gSTAT3_s{sequence}_tmask.h5ad"
    _write_stat_h5ad(h5ad)
    sha256 = hashlib.sha256(h5ad.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "stage": path_kind,
        "status": "success",
        "fingerprint": f"fp-{path_kind}-{seed}",
        "fingerprint_material": {
            "random_seed": seed,
            "fingerprint_config": {
                "stage_config": {
                    "perturb_config": {
                        "data": {
                            "src_dataset_file": tokenise_artifacts["src_dataset"],
                            "src_adata": tokenise_artifacts["src_h5ad"],
                            "tgt_dataset_folder": tokenise_artifacts["tgt_dataset_folder"],
                            "tgt_adata_folder": tokenise_artifacts["tgt_h5ad_folder"],
                        }
                    }
                }
            },
        },
        "outputs": {str(h5ad): {"kind": "file", "sha256": sha256}},
        "artifacts": {"result_h5ad": str(h5ad)},
    }
    (stage_dir / "stage_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return h5ad


def _stat_e2e_payload(run_root: Path) -> dict:
    return {
        "schema_version": "davf_perturbgen_e2e/v1",
        "intervention_type": "KO",
        "candidates": [
            {
                "intervention_type": "KO",
                "status": "pass",
                "invocation": {
                    "gene_symbol": "STAT3",
                    "ensembl_id": "ENSG00000168610",
                    "intervention_type": "KO",
                    "candidate": {"observed_direction": "up", "direction_gate_status": "pass"},
                },
            }
        ],
        "perturbgen_runs": [
            {
                "intervention_type": "KO",
                "gene_symbol": "STAT3",
                "ensembl_id": "ENSG00000168610",
                "output_root": str(run_root),
                "stages": [],
            }
        ],
    }


def test_assemble_statistical_evidence_chains_null_quality_pq_dual_path(tmp_path):
    run_root = tmp_path / "perturbgen" / "KO" / "ENSG00000168610"
    tokenise_root = run_root / "tokenise" / "artifacts"
    tokenise_root.mkdir(parents=True, exist_ok=True)
    (tokenise_root / "src.dataset").write_bytes(b"src")
    (tokenise_root / "src.h5ad").write_bytes(b"src-h5ad")
    (tokenise_root / "tgt_dataset").mkdir()
    (tokenise_root / "tgt_h5ad").mkdir()
    tokenise_artifacts = {
        "src_dataset": str(tokenise_root / "src.dataset"),
        "src_h5ad": str(tokenise_root / "src.h5ad"),
        "tgt_dataset_folder": str(tokenise_root / "tgt_dataset"),
        "tgt_h5ad_folder": str(tokenise_root / "tgt_h5ad"),
    }
    (run_root / "tokenise" / "stage_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "tokenise",
                "status": "success",
                "artifacts": tokenise_artifacts,
            }
        ),
        encoding="utf-8",
    )
    for path_kind in ("source_intervention", "within_state"):
        for seed in (0, 1, 2):
            _write_stat_stage(run_root, path_kind=path_kind, seed=seed, tokenise_artifacts=tokenise_artifacts)

    deg_table = tmp_path / "deg.csv"
    rows = [f"{donor},UP_A,1.0,0.01" for donor in ("D1", "D2", "D3")] + [
        f"{donor},DOWN_A,-1.0,0.01" for donor in ("D1", "D2", "D3")
    ]
    deg_table.write_text("donor,gene_symbol,log2fc,fdr\n" + "\n".join(rows) + "\n", encoding="utf-8")

    distributions = []
    for path_kind in ("source_intervention", "within_state"):
        for seed in (0, 1, 2):
            distributions.append(
                {
                    "schema_version": "perturbgen_null_distribution/v1",
                    "candidate_ensembl_id": "ENSG00000168610",
                    "path": path_kind,
                    "mode": "mask",
                    "seed": seed,
                    "required_count": 99,
                    "values": [round(-0.5 + 0.001 * index, 6) for index in range(99)],
                    "null_ensembl_ids": [f"ENSG0000000{index:05d}" for index in range(99)],
                }
            )
    null_manifest = tmp_path / "null_index.json"
    null_manifest.write_text(
        json.dumps(
            {
                "schema_version": "perturbgen_null_distribution/v1",
                "distributions": distributions,
            }
        ),
        encoding="utf-8",
    )

    payload = _stat_e2e_payload(run_root)
    evidence = e2e._assemble_statistical_evidence(
        payload,
        deg_table_path=deg_table,
        null_distribution_manifest_path=null_manifest,
        output_dir=tmp_path / "statistical_evidence",
        donor_obs_column="donor_id",
    )

    assert evidence["status"] == "assembled"
    assert evidence["evaluation_mode"] == "formal"
    assert evidence["scientific_acceptance"] is False
    assert len(evidence["candidates"]) == 1
    candidate = evidence["candidates"][0]
    assert candidate["ensembl_id"] == "ENSG00000168610"
    assert candidate["pvalue"] is not None and 0.0 <= candidate["pvalue"] <= 1.0
    assert candidate["q_value"] is not None and 0.0 <= candidate["q_value"] <= 1.0
    assert candidate["verdict"] in {"pass", "fail", "inconclusive"}
    assert candidate["scientific_acceptance"] is False
    assert Path(evidence["eval_input"]).is_file()
    assert Path(evidence["report_manifest"]).is_file()


def test_assemble_statistical_evidence_requires_completed_runs(tmp_path):
    payload = {"schema_version": "davf_perturbgen_e2e/v1", "perturbgen_runs": []}
    deg_table = tmp_path / "deg.csv"
    deg_table.write_text("donor,gene_symbol,log2fc,fdr\nd1,G1,1.0,0.01\n", encoding="utf-8")
    null_manifest = tmp_path / "null.json"
    null_manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="requires completed"):
        e2e._assemble_statistical_evidence(
            payload,
            deg_table_path=deg_table,
            null_distribution_manifest_path=null_manifest,
            output_dir=tmp_path / "statistical_evidence",
            donor_obs_column="donor_id",
        )
