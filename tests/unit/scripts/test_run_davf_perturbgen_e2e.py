from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

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
        }
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
        perturbgen_output_root=None,
        gpu_lock_file=tmp_path / "gpu.lock",
        run_perturbgen=True,
        resume=False,
        dry_run=True,
        seeds="0",
        sensitivity_modes="",
    )

    payload = e2e._run(args)

    assert len(prepare_calls) == 1
    assert prepare_calls[0][0] is context
    assert prepare_calls[0][1] == "Mono"
    assert payload["perturbgen_gate0"]["status"] == "pass"


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
