from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_davf_perturbgen_e2e import (
    _build_proposal,
    _load_candidate_spec,
    _load_davf_config,
)


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
