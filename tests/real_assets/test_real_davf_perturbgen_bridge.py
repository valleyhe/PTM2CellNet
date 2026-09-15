"""Real-asset DAVF -> PerturbGen bridge smoke.

This test deliberately stops before external PerturbGen training.  It verifies
the part that can be proved locally without a compliant normal/disease donor
cohort: both route-specific DAVF checkpoints load, scVI encodes/decodes with
the saved ``davf_batch`` schema, the strict mapper uses the explicit KO/KD
code, and the direction gate emits a runnable PerturbGen invocation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.integration.perturbgen.contracts import PTMSiteDirectionProposal
from src.integration.perturbgen.orchestrator import DAVFPerturbGenOrchestrator
from src.models.davf_inference import DAVFInferenceModule
from src.models.scvi_adapter import ScVIAdapter, ScVIAdapterConfig, SCVI_AVAILABLE
from tests.real_assets import real_assets_enabled


pytestmark = [
    pytest.mark.real_assets,
    pytest.mark.skipif(
        not real_assets_enabled(),
        reason="set PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 to run the real DAVF bridge smoke",
    ),
    pytest.mark.skipif(not SCVI_AVAILABLE, reason="scvi-tools is not installed"),
]


@pytest.mark.parametrize(
    "route,symbol,ensembl_id,batch_value",
    [
        ("KO", "CREB1", "ENSG00000118260", "DixitRegev2016_K562_TFs_7_days:168"),
        ("KD", "NOC2L", "ENSG00000188976", "NadigOConner2024_jurkat:27"),
    ],
)
def test_real_davf_bridge_builds_gated_invocation(route, symbol, ensembl_id, batch_value):
    """Run both current route checkpoints through the real bridge."""

    suffix = "dixit" if route == "KO" else "nadig"
    config_path = Path(f"configs/davf_{route.lower()}.yaml")
    required = (
        config_path,
        Path(f"checkpoints/scvi/davf_{route.lower()}_{suffix}"),
        Path(f"checkpoints/davf/davf_{route.lower()}_{suffix}/best_model.pt"),
        Path("outputs/perturbgen/embedding_asset_20260822"),
    )
    if not all(path.exists() for path in required):
        pytest.skip(f"real {route} DAVF/scVI/embedding assets are not available")

    import anndata as ad
    from scipy import sparse

    from scripts.run_davf_perturbgen_e2e import _load_davf_config

    config = _load_davf_config(config_path)
    config.num_steps = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    adapter = ScVIAdapter.from_trained_model(
        config.scvi_model_path,
        config=ScVIAdapterConfig(
            model_path=config.scvi_model_path,
            n_latent=64,
            device=device,
        ),
    )
    values = np.zeros((1, adapter.n_genes), dtype=np.float32)
    values[0, 0] = 1.0
    context = ad.AnnData(sparse.csr_matrix(values), obs={"davf_batch": [batch_value]})
    context.var_names = list(adapter.gene_names)
    z_0 = adapter.encode(context)

    davf = DAVFInferenceModule(config)
    davf.bind_scvi_adapter(adapter)
    orchestrator = DAVFPerturbGenOrchestrator(davf_module=davf)
    raw_mapper = orchestrator.mapper.map_intervention_targets([symbol], route)
    evidence = davf.predict_expression_direction(
        raw_mapper,
        z_0,
        adapter,
        target_gene_symbols=[symbol],
        target_ensembl_ids=[ensembl_id],
        scvi_context=context,
        n_samples=1,
    )[0]
    if evidence.predicted_direction is None:
        pytest.fail(f"real {route} DAVF returned a near-zero target delta")
    observed_log2fc = 1.0 if evidence.predicted_direction == "up" else -1.0
    proposal = PTMSiteDirectionProposal(
        gene_symbol=symbol,
        ensembl_id=ensembl_id,
        position=12,
        ptm_type="phosphorylation",
        proposed_direction=evidence.predicted_direction,
        site_probability=0.95,
        provenance="real-asset-bridge-smoke",
    )

    preparation = orchestrator.prepare_candidate(
        proposal,
        z_0,
        cell_type="K562" if route == "KO" else "Jurkat",
        ptm_context=f"{symbol}:S12",
        observed_log2fc=observed_log2fc,
        observed_fdr=0.01,
        observed_direction=evidence.predicted_direction,
        scvi_adapter=adapter,
        scvi_context=context,
        n_samples=1,
    )

    assert preparation.status == "pass"
    assert preparation.invocation is not None
    assert preparation.invocation.intervention_type == route
    assert preparation.invocation.target_token_id == int(raw_mapper.gene_ids[0, 0])
    assert adapter.resolve_target_gene_indices([ensembl_id])[0] != preparation.invocation.target_token_id


def test_real_davf_e2e_cli_writes_direction_manifest(tmp_path):
    """Exercise the user-facing CLI with a small real scVI-compatible context."""

    required = (
        Path("configs/davf_ko.yaml"),
        Path("checkpoints/scvi/davf_ko_dixit"),
        Path("checkpoints/davf/davf_ko_dixit/best_model.pt"),
        Path("outputs/perturbgen/embedding_asset_20260822"),
    )
    if not all(path.exists() for path in required):
        pytest.skip("real KO DAVF/scVI/embedding assets are not available")

    import anndata as ad
    from scipy import sparse

    from scripts.run_davf_perturbgen_e2e import main
    from scripts.run_davf_perturbgen_e2e import _load_davf_config

    config = _load_davf_config("configs/davf_ko.yaml")
    config.num_steps = 2
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = tmp_path / "davf_ko.yaml"
    config_path.write_text(yaml.safe_dump({"davf": config.__dict__}), encoding="utf-8")

    from src.models.scvi_adapter import ScVIAdapter, ScVIAdapterConfig

    adapter = ScVIAdapter.from_trained_model(
        config.scvi_model_path,
        config=ScVIAdapterConfig(
            model_path=config.scvi_model_path,
            n_latent=64,
            device=config.device,
        ),
    )
    values = np.zeros((1, adapter.n_genes), dtype=np.float32)
    values[0, 0] = 1.0
    context_path = tmp_path / "context.h5ad"
    context = ad.AnnData(
        sparse.csr_matrix(values),
        obs={"davf_batch": ["DixitRegev2016_K562_TFs_7_days:168"]},
    )
    context.var_names = list(adapter.gene_names)
    context.write_h5ad(context_path)
    spec_path = tmp_path / "candidates.json"
    spec_path.write_text(
        json.dumps(
            {
                "context_h5ad": str(context_path),
                "candidates": [
                    {
                        "context_cell_index": 0,
                        "gene_symbol": "CREB1",
                        "ensembl_id": "ENSG00000118260",
                        "position": 12,
                        "ptm_type": "phosphorylation",
                        "proposed_direction": "down",
                        "site_probability": 0.95,
                        "provenance": "real-cli-smoke",
                        "cell_type": "K562",
                        "ptm_context": "CREB1:S12",
                        "observed_log2fc": -1.0,
                        "observed_fdr": 0.01,
                        "observed_direction": "down",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "ko_report.json"

    assert (
        main(
            [
                "--davf-config",
                str(config_path),
                "--candidate-spec",
                str(spec_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "davf_perturbgen_e2e/v1"
    assert payload["candidates"][0]["status"] == "pass"
    assert payload["candidates"][0]["invocation"]["intervention_type"] == "KO"


@pytest.mark.parametrize(
    "route,symbol,ensembl_id,proposed_direction,cell_type,context_path",
    [
        (
            "KO",
            "CREB1",
            "ENSG00000118260",
            "down",
            "K562",
            Path("data/processed/davf_scperturb/ko/prepared.h5ad"),
        ),
        (
            "KD",
            "NOC2L",
            "ENSG00000188976",
            "up",
            "Jurkat",
            Path("data/processed/davf_scperturb/kd/prepared.h5ad"),
        ),
    ],
)
def test_real_formal_candidate_config_drives_cli(
    tmp_path,
    route,
    symbol,
    ensembl_id,
    proposed_direction,
    cell_type,
    context_path,
):
    """The dated candidate config must select its own formal route checkpoint."""

    config_path = Path(f"configs/davf_{route.lower()}_cell_baseline_formal_20260904.yaml")
    route_name = "dixit" if route == "KO" else "nadig"
    required = (
        config_path,
        context_path,
        Path(f"checkpoints/scvi/davf_{route.lower()}_{route_name}"),
        Path(f"checkpoints/davf/davf_{route.lower()}_cell_baseline_formal_20260904/best_model.pt"),
        Path("outputs/perturbgen/embedding_asset_20260822"),
    )
    if not all(path.exists() for path in required):
        pytest.skip(f"real formal {route} candidate assets are not available")

    from scripts.run_davf_perturbgen_e2e import main

    spec_path = tmp_path / f"{route.lower()}_candidate.json"
    spec_path.write_text(
        json.dumps(
            {
                "context_h5ad": str(context_path),
                "candidates": [
                    {
                        "context_cell_index": 0,
                        "gene_symbol": symbol,
                        "ensembl_id": ensembl_id,
                        "position": 12,
                        "ptm_type": "phosphorylation",
                        "proposed_direction": proposed_direction,
                        "site_probability": 0.95,
                        "provenance": "real-formal-candidate-cli",
                        "cell_type": cell_type,
                        "ptm_context": f"{symbol}:S12",
                        "observed_log2fc": 1.0 if proposed_direction == "up" else -1.0,
                        "observed_fdr": 0.01,
                        "observed_direction": proposed_direction,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / f"{route.lower()}_formal_report.json"

    assert (
        main(
            [
                "--davf-config",
                str(config_path),
                "--candidate-spec",
                str(spec_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    candidate = payload["candidates"][0]
    assert candidate["status"] == "pass"
    assert candidate["invocation"]["intervention_type"] == route
    assert candidate["invocation"]["gene_symbol"] == symbol
    assert candidate["davf_evidence"]["predicted_direction"] == proposed_direction
