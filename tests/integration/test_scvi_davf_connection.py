"""Real local scVI ↔ DAVF connection checks.

The first test stops at the scVI connection boundary. The second test uses
the current local PerturbGen-backed DAVF checkpoint and verifies the formal
direction path with the two distinct gene vocabularies.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule
from src.models.scvi_adapter import SCVI_AVAILABLE, ScVIAdapter


SCVI_MODEL_PATH = Path("checkpoints/scvi/ibd_norman_model")
DAVF_CHECKPOINT_PATH = Path("checkpoints/davf/latent_davf_perturbgen_4018/best_model.pt")
EMBEDDING_ASSET_PATH = Path("outputs/perturbgen/embedding_asset_20260822")
LATENT_PAIR_PATH = Path("data/processed/davf_latent/norman_gse133344/train.npz")
GENE_ALIAS_PATH = Path("data/raw/norman_adamson/GSE133344/GSE133344_filtered_genes.tsv.gz")


@pytest.mark.integration
@pytest.mark.skipif(not SCVI_AVAILABLE, reason="scvi-tools is not installed")
def test_real_scvi_checkpoint_connects_to_davf_schema():
    """Load the real local model, round-trip one context, and bind DAVF."""

    if not SCVI_MODEL_PATH.is_dir():
        pytest.skip(f"real scVI checkpoint is not available: {SCVI_MODEL_PATH}")

    anndata = pytest.importorskip("anndata")
    adapter = ScVIAdapter.from_trained_model(SCVI_MODEL_PATH)

    assert adapter.n_latent == 64
    assert adapter.n_genes == 4018
    assert len(adapter.gene_names) == adapter.n_genes

    adata = anndata.AnnData(
        X=np.ones((2, adapter.n_genes), dtype=np.float32),
        obs={"batch": ["ibd", "norman"], "dataset": ["ibd", "norman"]},
    )
    adata.var_names = list(adapter.gene_names)

    latent = adapter.encode(adata)
    decoded = adapter.decode(latent)

    assert latent.shape == (2, 64)
    assert decoded.shape == (2, 4018)
    assert np.isfinite(latent).all()
    assert np.isfinite(decoded).all()

    davf = DAVFInferenceModule(
        DAVFInferenceConfig(
            checkpoint_path="missing-current-davf.pt",
            scvi_model_path=str(SCVI_MODEL_PATH),
            latent_dim=adapter.n_latent,
            num_genes=adapter.n_genes,
            hidden_dim=32,
            feature_dim=16,
            device="cpu",
        )
    )
    bound_adapter = davf.load_scvi_adapter(adata)
    assert bound_adapter.n_latent == adapter.n_latent
    assert bound_adapter.n_genes == adapter.n_genes
    assert davf.scvi_adapter is bound_adapter

    target_index = bound_adapter.gene_names.index("IL6")
    bound_adapter.validate_target_gene_indices([target_index], ["IL6"])


@pytest.mark.integration
@pytest.mark.real_assets
@pytest.mark.skipif(not SCVI_AVAILABLE, reason="scvi-tools is not installed")
def test_real_current_davf_direction_keeps_token_and_decoder_indices_separate():
    """A real LCK request maps symbol→PerturbGen token and symbol→scVI row independently."""

    required = (
        SCVI_MODEL_PATH,
        DAVF_CHECKPOINT_PATH,
        EMBEDDING_ASSET_PATH,
        LATENT_PAIR_PATH,
        GENE_ALIAS_PATH,
    )
    if not all(path.exists() for path in required):
        pytest.skip("current real DAVF/scVI/PerturbGen assets are not available")

    anndata = pytest.importorskip("anndata")
    from scipy import sparse

    from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule
    from src.models.scvi_adapter import ScVIAdapterConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = ScVIAdapter.from_trained_model(
        SCVI_MODEL_PATH,
        config=ScVIAdapterConfig(
            model_path=str(SCVI_MODEL_PATH),
            n_latent=64,
            device=device,
        ),
    )
    context = anndata.AnnData(
        X=sparse.csr_matrix((2, adapter.n_genes), dtype=np.float32),
        obs={"batch": ["norman", "norman"], "dataset": ["norman", "norman"]},
    )
    context.var_names = list(adapter.gene_names)
    pair_data = np.load(LATENT_PAIR_PATH, allow_pickle=False)
    z_0 = torch.from_numpy(np.asarray(pair_data["z_0"][:2], dtype=np.float32))

    davf = DAVFInferenceModule(
        DAVFInferenceConfig(
            state_space="scvi_latent",
            checkpoint_path=str(DAVF_CHECKPOINT_PATH),
            scvi_model_path=str(SCVI_MODEL_PATH),
            embedding_asset_path=str(EMBEDDING_ASSET_PATH),
            gene_names_path=str(GENE_ALIAS_PATH),
            latent_dim=64,
            num_genes=4018,
            num_steps=8,
            device=device,
        )
    )
    davf.bind_scvi_adapter(adapter)
    mapper = davf.build_perturbgen_direction_mapper()
    mapper_output = mapper.map_ptms(
        [{"type": "phosphorylation"}],
        ["LCK"],
    )
    token_id = int(mapper_output.gene_ids[0, 0])
    scvi_index = adapter.resolve_target_gene_indices(["LCK"])[0]

    assert mapper_output.attention_mask[0, 0].item() == 1.0
    assert token_id == 328
    assert scvi_index == 113
    assert token_id != scvi_index

    repeated = type(mapper_output)(
        gene_ids=mapper_output.gene_ids.repeat(2, 1),
        directions=mapper_output.directions.repeat(2, 1),
        attention_mask=mapper_output.attention_mask.repeat(2, 1),
    )
    evidence = davf.predict_expression_direction(
        repeated,
        z_0,
        target_gene_symbols=["LCK", "LCK"],
        target_ensembl_ids=["ENSG00000182866", "ENSG00000182866"],
        scvi_context=context,
        n_samples=1,
    )

    assert len(evidence) == 2
    assert all(item.model_source == "davf" for item in evidence)
    assert all(np.isfinite(item.predicted_delta) for item in evidence)
