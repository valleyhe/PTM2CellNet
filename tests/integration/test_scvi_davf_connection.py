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
from src.models.scvi_adapter import SCVI_AVAILABLE, ScVIAdapter, ScVIAdapterConfig


SCVI_MODEL_PATH = Path("checkpoints/scvi/ibd_norman_model")
DAVF_CHECKPOINT_PATH = Path("checkpoints/davf/latent_davf_perturbgen_4018/best_model.pt")
EMBEDDING_ASSET_PATH = Path("outputs/perturbgen/embedding_asset_20260822")
LATENT_PAIR_PATH = Path("data/processed/davf_latent/norman_gse133344/train.npz")
GENE_ALIAS_PATH = Path("data/raw/norman_adamson/GSE133344/GSE133344_filtered_genes.tsv.gz")

# Current formal KD route (Nadig jurkat, canonical ENSG scVI gene order). The
# legacy Norman/Adamson checkpoint above is intentionally rejected by the
# canonical-ENSG gate (lessons L-2026-0914-03), so the direction test binds to
# this asset chain instead.
KD_SCVI_MODEL_PATH = Path("checkpoints/scvi/davf_kd_nadig")
KD_DAVF_CHECKPOINT_PATH = Path("checkpoints/davf/davf_kd_nadig/best_model.pt")
KD_LATENT_PAIR_PATH = Path("data/processed/davf_scperturb/kd/pairs/train.npz")
KD_GENE_ALIAS_PATH = Path("data/processed/davf_scperturb/kd/prepared.gene_aliases.tsv")
LCK_ENSEMBL_ID = "ENSG00000182866"
LCK_TOKEN_ID = 328
LCK_SCVI_DECODER_INDEX = 2260

# KO route extended with FrangiehIzar2021 (2026-09-16): adds real APOE KO
# perturbation signal; the axis is fully covered by the GSE174367 cohort.
KO_FRANGIEH_SCVI_MODEL_PATH = Path("checkpoints/scvi/davf_ko_frangieh")
KO_FRANGIEH_DAVF_CHECKPOINT_PATH = Path("checkpoints/davf/davf_ko_frangieh/best_model.pt")
KO_FRANGIEH_GENE_ALIAS_PATH = Path("data/processed/davf_scperturb/ko_frangieh/prepared.gene_aliases.tsv")
APOE_ENSEMBL_ID = "ENSG00000130203"
APOE_TOKEN_ID = 12707


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
        KD_SCVI_MODEL_PATH,
        KD_DAVF_CHECKPOINT_PATH,
        EMBEDDING_ASSET_PATH,
        KD_LATENT_PAIR_PATH,
        KD_GENE_ALIAS_PATH,
    )
    if not all(path.exists() for path in required):
        pytest.skip("current real DAVF/scVI/PerturbGen assets are not available")

    anndata = pytest.importorskip("anndata")
    from scipy import sparse

    from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule
    from src.models.scvi_adapter import ScVIAdapterConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = ScVIAdapter.from_trained_model(
        KD_SCVI_MODEL_PATH,
        config=ScVIAdapterConfig(
            model_path=str(KD_SCVI_MODEL_PATH),
            n_latent=64,
            device=device,
        ),
    )
    context = anndata.AnnData(
        X=sparse.csr_matrix((2, adapter.n_genes), dtype=np.float32),
        obs={"davf_batch": ["NadigOConner2024_jurkat:1", "NadigOConner2024_jurkat:1"]},
    )
    context.var_names = list(adapter.gene_names)
    pair_data = np.load(KD_LATENT_PAIR_PATH, allow_pickle=False)
    z_0 = torch.from_numpy(np.asarray(pair_data["z_0"][:2], dtype=np.float32))

    davf = DAVFInferenceModule(
        DAVFInferenceConfig(
            state_space="scvi_latent",
            checkpoint_path=str(KD_DAVF_CHECKPOINT_PATH),
            scvi_model_path=str(KD_SCVI_MODEL_PATH),
            embedding_asset_path=str(EMBEDDING_ASSET_PATH),
            gene_names_path=str(KD_GENE_ALIAS_PATH),
            latent_dim=64,
            num_genes=4018,
            num_steps=8,
            device=device,
        )
    )
    davf.bind_scvi_adapter(adapter)
    mapper = davf.build_perturbgen_direction_mapper()
    mapper_output = mapper.map_ptms(
        [{"type": "sumoylation"}],
        ["LCK"],
    )
    token_id = int(mapper_output.gene_ids[0, 0])
    scvi_index = adapter.resolve_target_gene_indices([LCK_ENSEMBL_ID])[0]

    assert mapper_output.attention_mask[0, 0].item() == 1.0
    assert token_id == LCK_TOKEN_ID
    assert scvi_index == LCK_SCVI_DECODER_INDEX
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
        target_ensembl_ids=[LCK_ENSEMBL_ID, LCK_ENSEMBL_ID],
        scvi_context=context,
        n_samples=1,
    )

    assert len(evidence) == 2
    assert all(item.model_source == "davf" for item in evidence)
    assert all(np.isfinite(item.predicted_delta) for item in evidence)


@pytest.mark.integration
@pytest.mark.skipif(not SCVI_AVAILABLE, reason="scvi-tools is not installed")
def test_real_ko_frangieh_route_serves_apoe_ko_direction():
    """Dixit+Frangieh KO chain decodes APOE, a target absent from the Dixit-only axis.

    The Frangieh extension (2026-09-16) adds real APOE perturbation signal to
    the KO route: 216 training targets (vs 10 in the Dixit-only chain) and a
    4018-gene axis fully covered by the GSE174367 cohort. Token and scVI
    decoder indices must stay separate (KO chain serves direction code 0).
    """

    required = (
        KO_FRANGIEH_SCVI_MODEL_PATH,
        KO_FRANGIEH_DAVF_CHECKPOINT_PATH,
        KO_FRANGIEH_GENE_ALIAS_PATH,
    )
    if not all(path.exists() for path in required):
        pytest.skip("KO Dixit+Frangieh real DAVF/scVI assets are not available")

    anndata = pytest.importorskip("anndata")
    from scipy import sparse

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = ScVIAdapter.from_trained_model(
        KO_FRANGIEH_SCVI_MODEL_PATH,
        config=ScVIAdapterConfig(
            model_path=str(KO_FRANGIEH_SCVI_MODEL_PATH),
            n_latent=64,
            device=device,
        ),
    )
    batch_values = [
        str(value)
        for value in np.asarray(
            adapter.model.registry_["field_registries"]["batch"]["state_registry"]["categorical_mapping"]
        ).ravel()
    ]
    context = anndata.AnnData(
        X=sparse.csr_matrix((2, adapter.n_genes), dtype=np.float32),
        obs={"davf_batch": [batch_values[0], batch_values[0]]},
    )
    context.var_names = list(adapter.gene_names)
    z_0 = torch.zeros((2, 64), dtype=torch.float32, device=device)

    davf = DAVFInferenceModule(
        DAVFInferenceConfig(
            state_space="scvi_latent",
            intervention_type="KO",
            checkpoint_path=str(KO_FRANGIEH_DAVF_CHECKPOINT_PATH),
            scvi_model_path=str(KO_FRANGIEH_SCVI_MODEL_PATH),
            embedding_asset_path=str(EMBEDDING_ASSET_PATH),
            gene_names_path=str(KO_FRANGIEH_GENE_ALIAS_PATH),
            latent_dim=64,
            num_genes=4018,
            num_steps=8,
            device=device,
        )
    )
    davf.bind_scvi_adapter(adapter)
    mapper = davf.build_perturbgen_direction_mapper()
    mapper_output = mapper.map_ptms(
        [{"type": "ubiquitination"}],
        ["APOE"],
    )
    token_id = int(mapper_output.gene_ids[0, 0])
    scvi_index = adapter.resolve_target_gene_indices([APOE_ENSEMBL_ID])[0]

    assert mapper_output.attention_mask[0, 0].item() == 1.0
    assert float(mapper_output.directions[0, 0]) == 0.0
    assert token_id == APOE_TOKEN_ID
    assert token_id != int(scvi_index)

    repeated = type(mapper_output)(
        gene_ids=mapper_output.gene_ids.repeat(2, 1),
        directions=mapper_output.directions.repeat(2, 1),
        attention_mask=mapper_output.attention_mask.repeat(2, 1),
    )
    evidence = davf.predict_expression_direction(
        repeated,
        z_0,
        target_gene_symbols=["APOE", "APOE"],
        target_ensembl_ids=[APOE_ENSEMBL_ID, APOE_ENSEMBL_ID],
        scvi_context=context,
        n_samples=1,
    )

    assert len(evidence) == 2
    assert all(item.model_source == "davf" for item in evidence)
    assert all(np.isfinite(item.predicted_delta) for item in evidence)
