"""Perturbation execution pipeline for GenKI and array-file backends."""

import importlib
from pathlib import Path
import sys
from typing import Any, List, Optional, TYPE_CHECKING

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from ..contracts import GenePerturbationRequest, PerturbationResult
from ..ptm_virtual_perturbation import PTMPerturbationProfile, apply_soft_perturbation
from .graph_utils import GraphUtilities
from .reference_data import ReferenceDataLoader

if TYPE_CHECKING:
    from .significance import SignificanceAnalyzer


class PerturbationExecutor:
    """Execute perturbation requests and compute perturbation scores."""

    def __init__(
        self,
        ref_loader: ReferenceDataLoader,
        gene_list_file: str | None = None,
        network_file: str | None = None,
        counts_file: str | None = None,
        adata_file: str | None = None,
        grn_file_dir: str | None = None,
        pcnet_name: str = "pcNet",
        cutoff: int = 85,
        target_cell: str | None = None,
        obs_label: str = "ident",
        scoring_method: str = "shift",
        trainer_epochs: int = 10,
        trainer_lr: float = 7e-4,
        trainer_beta: float = 1e-4,
        trainer_seed: int | None = None,
        trainer_out_channels: int = 2,
        null_permutations: int = 32,
        null_seed: int | None = None,
        significance_alpha: float = 0.05,
        bagging_threshold: float = 0.05,
        bagging_cutoff: float = 0.95,
        graph: GraphUtilities | None = None,
        significance_analyzer: Optional["SignificanceAnalyzer"] = None,
    ) -> None:
        self._ref_loader = ref_loader
        self.gene_list_file = gene_list_file
        self.network_file = network_file
        self.counts_file = counts_file
        self.adata_file = adata_file
        self.grn_file_dir = grn_file_dir
        self.pcnet_name = pcnet_name
        self.cutoff = cutoff
        self.target_cell = target_cell
        self.obs_label = obs_label
        self.scoring_method = scoring_method
        self.trainer_epochs = trainer_epochs
        self.trainer_lr = trainer_lr
        self.trainer_beta = trainer_beta
        self.trainer_seed = trainer_seed
        self.trainer_out_channels = trainer_out_channels
        self.null_permutations = max(0, int(null_permutations))
        self.null_seed = null_seed
        self.significance_alpha = float(significance_alpha)
        self.bagging_threshold = float(bagging_threshold)
        self.bagging_cutoff = float(bagging_cutoff)
        self._graph = graph or GraphUtilities()
        self._significance_analyzer = significance_analyzer

    def set_significance_analyzer(self, analyzer: "SignificanceAnalyzer") -> None:
        self._significance_analyzer = analyzer

    def _require_significance(self) -> "SignificanceAnalyzer":
        if self._significance_analyzer is None:
            raise RuntimeError("SignificanceAnalyzer is not configured for PerturbationExecutor")
        return self._significance_analyzer

    def build_request(self, gene_symbol: str, mode: str, magnitude: float) -> GenePerturbationRequest:
        return GenePerturbationRequest(
            gene_symbol=gene_symbol,
            source_protein_id="",
            source_ptm_type="",
            source_ptm_position=-1,
            magnitude=magnitude,
            mode=mode,
        )

    def run(self, request: GenePerturbationRequest) -> PerturbationResult:
        reference = self._ref_loader.load_reference_data()
        if reference["backend"] == "genki_source":
            return self._run_with_genki_source(request)

        gene_names: List[str] = reference["gene_names"]
        if request.gene_symbol not in gene_names:
            raise KeyError(f"Unknown gene symbol: {request.gene_symbol}")

        gene_index = gene_names.index(request.gene_symbol)
        baseline_counts = np.asarray(reference["counts"], dtype=float)
        baseline_network = np.asarray(reference["network"], dtype=float)
        perturbed_counts = baseline_counts.copy()
        perturbed_network = baseline_network.copy()

        if request.mode == "hard_ko":
            perturbed_counts[:, gene_index] = 0.0
            perturbed_network[:, gene_index] = 0.0
            perturbed_network[gene_index, :] = 0.0
        elif request.mode == "soft_ptm":
            profile = PTMPerturbationProfile(
                target_gene_index=gene_index,
                node_decay=max(0.0, 1.0 - request.magnitude),
                edge_scale=max(0.0, 1.0 - request.magnitude),
            )
            perturbed_counts, perturbed_network = apply_soft_perturbation(
                baseline_counts,
                baseline_network,
                profile,
            )
        else:
            raise ValueError(f"Unsupported perturbation mode: {request.mode}")

        combined_shift = self._graph._score_from_dense_matrices(
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            perturbed_counts=perturbed_counts,
            perturbed_network=perturbed_network,
        )
        ranked_indices = np.argsort(-combined_shift)
        ranked_genes = [gene_names[idx] for idx in ranked_indices if gene_names[idx] != request.gene_symbol]
        distance_score = float(np.linalg.norm(combined_shift, ord=2))
        metadata = self._require_significance()._build_score_metadata(
            gene_names=gene_names,
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            combined_shift=combined_shift,
            backend=reference["backend"],
        )

        return PerturbationResult(
            gene_symbol=request.gene_symbol,
            mode=request.mode,
            distance_score=distance_score,
            ranked_genes=ranked_genes,
            metadata=metadata,
        )

    def run_virtual_ko(self, gene_symbol: str, top_k: int = 20) -> PerturbationResult:
        result = self.run(self.build_request(gene_symbol=gene_symbol, mode="hard_ko", magnitude=1.0))
        return PerturbationResult(
            gene_symbol=result.gene_symbol,
            mode=result.mode,
            distance_score=result.distance_score,
            ranked_genes=result.ranked_genes[:top_k],
            metadata={**result.metadata, "top_k": top_k},
        )

    def _run_with_genki_source(self, request: GenePerturbationRequest) -> PerturbationResult:
        loader = self._build_genki_loader(request.gene_symbol)
        wt_data = loader.load_data()
        gene_names = [str(gene_name) for gene_name in wt_data.y]
        if request.gene_symbol not in gene_names:
            raise KeyError(f"Unknown gene symbol: {request.gene_symbol}")

        gene_index = gene_names.index(request.gene_symbol)
        baseline_counts = wt_data.x.detach().cpu().numpy().T
        baseline_network = self._graph._edge_index_to_adjacency(
            wt_data.edge_index.detach().cpu().numpy(), len(gene_names)
        )

        if request.mode == "hard_ko":
            perturbed_data = loader.load_kodata()
            perturbed_counts = perturbed_data.x.detach().cpu().numpy().T
            perturbed_network = self._graph._edge_index_to_adjacency(
                perturbed_data.edge_index.detach().cpu().numpy(),
                len(gene_names),
            )
        elif request.mode == "soft_ptm":
            raw_counts = (
                loader.counts.toarray()
                if sp.issparse(loader.counts)
                else np.asarray(loader.counts, dtype=float)
            )
            raw_network = loader.net.toarray() if sp.issparse(loader.net) else np.asarray(loader.net, dtype=float)
            profile = PTMPerturbationProfile(
                target_gene_index=gene_index,
                node_decay=max(0.0, 1.0 - request.magnitude),
                edge_scale=max(0.0, 1.0 - request.magnitude),
            )
            perturbed_counts, perturbed_network_dense = apply_soft_perturbation(raw_counts, raw_network, profile)
            perturbed_network = self._graph._edge_index_to_adjacency(
                np.asarray(loader._build_edges(perturbed_network_dense), dtype=int),
                len(gene_names),
            )
        else:
            raise ValueError(f"Unsupported perturbation mode: {request.mode}")

        if self.scoring_method == "latent_vgae":
            combined_shift = self._score_with_latent_vgae(
                wt_data=wt_data,
                perturbed_counts=perturbed_counts,
                perturbed_network=perturbed_network,
            )
        else:
            count_shift = np.abs(perturbed_counts.mean(axis=0) - baseline_counts.mean(axis=0))
            edge_shift = np.abs(perturbed_network - baseline_network).sum(axis=0) + np.abs(
                perturbed_network - baseline_network
            ).sum(axis=1)
            combined_shift = count_shift + edge_shift

        ranked_indices = np.argsort(-combined_shift)
        ranked_genes = [gene_names[idx] for idx in ranked_indices if gene_names[idx] != request.gene_symbol]
        distance_score = float(np.linalg.norm(combined_shift, ord=2))
        raw_counts = loader.counts.toarray() if sp.issparse(loader.counts) else np.asarray(loader.counts, dtype=float)
        raw_network = loader.net.toarray() if sp.issparse(loader.net) else np.asarray(loader.net, dtype=float)
        metadata = self._require_significance()._build_score_metadata(
            gene_names=gene_names,
            gene_index=gene_index,
            request=request,
            baseline_counts=raw_counts,
            baseline_network=raw_network,
            combined_shift=combined_shift,
            backend="genki_source",
        )

        return PerturbationResult(
            gene_symbol=request.gene_symbol,
            mode=request.mode,
            distance_score=distance_score,
            ranked_genes=ranked_genes,
            metadata=metadata,
        )

    def _build_genki_loader(self, gene_symbol: str):
        self._ref_loader.validate_runtime_ready()
        if self.adata_file is None or self.grn_file_dir is None:
            raise ValueError("genki_source backend requires adata_file and grn_file_dir")

        ref_root = str(Path(self._ref_loader.ref_root))
        if ref_root not in sys.path:
            sys.path.insert(0, ref_root)
        module = importlib.import_module("GenKI.dataLoader")
        data_loader_cls = getattr(module, "DataLoader")
        adata = ad.read_h5ad(self.adata_file)
        return data_loader_cls(
            adata=adata,
            target_gene=[gene_symbol],
            target_cell=self.target_cell,
            obs_label=self.obs_label,
            GRN_file_dir=self.grn_file_dir,
            rebuild_GRN=False,
            pcNet_name=self.pcnet_name,
            cutoff=self.cutoff,
            verbose=False,
        )

    def _score_with_latent_vgae(
        self,
        wt_data: Any,
        perturbed_counts: np.ndarray,
        perturbed_network: np.ndarray,
    ) -> np.ndarray:
        train_module = importlib.import_module("GenKI.train")
        utils_module = importlib.import_module("GenKI.utils")
        data_cls = importlib.import_module("torch_geometric.data").Data
        vgae_cls = getattr(train_module, "VGAE")
        encoder_cls = getattr(train_module, "VariationalGCNEncoder")
        get_distance = getattr(utils_module, "get_distance")
        device = torch.device("cpu")
        if self.trainer_seed is not None:
            torch.manual_seed(self.trainer_seed)

        model = vgae_cls(encoder_cls(wt_data.num_features, self.trainer_out_channels)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.trainer_lr)
        wt_train = wt_data.to(device)
        for _ in range(self.trainer_epochs):
            model.train()
            optimizer.zero_grad()
            z = model.encode(wt_train.x, wt_train.edge_index)
            recon_loss = model.recon_loss(z, wt_train.edge_index)
            kl_loss = self.trainer_beta * model.kl_loss()
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

        perturbed_edge_index = self._graph._adjacency_to_edge_index(perturbed_network)
        data_v = data_cls(
            x=torch.tensor(np.asarray(perturbed_counts, dtype=float).T, dtype=torch.float),
            edge_index=torch.tensor(perturbed_edge_index, dtype=torch.long),
            y=wt_data.y,
        )
        z_mu, z_std = self._graph._extract_latent_vars(model, wt_data.to(device))
        z_mu_v, z_std_v = self._graph._extract_latent_vars(model, data_v.to(device))
        return np.asarray(get_distance(z_mu_v, z_std_v, z_mu, z_std, by="KL"), dtype=float)
