"""PTM-aware virtual perturbation helpers."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PTMPerturbationProfile:
    target_gene_index: int
    node_decay: float
    edge_scale: float


def apply_soft_perturbation(
    counts: np.ndarray,
    net: np.ndarray,
    profile: PTMPerturbationProfile,
) -> Tuple[np.ndarray, np.ndarray]:
    counts_new = np.array(counts, dtype=float, copy=True)
    net_new = np.array(net, dtype=float, copy=True)

    counts_new[:, profile.target_gene_index] *= profile.node_decay
    net_new[:, profile.target_gene_index] *= profile.edge_scale
    net_new[profile.target_gene_index, :] *= profile.edge_scale
    return counts_new, net_new
