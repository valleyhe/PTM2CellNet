import numpy as np

from src.integration.ptm_virtual_perturbation import PTMPerturbationProfile, apply_soft_perturbation


def test_soft_perturbation_scales_node_and_incident_edges() -> None:
    counts = np.ones((4, 3), dtype=float)
    net = np.array(
        [
            [0.0, 0.9, 0.0],
            [0.9, 0.0, 0.7],
            [0.0, 0.7, 0.0],
        ],
        dtype=float,
    )
    profile = PTMPerturbationProfile(target_gene_index=1, node_decay=0.4, edge_scale=0.5)

    counts_new, net_new = apply_soft_perturbation(counts, net, profile)

    assert counts_new[:, 1].mean() < counts[:, 1].mean()
    assert net_new[0, 1] == 0.45
    assert net_new[1, 2] == 0.35


def test_soft_profile_can_degenerate_to_hard_ko() -> None:
    profile = PTMPerturbationProfile(target_gene_index=0, node_decay=0.0, edge_scale=0.0)

    counts_new, net_new = apply_soft_perturbation(np.ones((2, 2)), np.ones((2, 2)), profile)

    assert counts_new[:, 0].sum() == 0.0
    assert net_new[0, :].sum() == 0.0
    assert net_new[:, 0].sum() == 0.0
