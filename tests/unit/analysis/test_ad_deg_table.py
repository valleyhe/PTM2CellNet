"""Unit tests for donor-level AD DEG tables."""

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from src.analysis.ad_deg_table import ADDEGError, build_ad_deg_tables, write_ad_deg_tables
from src.analysis.ptm_gene_score import load_deg_table as load_ptm_deg_table
from src.integration.perturbgen.replay_evaluation import load_deg_table as load_replay_deg_table
from src.integration.perturbgen.results import build_held_out_signature


def _adata():
    counts = np.array(
        [
            [2000, 8000],
            [3000, 7000],
            [6000, 4000],
            [7000, 3000],
        ],
        dtype=np.int64,
    )
    return anndata.AnnData(
        X=counts,
        layers={"counts": counts.copy()},
        obs=pd.DataFrame(
            {
                "cell_type": ["neuron"] * 4,
                "state": ["normal", "normal", "disease", "disease"],
                "donor": ["N1", "N2", "D1", "D2"],
            },
            index=["cell-1", "cell-2", "cell-3", "cell-4"],
        ),
        var=pd.DataFrame(
            {
                "ensembl_id": ["ENSG00000000001", "ENSG00000000002"],
                "gene_symbol": ["GENE1", "GENE2"],
            },
            index=["gene-1", "gene-2"],
        ),
    )


def _consumer_adata():
    counts = np.array(
        [
            [100, 9900],
            [110, 9890],
            [9900, 100],
            [9910, 90],
            [9890, 110],
        ],
        dtype=np.int64,
    )
    return anndata.AnnData(
        X=counts,
        layers={"counts": counts.copy()},
        obs=pd.DataFrame(
            {
                "cell_type": ["neuron"] * 5,
                "state": ["normal", "normal", "disease", "disease", "disease"],
                "donor": ["N1", "N2", "D1", "D2", "D3"],
            },
            index=["cell-1", "cell-2", "cell-3", "cell-4", "cell-5"],
        ),
        var=pd.DataFrame(
            {
                "ensembl_id": ["ENSG00000000001", "ENSG00000000002"],
                "gene_symbol": ["UP_GENE", "DOWN_GENE"],
            },
            index=["gene-1", "gene-2"],
        ),
    )


def test_build_ad_deg_tables_uses_donor_log2fc_and_disease_donors():
    aggregate, donor, audit = build_ad_deg_tables(
        _adata(), cell_types=["neuron"], cohort_pairing="between_donor", min_donors_per_state=2
    )

    assert set(aggregate.columns) == {
        "cell_type",
        "ensembl_id",
        "gene_symbol",
        "log2fc",
        "fdr",
        "observed_direction",
        "n_normal_donors",
        "n_disease_donors",
    }

    gene1 = aggregate.loc[aggregate["ensembl_id"] == "ENSG00000000001"].iloc[0]
    expected_gene1_log2fc = np.mean(
        [
            np.log2(6000 / 10000 * 10000 + 1),
            np.log2(7000 / 10000 * 10000 + 1),
        ]
    ) - np.mean(
        [
            np.log2(2000 / 10000 * 10000 + 1),
            np.log2(3000 / 10000 * 10000 + 1),
        ]
    )
    assert gene1["log2fc"] == pytest.approx(expected_gene1_log2fc)
    assert gene1["observed_direction"] == "up"
    assert gene1["n_normal_donors"] == 2
    assert gene1["n_disease_donors"] == 2
    assert audit["donor_counts"] == {"neuron": {"normal": 2, "disease": 2}}
    assert audit["normal_state"] == "normal"
    assert audit["disease_state"] == "disease"
    assert audit["normal_reference"] == "per cell type normal donor-level log2(normalized counts) mean"
    assert audit["effect_scale"] == "disease donor mean minus normal donor mean"

    assert len(donor) == 4
    assert donor.groupby("ensembl_id").size().to_dict() == {
        "ENSG00000000001": 2,
        "ENSG00000000002": 2,
    }
    assert all(set(rows["donor"]) == {"D1", "D2"} for _, rows in donor.groupby("ensembl_id"))
    assert donor["log2fc"].notna().all()
    assert donor["fdr"].notna().all()


def test_written_ad_deg_tables_satisfy_both_consumers(tmp_path):
    aggregate, donor, _ = build_ad_deg_tables(
        _consumer_adata(), cell_types=["neuron"], cohort_pairing="between_donor", min_donors_per_state=2
    )
    aggregate_path = tmp_path / "aggregate.tsv"
    donor_path = tmp_path / "donor.csv"

    write_ad_deg_tables(aggregate, donor, aggregate_path, donor_path)

    loaded_aggregate = load_ptm_deg_table(aggregate_path)
    assert len(loaded_aggregate.columns) == 8

    loaded_donor = load_replay_deg_table(donor_path)
    signature = build_held_out_signature(
        loaded_donor,
        held_out_donor="D3",
        donor_column="donor",
        gene_column="gene_symbol",
        effect_column="log2fc",
        fdr_column="fdr",
        min_training_donors=2,
    )

    assert signature.training_donors == ("D1", "D2")
    assert signature.up_genes == ("UP_GENE",)
    assert signature.down_genes == ("DOWN_GENE",)


def test_rejects_donor_present_in_both_states():
    adata = _adata()
    adata.obs.loc["cell-3", "donor"] = "N1"

    with pytest.raises(ADDEGError, match="donor cannot occur in both states"):
        build_ad_deg_tables(adata, cell_types=["neuron"], cohort_pairing="between_donor", min_donors_per_state=2)


def test_rejects_cell_type_without_donors_in_one_state():
    adata = _adata()
    adata.obs.loc[:, "state"] = "normal"

    with pytest.raises(ADDEGError, match="requires at least 2 donors"):
        build_ad_deg_tables(adata, cell_types=["neuron"], cohort_pairing="between_donor", min_donors_per_state=2)


def test_rejects_fewer_than_configured_donors():
    with pytest.raises(ADDEGError, match="requires at least 3 donors"):
        build_ad_deg_tables(_adata(), cell_types=["neuron"], cohort_pairing="between_donor", min_donors_per_state=3)


def test_rejects_donor_leakage_across_cell_types():
    adata = _adata()
    adata.obs.loc["cell-3", "cell_type"] = "other"
    adata.obs.loc["cell-3", "donor"] = "N1"
    with pytest.raises(ADDEGError, match="globally disjoint"):
        build_ad_deg_tables(adata, cell_types=["neuron"], cohort_pairing="between_donor", min_donors_per_state=1)


def test_rejects_within_donor_pairing_for_independent_deg():
    with pytest.raises(ADDEGError, match="requires between_donor pairing"):
        build_ad_deg_tables(_adata(), cell_types=["neuron"], cohort_pairing="within_donor", min_donors_per_state=2)


def _pseudobulk_adata():
    counts = np.array(
        [
            [2, 8],
            [4, 6],
            [8, 2],
            [9, 1],
        ],
        dtype=np.int64,
    )
    return anndata.AnnData(
        X=counts,
        layers={"counts": counts.copy()},
        obs=pd.DataFrame(
            {
                "cell_type": ["neuron"] * 4,
                "state": ["normal", "normal", "disease", "disease"],
                "donor": ["N1", "N2", "D1", "D2"],
            },
            index=["cell-1", "cell-2", "cell-3", "cell-4"],
        ),
        var=pd.DataFrame(
            {
                "ensembl_id": ["ENSG00000000001", "ENSG00000000002"],
                "gene_symbol": ["GENE1", "GENE2"],
            },
            index=["gene-1", "gene-2"],
        ),
    )


def test_pseudobulk_aggregation_matches_manual_donor_bulk_profiles():
    aggregate, _, audit = build_ad_deg_tables(
        _pseudobulk_adata(),
        cell_types=["neuron"],
        cohort_pairing="between_donor",
        min_donors_per_state=2,
        donor_aggregation="pseudobulk_counts",
    )

    # donor N1: bulk [2, 8], library 10 -> log2(2*1000+1), log2(8*1000+1)
    expected_n1 = np.log2(np.array([2.0, 8.0]) * 1000.0 + 1.0)
    # donor D1: bulk [8, 2] -> log2(8*1000+1), log2(2*1000+1)
    expected_d1 = np.log2(np.array([8.0, 2.0]) * 1000.0 + 1.0)
    expected_n2 = np.log2(np.array([4.0, 6.0]) * 1000.0 + 1.0)
    expected_d2 = np.log2(np.array([9.0, 1.0]) * 1000.0 + 1.0)
    row_gene1 = aggregate[aggregate["ensembl_id"] == "ENSG00000000001"].iloc[0]
    row_gene2 = aggregate[aggregate["ensembl_id"] == "ENSG00000000002"].iloc[0]

    assert row_gene1["log2fc"] == pytest.approx(((expected_d1 + expected_d2) / 2 - (expected_n1 + expected_n2) / 2)[0])
    assert row_gene2["log2fc"] == pytest.approx(((expected_d1 + expected_d2) / 2 - (expected_n1 + expected_n2) / 2)[1])
    assert row_gene1["observed_direction"] == "up"
    assert row_gene2["observed_direction"] == "down"
    assert audit["donor_aggregation"] == "pseudobulk_counts"
    assert "pseudobulk" in audit["normal_reference"]


def test_pseudobulk_keeps_donor_library_composition_that_per_cell_averaging_removes():
    # Both donors have the same per-cell fractions, so per-cell normalization
    # sees identical profiles; the disease donor carries twice the library.
    counts = np.array(
        [
            [100, 100],
            [100, 100],
            [200, 200],
            [200, 200],
        ],
        dtype=np.int64,
    )
    adata = anndata.AnnData(
        X=counts,
        layers={"counts": counts.copy()},
        obs=pd.DataFrame(
            {
                "cell_type": ["neuron"] * 4,
                "state": ["normal", "normal", "disease", "disease"],
                "donor": ["N1", "N2", "D1", "D2"],
            },
            index=["cell-1", "cell-2", "cell-3", "cell-4"],
        ),
        var=pd.DataFrame(
            {
                "ensembl_id": ["ENSG00000000001", "ENSG00000000002"],
                "gene_symbol": ["GENE1", "GENE2"],
            },
            index=["gene-1", "gene-2"],
        ),
    )

    per_cell, _, per_cell_audit = build_ad_deg_tables(
        adata, cell_types=["neuron"], cohort_pairing="between_donor", min_donors_per_state=2
    )
    pseudobulk, _, pseudobulk_audit = build_ad_deg_tables(
        adata,
        cell_types=["neuron"],
        cohort_pairing="between_donor",
        min_donors_per_state=2,
        donor_aggregation="pseudobulk_counts",
    )

    per_cell_delta = per_cell["log2fc"].to_numpy()
    pseudobulk_delta = pseudobulk["log2fc"].to_numpy()
    assert np.allclose(per_cell_delta, 0.0)
    assert np.allclose(pseudobulk_delta, 0.0)
    assert per_cell_audit["donor_aggregation"] == "per_cell_log2_mean"
    assert pseudobulk_audit["donor_aggregation"] == "pseudobulk_counts"


def test_pseudobulk_separates_profiles_that_per_cell_averaging_blurs():
    # Per-cell fractions differ inside each donor only through composition:
    # normal donors are 30/70, disease donors are 80/20. Under per-cell
    # normalization the delta is the mean of per-cell log2 fold changes;
    # under pseudobulk it is the difference of aggregated profiles. Both
    # directions must agree in sign, but the numeric estimands differ.
    rng = np.random.default_rng(11)
    counts = np.vstack(
        [
            rng.multinomial(10_000, [0.3, 0.7], size=4),  # normal donors N1, N2
            rng.multinomial(10_000, [0.8, 0.2], size=4),  # disease donors D1, D2
        ]
    ).astype(np.int64)
    obs = pd.DataFrame(
        {
            "cell_type": ["neuron"] * 8,
            "state": ["normal"] * 4 + ["disease"] * 4,
            "donor": ["N1", "N1", "N2", "N2", "D1", "D1", "D2", "D2"],
        },
        index=[f"cell-{i}" for i in range(8)],
    )
    var = pd.DataFrame(
        {
            "ensembl_id": ["ENSG00000000001", "ENSG00000000002"],
            "gene_symbol": ["GENE1", "GENE2"],
        },
        index=["gene-1", "gene-2"],
    )
    adata = anndata.AnnData(X=counts, layers={"counts": counts.copy()}, obs=obs, var=var)

    per_cell, _, _ = build_ad_deg_tables(
        adata, cell_types=["neuron"], cohort_pairing="between_donor", min_donors_per_state=2
    )
    pseudobulk, _, _ = build_ad_deg_tables(
        adata,
        cell_types=["neuron"],
        cohort_pairing="between_donor",
        min_donors_per_state=2,
        donor_aggregation="pseudobulk_counts",
    )

    for frame in (per_cell, pseudobulk):
        directions = dict(zip(frame["ensembl_id"], frame["observed_direction"], strict=True))
        assert directions["ENSG00000000001"] == "up"
        assert directions["ENSG00000000002"] == "down"
    assert not np.allclose(per_cell["log2fc"].to_numpy(), pseudobulk["log2fc"].to_numpy())


def test_rejects_unknown_donor_aggregation():
    with pytest.raises(ADDEGError, match="donor_aggregation must be one of"):
        build_ad_deg_tables(
            _adata(),
            cell_types=["neuron"],
            cohort_pairing="between_donor",
            min_donors_per_state=2,
            donor_aggregation="median_of_medians",
        )


def test_pseudobulk_rejects_zero_library_donor():
    counts = np.array(
        [
            [1, 9],
            [4, 6],
            [8, 2],
            [9, 1],
            [0, 0],
        ],
        dtype=np.int64,
    )
    adata = anndata.AnnData(
        X=counts,
        layers={"counts": counts.copy()},
        obs=pd.DataFrame(
            {
                "cell_type": ["neuron"] * 5,
                "state": ["normal", "normal", "disease", "disease", "disease"],
                "donor": ["N1", "N2", "D1", "D2", "D3"],
            },
            index=[f"cell-{i}" for i in range(5)],
        ),
        var=pd.DataFrame(
            {
                "ensembl_id": ["ENSG00000000001", "ENSG00000000002"],
                "gene_symbol": ["GENE1", "GENE2"],
            },
            index=["gene-1", "gene-2"],
        ),
    )

    with pytest.raises(ADDEGError, match="zero-library donor pseudobulk"):
        build_ad_deg_tables(
            adata,
            cell_types=["neuron"],
            cohort_pairing="between_donor",
            min_donors_per_state=2,
            donor_aggregation="pseudobulk_counts",
        )
