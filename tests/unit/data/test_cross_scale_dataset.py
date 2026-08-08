"""Unit tests for the versioned cross-scale NPZ contract."""

import numpy as np
import pytest

from src.data.cross_scale_dataset import (
    CROSS_SCALE_DATA_SCHEMA_VERSION,
    CrossScaleDataContractError,
    CrossScaleNPZDataset,
    cross_scale_collate,
)


def _write(path, *, samples=3, bad_version=False):
    length, genes = 4, 5
    np.savez(
        path,
        schema_version=np.array("bad" if bad_version else CROSS_SCALE_DATA_SCHEMA_VERSION),
        ankh39_embeddings=np.ones((samples, length, 2), dtype=np.float32),
        esm2_embeddings=np.ones((samples, length, 3), dtype=np.float32),
        prott5_embeddings=np.ones((samples, length, 4), dtype=np.float32),
        protein_attention_mask=np.ones((samples, length), dtype=np.float32),
        ptm_types=np.ones((samples, 1), dtype=np.int64),
        ptm_positions=np.ones((samples, 1), dtype=np.int64),
        signal_edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        signal_gene_map=np.ones((length, genes), dtype=np.float32),
        cell_edge_index=np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
        cell_gene_features=np.ones((samples, genes, 2), dtype=np.float32),
        gene_mask=np.ones((samples, genes), dtype=np.float32),
        delta_expression=np.zeros((samples, genes), dtype=np.float32),
        cell_state=np.arange(samples, dtype=np.int64) % 2,
        sample_id=np.array([f"s{index}" for index in range(samples)]),
    )


def test_dataset_validates_and_collates_static_graph_once(tmp_path):
    path = tmp_path / "split.npz"
    _write(path)
    dataset = CrossScaleNPZDataset(path)
    batch = cross_scale_collate([dataset[0], dataset[1]])

    assert len(dataset) == 3
    assert batch["inputs"]["ankh39_embeddings"].shape == (2, 4, 2)
    assert batch["inputs"]["signal_edge_index"].shape == (2, 3)
    assert batch["targets"]["delta_expression"].shape == (2, 5)
    assert dataset.contract()["sha256"]


def test_dataset_rejects_wrong_schema_and_nonfinite_embedding(tmp_path):
    bad_version = tmp_path / "bad-version.npz"
    _write(bad_version, bad_version=True)
    with pytest.raises(CrossScaleDataContractError, match="schema_version"):
        CrossScaleNPZDataset(bad_version)

    invalid = tmp_path / "invalid.npz"
    _write(invalid)
    with np.load(invalid, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["esm2_embeddings"][0, 0, 0] = np.nan
    np.savez(invalid, **arrays)
    with pytest.raises(CrossScaleDataContractError, match="有限浮点"):
        CrossScaleNPZDataset(invalid)
