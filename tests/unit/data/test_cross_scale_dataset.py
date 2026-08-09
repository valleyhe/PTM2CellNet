"""Unit tests for the versioned cross-scale NPZ contract."""

import numpy as np
import pytest

from src.data.cross_scale_dataset import (
    CROSS_SCALE_DATA_SCHEMA_VERSION,
    CrossScaleDataContractError,
    CrossScaleNPZDataset,
    CrossScaleDataModule,
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


def _write_with_edge_weight(path, *, samples=3, weight_values=None, weight_dtype=np.float32):
    """带 cell_edge_weight 的合法 NPZ（4 条 cell 边，与 _write 图结构一致）。"""
    length, genes = 4, 5
    if weight_values is None:
        weight_values = np.array([0.5, 1.0, 0.8, 0.2], dtype=weight_dtype)
    np.savez(
        path,
        schema_version=np.array(CROSS_SCALE_DATA_SCHEMA_VERSION),
        ankh39_embeddings=np.ones((samples, length, 2), dtype=np.float32),
        esm2_embeddings=np.ones((samples, length, 3), dtype=np.float32),
        prott5_embeddings=np.ones((samples, length, 4), dtype=np.float32),
        signal_edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        signal_gene_map=np.ones((length, genes), dtype=np.float32),
        cell_edge_index=np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
        cell_edge_weight=weight_values,
        delta_expression=np.zeros((samples, genes), dtype=np.float32),
        cell_state=np.arange(samples, dtype=np.int64) % 2,
    )


def _load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


class TestCellEdgeWeightContract:
    """TD-M02: NPZ schema 承载 cell_edge_weight 的契约测试"""

    def test_edge_weight_passed_through_and_contract_reports(self, tmp_path):
        path = tmp_path / "weighted.npz"
        _write_with_edge_weight(path)
        dataset = CrossScaleNPZDataset(path)
        item = dataset[0]
        assert "cell_edge_weight" in item["inputs"]
        assert item["inputs"]["cell_edge_weight"].shape == (4,)
        assert dataset.contract()["has_cell_edge_weight"] is True
        # collate 按共享静态数组处理（不堆叠）
        batch = cross_scale_collate([dataset[0], dataset[1]])
        assert batch["inputs"]["cell_edge_weight"].shape == (4,)

    def test_edge_weight_absent_stays_compatible(self, tmp_path):
        path = tmp_path / "unweighted.npz"
        _write(path)
        dataset = CrossScaleNPZDataset(path)
        assert "cell_edge_weight" not in dataset[0]["inputs"]
        assert dataset.contract()["has_cell_edge_weight"] is False

    def test_edge_weight_shape_mismatch_rejected(self, tmp_path):
        path = tmp_path / "bad-shape.npz"
        _write_with_edge_weight(path, weight_values=np.array([0.5, 1.0], dtype=np.float32))
        with pytest.raises(CrossScaleDataContractError, match="边数一致"):
            CrossScaleNPZDataset(path)

    def test_edge_weight_nonfinite_rejected(self, tmp_path):
        path = tmp_path / "nan.npz"
        weights = np.array([0.5, np.nan, 0.8, 0.2], dtype=np.float32)
        _write_with_edge_weight(path, weight_values=weights)
        with pytest.raises(CrossScaleDataContractError, match="有限浮点"):
            CrossScaleNPZDataset(path)

    def test_edge_weight_negative_rejected(self, tmp_path):
        path = tmp_path / "negative.npz"
        weights = np.array([0.5, -0.1, 0.8, 0.2], dtype=np.float32)
        _write_with_edge_weight(path, weight_values=weights)
        with pytest.raises(CrossScaleDataContractError, match="非负"):
            CrossScaleNPZDataset(path)

    def test_edge_weight_integer_dtype_rejected(self, tmp_path):
        path = tmp_path / "int-dtype.npz"
        weights = np.array([1, 1, 1, 1], dtype=np.int64)
        _write_with_edge_weight(path, weight_values=weights)
        with pytest.raises(CrossScaleDataContractError, match="有限浮点"):
            CrossScaleNPZDataset(path)

    def test_split_compatibility_reports_edge_weight_independently(self, tmp_path):
        """train 带权、val 不带权时拆分仍兼容（缺省字段独立于兼容性检查）"""
        train_path = tmp_path / "train.npz"
        val_path = tmp_path / "val.npz"
        _write_with_edge_weight(train_path)
        _write(val_path)
        module = CrossScaleDataModule(train_path, val_path, val_path, batch_size=2)
        assert module.train_dataset.contract()["has_cell_edge_weight"] is True
        assert module.val_dataset.contract()["has_cell_edge_weight"] is False
