"""
多任务PTM数据集和PTM位点数据集单元测试
"""

import os
import tempfile

import pandas as pd
import pytest
import torch

from src.data.multitask_dataset import (
    MultiTaskPTMDataset,
    MultiTaskPTMDataModule,
    collate_multitask_batch,
)
from src.data.ptm_site_dataset import PTMSiteDataset, PTMSiteDataModule


class TestMultiTaskPTMDataset:
    """MultiTaskPTMDataset类测试"""

    @pytest.fixture
    def temp_data_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "uniprot_id": ["P1", "P2", "P3", "P4"],
                "position": [10, 20, 30, 40],
                "aa": ["S", "T", "Y", "K"],
                "sequence_window": ["A" * 31, "C" * 31, "D" * 31, "E" * 31],
                "label": [1, 0, 1, 0],
            })
            phos_path = os.path.join(tmpdir, "ptm_train_phosphorylation.csv")
            acetyl_path = os.path.join(tmpdir, "ptm_train_acetylation.csv")
            df.to_csv(phos_path, index=False)
            df.to_csv(acetyl_path, index=False)
            yield {
                "Phosphorylation": phos_path,
                "Acetylation": acetyl_path,
            }

    def test_dataset_len(self, temp_data_files):
        dataset = MultiTaskPTMDataset(temp_data_files)
        assert len(dataset) == 8

    def test_getitem_returns_dict(self, temp_data_files):
        dataset = MultiTaskPTMDataset(temp_data_files)
        sample = dataset[0]
        assert "sequence_indices" in sample
        assert "label" in sample
        assert "ptm_type" in sample

    def test_get_type_weights(self, temp_data_files):
        dataset = MultiTaskPTMDataset(temp_data_files)
        weights = dataset.get_type_weights()
        assert "Phosphorylation" in weights
        assert "Acetylation" in weights
        assert weights["Phosphorylation"] > 0
        assert weights["Acetylation"] > 0


class TestMultiTaskPTMDataModule:
    """MultiTaskPTMDataModule类测试"""

    @pytest.fixture
    def temp_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "uniprot_id": [f"P{i}" for i in range(20)],
                "position": list(range(1, 21)),
                "aa": ["S"] * 20,
                "sequence_window": ["A" * 31] * 20,
                "label": ([1] * 10 + [0] * 10),
            })
            phos_path = os.path.join(tmpdir, "ptm_train_phosphorylation.csv")
            df.to_csv(phos_path, index=False)
            yield tmpdir

    def test_setup_splits_data(self, temp_data_dir):
        dm = MultiTaskPTMDataModule(
            data_dir=temp_data_dir,
            ptm_types=["Phosphorylation"],
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.2,
        )
        dm.setup()
        total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
        assert total == 20
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0
        assert len(dm.test_dataset) >= 0

    def test_train_dataloader_returns_batches(self, temp_data_dir):
        dm = MultiTaskPTMDataModule(
            data_dir=temp_data_dir,
            ptm_types=["Phosphorylation"],
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.2,
            num_workers=0,
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        assert train_loader is not None
        assert train_loader.batch_size == 4


class TestPTMSiteDataset:
    """PTMSiteDataset类测试"""

    @pytest.fixture
    def temp_site_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "uniprot_id": ["P1", "P2", "P3"],
                "position": [10, 20, 30],
                "aa": ["S", "T", "Y"],
                "sequence_window": ["ACDEFGHIKLMNPQRSTVWY"] * 3,
                "label": [1, 0, 1],
                "ptm_type": ["Phosphorylation"] * 3,
            })
            path = os.path.join(tmpdir, "sites.csv")
            df.to_csv(path, index=False)
            yield path

    def test_ptm_site_dataset_len(self, temp_site_csv):
        dataset = PTMSiteDataset(temp_site_csv)
        assert len(dataset) == 3

    def test_ptm_site_getitem(self, temp_site_csv):
        dataset = PTMSiteDataset(temp_site_csv)
        sample = dataset[0]
        assert "sequence_indices" in sample
        assert "label" in sample


class TestPTMSiteDataModule:
    """PTMSiteDataModule类测试"""

    @pytest.fixture
    def temp_site_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "uniprot_id": [f"P{i}" for i in range(30)],
                "position": list(range(1, 31)),
                "aa": ["S"] * 30,
                "sequence_window": ["ACDEFGHIKLMNPQRSTVWY"] * 30,
                "label": ([1] * 15 + [0] * 15),
                "ptm_type": ["Phosphorylation"] * 30,
            })
            path = os.path.join(tmpdir, "sites.csv")
            df.to_csv(path, index=False)
            yield path

    def test_ptm_site_setup_and_dataloader(self, temp_site_csv):
        dm = PTMSiteDataModule(
            train_path=temp_site_csv,
            batch_size=8,
            num_workers=0,
            val_split=0.2,
            test_split=0.1,
            seed=42,
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        assert train_loader is not None


class TestCollate:
    """collate函数测试"""

    def test_collate_multitask_batch_pads(self):
        batch = [
            {
                "sequence_indices": torch.tensor([1, 2, 3]),
                "label": torch.tensor(1),
                "ptm_type": "Phosphorylation",
                "ptm_idx": torch.tensor(0),
            },
            {
                "sequence_indices": torch.tensor([4, 5, 6]),
                "label": torch.tensor(0),
                "ptm_type": "Phosphorylation",
                "ptm_idx": torch.tensor(0),
            },
        ]
        result = collate_multitask_batch(batch)
        assert "sequence_indices" in result
        assert "labels" in result
        assert result["sequence_indices"].shape == (2, 3)
        assert result["labels"]["Phosphorylation"].shape == (2,)
