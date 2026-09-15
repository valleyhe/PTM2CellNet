import json

import pandas as pd
import torch

from src.data import PTMLightningDataModule, PTMPlainDataModule, PTMDataModule
from src.data.datasets import PTMDataset
from src.data.schemas import PTMRecord, PTMSite, validate_protein_data


class SequenceShiftAugmenter:
    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        augmented = sequence.clone()
        augmented[augmented > 0] += 1
        return augmented


class PTMZeroAugmenter:
    def __call__(self, ptm_mask: torch.Tensor, ptm_types: torch.Tensor):
        return torch.zeros_like(ptm_mask), torch.zeros_like(ptm_types)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence": "ACDE",
                "ptm_sites": json.dumps([{"position": 2, "type": "phosphorylation"}]),
                "cell_state": "activated",
            }
        ]
    )


def test_datamodule_alias_points_to_lightning_class():
    assert PTMDataModule is PTMLightningDataModule
    assert PTMLightningDataModule.__name__ == "PTMLightningDataModule"


def test_dataset_defaults_to_training_mode():
    dataset = PTMDataset(_sample_df(), config={"data": {"max_sequence_length": 4}})
    assert dataset.training is True


def test_dataset_returns_feature_extractor_outputs_when_enabled():
    dataset = PTMDataset(
        _sample_df(),
        config={
            "data": {
                "max_sequence_length": 4,
                "ptm_types": ["phosphorylation", "acetylation"],
            },
            "features": {
                "use_feature_extractor": True,
                "sequence_encoding": "onehot",
                "include_physicochemical": False,
                "include_ptm_features": True,
                "include_structural_features": False,
            },
        },
        use_feature_extractor=True,
    )

    sample = dataset[0]

    assert "sequence_features" in sample
    assert "ptm_features" in sample
    assert sample["sequence_features"].dtype == torch.float32
    assert sample["ptm_features"].dtype == torch.float32


def test_dataset_builds_augmenters_from_string_preset():
    dataset = PTMDataset(
        _sample_df(),
        config={
            "data": {"max_sequence_length": 4, "ptm_types": ["phosphorylation"]},
            "augmentation": "medium",
        },
    )

    assert dataset.sequence_augmenter is not None
    assert dataset.ptm_augmenter is not None
    assert dataset.sequence_augmenter.augment_prob == 0.5
    assert dataset.ptm_augmenter.drop_prob == 0.1


def test_dataset_applies_augmentation_only_in_training_mode():
    dataset = PTMDataset(
        _sample_df(),
        config={"data": {"max_sequence_length": 4}},
        training=True,
        sequence_augmenter=SequenceShiftAugmenter(),
        ptm_augmenter=PTMZeroAugmenter(),
    )

    sample = dataset[0]

    assert torch.equal(sample["sequence"][:4], torch.tensor([2, 3, 4, 5], dtype=torch.long))
    assert sample["ptm_mask"].sum().item() == 0.0
    assert sample["ptm_types"].sum().item() == 0

    eval_dataset = PTMDataset(
        _sample_df(),
        config={"data": {"max_sequence_length": 4}},
        training=False,
        sequence_augmenter=SequenceShiftAugmenter(),
        ptm_augmenter=PTMZeroAugmenter(),
    )
    eval_sample = eval_dataset[0]
    assert torch.equal(eval_sample["sequence"][:4], torch.tensor([1, 2, 3, 4], dtype=torch.long))
    assert eval_sample["ptm_mask"].sum().item() == 1.0


def test_lightning_datamodule_sets_training_flags():
    df = pd.concat([_sample_df(), _sample_df()], ignore_index=True)
    dm = PTMLightningDataModule(
        train_df=df,
        val_df=df.iloc[:1].copy(),
        test_df=df.iloc[:1].copy(),
        config={
            "data": {"max_sequence_length": 4, "ptm_types": ["phosphorylation"]},
            "training": {"batch_size": 2},
        },
    )
    dm.setup("fit")

    assert dm.train_dataset.training is True
    assert dm.val_dataset.training is False

    dm.setup("test")
    assert dm.test_dataset.training is False


def test_plain_datamodule_sets_training_flags_and_propagates_feature_flag():
    df = pd.concat([_sample_df(), _sample_df()], ignore_index=True)
    datamodule = PTMPlainDataModule(
        train_df=df,
        val_df=df.iloc[:1].copy(),
        test_df=df.iloc[:1].copy(),
        config={
            "data": {
                "max_sequence_length": 4,
                "ptm_types": ["phosphorylation", "acetylation"],
            },
            "features": {
                "use_feature_extractor": True,
                "sequence_encoding": "onehot",
                "include_physicochemical": False,
                "include_ptm_features": True,
                "include_structural_features": False,
            },
            "augmentation": {
                "sequence_augment_prob": 0.0,
                "ptm_drop_prob": 0.0,
            },
        },
        use_feature_extractor=True,
    )

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    assert datamodule.train_dataset.training is True
    assert datamodule.val_dataset.training is False
    assert datamodule.test_dataset.training is False
    assert datamodule.train_dataset.use_feature_extractor is True
    assert datamodule.train_dataset.sequence_augmenter is not None
    assert datamodule.train_dataset.ptm_augmenter is not None


def test_schema_helpers_round_trip_and_validation():
    site = PTMSite(position=3, type="phosphorylation", amino_acid="D")
    record = PTMRecord(
        protein_accession="P12345",
        position=3,
        ptm_type="phosphorylation",
        amino_acid="D",
        confidence=0.8,
        source="unit-test",
    )

    assert PTMSite.from_dict(site.to_dict()) == site
    assert PTMRecord.from_dict(record.to_dict()) == record
    assert (
        validate_protein_data(
            {
                "accession": "P12345",
                "sequence": "ACDE",
                "ptm_sites": [site.to_dict()],
            }
        )
        is True
    )
    assert validate_protein_data({"accession": "", "sequence": "ACDE"}) is False
