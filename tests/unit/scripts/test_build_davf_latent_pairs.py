"""Unit tests for the strict Norman-to-DAVF latent-pair builder."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import scripts.build_davf_latent_pairs as builder


def _write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def _make_norman(root: Path, *, missing_guide: bool = False) -> Path:
    dataset = root / "GSE133344"
    dataset.mkdir()
    # The first two raw genes are deliberately reversed relative to the
    # adapter order.  The remaining genes make the formal 4018-gene fixture.
    target_symbols = ["AHR", "FEV", "KLF1", "BCL2L11", "BAK1", "C3orf72", "FOXL2", "CBL", "UBASH3A"]
    genes = ["ENSG1\tAHR", "ENSG0\tFEV"]
    genes.extend(f"ENSG{i}\t{symbol}" for i, symbol in enumerate(target_symbols[2:], start=2))
    genes.extend(f"ENSG{i}\tG{i}" for i in range(len(genes), 4018))
    _write_gz(dataset / "GSE133344_filtered_genes.tsv.gz", "\n".join(genes) + "\n")
    _write_gz(dataset / "GSE133344_filtered_barcodes.tsv.gz", "c0\nc1\nc2\nc3\nc4\nc5\nc6\nc7\nc8\n")
    entries = [
        (1, 1, 10), (2, 1, 20),
        (1, 2, 11), (2, 2, 21),
        (1, 3, 12), (2, 3, 22),
        (1, 4, 13), (2, 4, 23),
        (1, 5, 14), (2, 5, 24),
        (1, 6, 15), (2, 6, 25),
        (1, 7, 16), (2, 7, 17),
        (1, 8, 18), (2, 8, 19),
        (1, 9, 20), (2, 9, 21),
    ]
    lines = [
        "%%MatrixMarket matrix coordinate integer general",
        "%",
        f"4018 9 {len(entries)}",
        *(f"{row} {column} {value}" for row, column, value in entries),
    ]
    _write_gz(dataset / "GSE133344_filtered_matrix.mtx.gz", "\n".join(lines) + "\n")
    header = "cell_barcode,guide_identity,read_count,gemgroup\n"
    rows = (
        "c0,AHR_NegCtrl0__AHR_NegCtrl0,1,1\n"
        "c1,AHR_FEV__AHR_FEV,1,1\n"
        "c2,AHR_KLF1__AHR_KLF1_1,1,1\n"
        "c3,BCL2L11_BAK1__BCL2L11_BAK1,1,1\n"
        "c4,C3orf72_FOXL2__C3orf72_FOXL2_2,1,1\n"
        "c5,CBL_UBASH3A__CBL_UBASH3A,1,1\n"
        "c6,AHR_NegCtrl1__AHR_NegCtrl1,1,1\n"
        "c7,AHR_NegCtrl2__AHR_NegCtrl2,1,1\n"
        "c8,AHR_NegCtrl3__AHR_NegCtrl3,1,1\n"
    )
    if missing_guide:
        header = "cell_barcode,read_count,gemgroup\n"
        rows = rows.replace("AHR_NegCtrl0__AHR_NegCtrl0,1", "1")
    _write_gz(dataset / "GSE133344_filtered_cell_identities.csv.gz", header + rows)
    return dataset


class _FakeAdapter:
    def __init__(self, gene_names: tuple[str, ...]):
        self.gene_names = gene_names
        self.model = SimpleNamespace(registry_={"setup_args": {}})

    def validate_compatibility(self, **kwargs):
        assert kwargs["expected_num_genes"] == 4018

    def encode(self, adata):
        # The aligned first two columns are 10*c + 20 and 10*c + 10,
        # making the matrix reordering observable in the latent pair.
        first = np.asarray(adata.X[:, 0].todense()).ravel()
        second = np.asarray(adata.X[:, 1].todense()).ravel()
        latent = np.zeros((adata.n_obs, 64), dtype=np.float32)
        latent[:, 0] = first
        latent[:, 1] = second
        return latent


def _asset():
    return SimpleNamespace(
        gene_to_token={
            "ENSG1": 17, "ENSG0": 23, "ENSG2": 29, "ENSG3": 31, "ENSG4": 37,
            "ENSG5": 41, "ENSG6": 43, "ENSG7": 45, "ENSG8": 47,
        },
        vocab_size=50,
        embedding_dim=8,
        manifest={"schema_version": 1},
    )


def test_parse_guide_identity_supports_controls_double_genes_and_copy_suffixes():
    assert builder.parse_guide_identity("ARID1A_NegCtrl0__ARID1A_NegCtrl0") == ("control", True)
    assert builder.parse_guide_identity("TGFBR2_IGDCC3__TGFBR2_IGDCC3_2") == ("TGFBR2_IGDCC3", False)
    assert builder.parse_guide_identity("NegCtrl0_UBASH3B__NegCtrl0_UBASH3B_1") == ("UBASH3B", False)
    with pytest.raises(builder.BuildDAVFLatentPairsError, match="sides disagree"):
        builder.parse_guide_identity("A_B__A_C")


def test_missing_guide_identity_column_fails(tmp_path):
    dataset = _make_norman(tmp_path, missing_guide=True)
    with pytest.raises(builder.BuildDAVFLatentPairsError, match="guide_identity"):
        builder.load_norman_10x(dataset)


def test_build_is_reproducible_and_uses_perturbgen_tokens(tmp_path):
    dataset = _make_norman(tmp_path)
    names = tuple(f"ENSG{i}" for i in range(4018))
    adapter = _FakeAdapter(names)
    asset = _asset()

    first = builder.build_latent_pairs(
        dataset, tmp_path / "scvi", tmp_path / "asset", tmp_path / "out_a", seed=7,
        adapter=adapter, asset=asset,
    )
    second = builder.build_latent_pairs(
        dataset, tmp_path / "scvi", tmp_path / "asset", tmp_path / "out_b", seed=7,
        adapter=adapter, asset=asset,
    )
    assert set(first) == {"train", "val", "test"}
    for split in first:
        with np.load(first[split]["path"], allow_pickle=False) as a, np.load(second[split]["path"], allow_pickle=False) as b:
            for key in ("z_0", "z_1", "gene_ids", "directions", "attention_mask", "metadata_json"):
                assert np.array_equal(a[key], b[key])
            assert a["gene_ids"].max() < asset.vocab_size
            assert set(a["directions"].ravel()) == {0}
            assert a["attention_mask"].shape[1] == 2
            metadata = json.loads(a["metadata_json"].item())
            assert metadata["schema_version"] == "ptm2cellnet.latent-davf-pairs.v1"
            assert metadata["scvi"]["num_genes"] == 4018
            assert metadata["embedding_asset"]["vocab_size"] == 50
            assert metadata["dataset"]["intervention_type"] == "KO"
            assert metadata["dataset"]["direction_code"] == 0

            from src.data.latent_davf_dataset import load_latent_davf_pairs

            loaded = load_latent_davf_pairs(
                first[split]["path"],
                latent_dim=64,
                perturbgen_vocab_size=50,
                expected_intervention_type="KO",
            )
            assert len(loaded) == first[split]["samples"]
    # These are asset token rows, not the scVI positions 0/1.
    all_gene_ids = np.concatenate([
        np.load(first[split]["path"], allow_pickle=False)["gene_ids"] for split in first
    ])
    assert {17, 23}.issubset(set(all_gene_ids.ravel()))


def test_split_is_by_label_and_deterministic():
    labels = ["A", "A", "B", "C", "D", "D"]
    split_a = builder.split_perturbation_labels(labels, seed=11)
    split_b = builder.split_perturbation_labels(labels, seed=11)
    assert split_a == split_b
    assert set().union(*(set(values) for values in split_a.values())) == {"A", "B", "C", "D"}
    assert not (set(split_a["train"]) & set(split_a["val"]))
    assert not (set(split_a["train"]) & set(split_a["test"]))
    assert not (set(split_a["val"]) & set(split_a["test"]))


def test_control_pools_are_disjoint_between_splits():
    data = builder.Norman10xData(
        matrix=sp.csr_matrix(np.ones((6, 1), dtype=np.float32)),
        raw_gene_ids=("ENSG1",),
        raw_gene_symbols=("G1",),
        barcodes=tuple(f"cell_{index}" for index in range(6)),
        labels=("control", "control", "control", "A", "B", "C"),
        pairing_groups=("gem", "gem", "gem", "gem", "gem", "gem"),
    )
    pools = builder._allocate_control_pools(
        data,
        {"train": ("A",), "val": ("B",), "test": ("C",)},
        np.random.default_rng(5),
    )
    train = set(pools["train"]["gem"].tolist())
    val = set(pools["val"]["gem"].tolist())
    test = set(pools["test"]["gem"].tolist())
    assert not (train & val or train & test or val & test)
    assert train | val | test == {0, 1, 2}


def test_alignment_rejects_incomplete_raw_gene_mapping(tmp_path):
    dataset = _make_norman(tmp_path)
    data = builder.load_norman_10x(dataset)
    with pytest.raises(builder.BuildDAVFLatentPairsError, match="do not completely cover"):
        builder.align_to_scvi_gene_order(data, ("MISSING",) + tuple(f"ENSG{i}" for i in range(4017)))


def test_alignment_sums_duplicate_raw_symbol_columns():
    data = builder.Norman10xData(
        matrix=sp.csr_matrix(np.arange(4019, dtype=np.float32).reshape(1, 4019)),
        raw_gene_ids=tuple(f"raw_{index}" for index in range(4019)),
        raw_gene_symbols=("DUP", "DUP") + tuple(f"G{index}" for index in range(4017)),
        barcodes=("cell",),
        labels=("control",),
        pairing_groups=("1",),
    )

    aligned = builder.align_to_scvi_gene_order(
        data,
        ("DUP",) + tuple(f"G{index}" for index in range(4017)),
    )

    assert aligned.shape == (1, 4018)
    assert aligned[0, 0] == 1.0


def test_missing_perturbgen_mapping_fails_before_encoding(tmp_path):
    dataset = _make_norman(tmp_path)
    names = tuple(f"ENSG{i}" for i in range(4018))
    adapter = _FakeAdapter(names)
    asset = _asset()
    asset.gene_to_token.pop("ENSG1")
    with pytest.raises(builder.BuildDAVFLatentPairsError, match="absent from the PerturbGen asset"):
        builder.build_latent_pairs(
            dataset, tmp_path / "scvi", tmp_path / "asset", tmp_path / "out", adapter=adapter, asset=asset,
        )
