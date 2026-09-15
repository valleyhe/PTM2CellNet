"""Unit tests for scripts/import_norman_adamson.py (synthetic 10x fixtures)."""

import gzip
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import scripts.import_norman_adamson as imp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def _make_gse133344(root: Path) -> Path:
    """6 cells x 5 genes fixture with two perturbation groups and NT control.

    Design (all genes baseline 1.0):
      b1,b2 -> NT control
      b3,b4 -> target GENE1 (gene0 = 3.0)
      b5,b6 -> target GENE2 (gene1 = 2.0)
    Expected delta vs control: GENE1 [2,0,0,0,0], GENE2 [0,1,0,0,0].
    """
    dataset = root / "GSE133344"
    dataset.mkdir(parents=True, exist_ok=True)

    _write_gz(
        dataset / "GSE133344_filtered_matrix.mtx.gz",
        # 10x 惯例：行=基因(5)，列=细胞(6)
        "%%MatrixMarket matrix coordinate real general\n"
        "%\n"
        "5 6 30\n"
        "1 1 1\n2 1 1\n3 1 1\n4 1 1\n5 1 1\n"
        "1 2 1\n2 2 1\n3 2 1\n4 2 1\n5 2 1\n"
        "1 3 3\n2 3 1\n3 3 1\n4 3 1\n5 3 1\n"
        "1 4 3\n2 4 1\n3 4 1\n4 4 1\n5 4 1\n"
        "1 5 1\n2 5 2\n3 5 1\n4 5 1\n5 5 1\n"
        "1 6 1\n2 6 2\n3 6 1\n4 6 1\n5 6 1\n",
    )
    _write_gz(
        dataset / "GSE133344_filtered_genes.tsv.gz",
        "ENSG00000000001\tGENE1\n"
        "ENSG00000000002\tGENE2\n"
        "ENSG00000000003\tGENE3\n"
        "ENSG00000000004\tGENE4\n"
        "ENSG00000000005\tGENE5\n",
    )
    _write_gz(
        dataset / "GSE133344_filtered_barcodes.tsv.gz",
        "b1\nb2\nb3\nb4\nb5\nb6\n",
    )
    _write_gz(
        dataset / "GSE133344_filtered_cell_identities.csv.gz",
        "barcode,target_gene,guide\n"
        "b1,NT,guide-nt1\n"
        "b2,NT,guide-nt2\n"
        "b3,GENE1,guide-a1\n"
        "b4,GENE1,guide-a2\n"
        "b5,GENE2,guide-b1\n"
        "b6,GENE2,guide-b2\n",
    )
    return dataset


class TestReaders:
    def test_read_mtx_returns_csc_with_expected_values(self, tmp_path):
        dataset = _make_gse133344(tmp_path)
        matrix = imp.read_mtx(dataset / "GSE133344_filtered_matrix.mtx.gz")
        assert sp.issparse(matrix)
        # 10x 惯例：行=基因、列=细胞
        assert matrix.shape == (5, 6)
        assert matrix[0, 2] == 3.0  # 基因1 × 细胞3
        assert matrix[1, 4] == 2.0  # 基因2 × 细胞5

    def test_read_genes_two_columns(self, tmp_path):
        dataset = _make_gse133344(tmp_path)
        ensembl, symbols = imp.read_genes(dataset / "GSE133344_filtered_genes.tsv.gz")
        assert ensembl[0] == "ENSG00000000001"
        assert symbols == ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

    def test_read_barcodes(self, tmp_path):
        dataset = _make_gse133344(tmp_path)
        assert imp.read_barcodes(dataset / "GSE133344_filtered_barcodes.tsv.gz") == [
            "b1",
            "b2",
            "b3",
            "b4",
            "b5",
            "b6",
        ]


class TestIdentityParsing:
    def test_detect_schema_aliases(self):
        rows = [["Barcode", "Target Gene", "Guide ID"], ["b1", "GENE1", "g1"]]
        schema = imp.detect_identity_schema(rows)
        assert schema["barcode"] == 0
        assert schema["target"] == 1
        assert schema["guide"] == 2

    def test_detect_schema_fails_without_annotation(self):
        with pytest.raises(imp.ImportError_, match="无法识别扰动注释列"):
            imp.detect_identity_schema([["foo", "bar"], ["x", "y"]])

    def test_parse_identities_and_perturbation_ids(self, tmp_path):
        dataset = _make_gse133344(tmp_path)
        rows = imp.read_tsv(dataset / "GSE133344_filtered_cell_identities.csv.gz")
        annotations = imp.parse_identities(rows)
        assert annotations["b3"] == {"target": "GENE1", "guide": "guide-a1", "perturbation": "", "is_control": ""}
        assert imp.perturbation_id_from_annotation(annotations["b1"]) == "control"
        assert imp.perturbation_id_from_annotation(annotations["b3"]) == "GENE1"

    def test_control_markers(self):
        for label in ("NT", "Non-targeting", "non_targeting", "Control", "scramble"):
            assert imp.is_control_label(label)
        assert not imp.is_control_label("GENE1")


class TestDeltaExpression:
    def test_delta_vs_control_hand_computed(self, tmp_path):
        dataset = _make_gse133344(tmp_path)
        matrix = imp.read_mtx(dataset / "GSE133344_filtered_matrix.mtx.gz").T.tocsc()
        barcodes = imp.read_barcodes(dataset / "GSE133344_filtered_barcodes.tsv.gz")
        rows = imp.read_tsv(dataset / "GSE133344_filtered_cell_identities.csv.gz")
        annotations = imp.parse_identities(rows)

        delta, sample_ids, counts, control_mean, unassigned = imp.compute_delta_expression(
            matrix, annotations, barcodes
        )
        assert set(sample_ids) == {"GENE1", "GENE2"}
        assert counts.tolist() == [2, 2]
        np.testing.assert_allclose(control_mean, [1.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(delta[sample_ids.index("GENE1")], [2.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(delta[sample_ids.index("GENE2")], [0.0, 1.0, 0.0, 0.0, 0.0])
        assert unassigned.sum() == 0

    def test_unassigned_cells_are_excluded(self):
        matrix = sp.csc_matrix(np.ones((2, 3), dtype=np.float32))
        delta, sample_ids, counts, _, unassigned = imp.compute_delta_expression(matrix, {}, ["x1", "x2"])
        assert delta.shape == (0, 3)
        assert unassigned.sum() == 2


class TestImportGSE133344EndToEnd:
    def test_full_import_writes_expected_artifacts(self, tmp_path):
        _make_gse133344(tmp_path)
        output = tmp_path / "out"
        summary = imp.import_gse133344(tmp_path, output)

        assert summary["cells"] == 6
        assert summary["genes"] == 5
        assert summary["n_perturbations"] == 2
        assert summary["controls_present"] is True
        assert summary["unassigned_cells"] == 0

        # expression.npz sparse roundtrip
        with np.load(output / "GSE133344_expression.npz", allow_pickle=False) as archive:
            shape = tuple(archive["shape"])
            rebuilt = sp.csc_matrix((archive["data"], (archive["row"], archive["col"])), shape=shape)
        assert rebuilt.shape == (6, 5)
        assert rebuilt[2, 0] == 3.0

        # perturbations.tsv
        table = (output / "GSE133344_perturbations.tsv").read_text(encoding="utf-8")
        assert "GENE1\t2" in table and "GENE2\t2" in table

        # delta_expression.npz
        with np.load(output / "GSE133344_delta_expression.npz", allow_pickle=True) as archive:
            delta = archive["delta"]
            ids = list(archive["sample_ids"])
            genes = list(archive["gene_symbols"])
        assert ids == ["GENE1", "GENE2"]
        assert genes == ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        np.testing.assert_allclose(delta[0], [2.0, 0.0, 0.0, 0.0, 0.0])

        # sequence_requests.json
        requests = json.loads((output / "sequence_requests.json").read_text(encoding="utf-8"))
        assert requests["n_genes"] == 2
        assert set(requests["genes"]) == {"GENE1", "GENE2"}

    def test_missing_dataset_fails_soft_in_main_flow(self, tmp_path):
        output = tmp_path / "out"
        rc = imp.main(["--geo-root", str(tmp_path / "empty"), "--output", str(output), "--skip-gse90546"])
        assert rc == 0
        assert (output / "import_manifest.json").is_file()
        manifest = json.loads((output / "import_manifest.json").read_text(encoding="utf-8"))
        assert manifest["gse133344_summary"]["cells"] is None
        assert manifest["gse90546"]["status"] == "skipped"


class TestGSE90546Probe:
    def test_probe_missing_tar(self, tmp_path):
        report = imp.probe_gse90546(tmp_path)
        assert report["status"] == "missing"

    def test_probe_lists_tar_members(self, tmp_path):
        dataset = tmp_path / "GSE90546"
        dataset.mkdir(parents=True)
        tar_path = dataset / "GSE90546_RAW.tar"
        with tarfile.open(tar_path, "w:gz") as archive:
            content = b"gene\tb1\tb2\nGENE1\t1\t2\n"
            info = tarfile.TarInfo("counts.tsv")
            info.size = len(content)
            archive.addfile(info, __import__("io").BytesIO(content))
        report = imp.probe_gse90546(tmp_path)
        assert report["status"] == "probed"
        names = [member["name"] for member in report["members"]]
        assert "counts.tsv" in names
        assert report["members"][0]["looks_tabular"] is True


class TestGSE90546FullParse:
    def _make_tar(self, root: Path) -> Path:
        dataset = root / "GSE90546"
        dataset.mkdir(parents=True, exist_ok=True)
        tar_path = dataset / "GSE90546_RAW.tar"
        prefix = "GSM0001_10X001"
        members = {
            f"{prefix}_genes.tsv.gz": "ENSG1\tGENE1\nENSG2\tGENE2\nENSG3\tGENE3\n",
            f"{prefix}_barcodes.tsv.gz": "b1\nb2\nb3\nb4\n",
            f"{prefix}_cell_identities.csv.gz": (
                "cell,target_gene,guide\nb1,NT,g1\nb2,NT,g2\nb3,GENE1,g3\nb4,GENE1,g4\n"
            ),
            f"{prefix}_matrix.mtx.txt.gz": (
                "%%MatrixMarket matrix coordinate real general\n%\n"
                "3 4 12\n"
                "1 1 1\n2 1 1\n3 1 1\n"
                "1 2 1\n2 2 1\n3 2 1\n"
                "1 3 3\n2 3 1\n3 3 1\n"
                "1 4 3\n2 4 1\n3 4 1\n"
            ),
        }
        with tarfile.open(tar_path, "w") as archive:
            for name, text in members.items():
                compressed = io.BytesIO()
                with gzip.GzipFile(fileobj=compressed, mode="wb") as gz:
                    gz.write(text.encode("utf-8"))
                payload = compressed.getvalue()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return tar_path

    def test_parse_writes_per_experiment_artifacts(self, tmp_path) -> None:
        self._make_tar(tmp_path)
        output = tmp_path / "out"
        report = imp.parse_gse90546(tmp_path, output)

        assert report["status"] == "parsed"
        assert report["n_experiments"] == 1
        experiment = report["experiments"][0]
        assert experiment["cells"] == 4
        assert experiment["genes"] == 3
        assert (output / "GSE90546_GSM0001_10X001_expression.npz").is_file()
        assert (output / "GSE90546_GSM0001_10X001_delta_expression.npz").is_file()
        table = (output / "GSE90546_perturbations.tsv").read_text(encoding="utf-8")
        assert "GSM0001_10X001\tGENE1\t2" in table

        with np.load(output / "GSE90546_GSM0001_10X001_expression.npz", allow_pickle=False) as archive:
            assert tuple(archive["shape"]) == (4, 3)
            rebuilt = sp.csc_matrix((archive["data"], (archive["row"], archive["col"])), shape=tuple(archive["shape"]))
        assert rebuilt[2, 0] == 3
        with np.load(output / "GSE90546_GSM0001_10X001_delta_expression.npz", allow_pickle=False) as archive:
            assert list(archive["sample_ids"]) == ["GENE1"]
            np.testing.assert_allclose(archive["delta"][0], [2.0, 0.0, 0.0])

    def test_import_switches_to_parsed_status_only_with_explicit_flag(self, tmp_path) -> None:
        self._make_tar(tmp_path)
        output = tmp_path / "out"
        report = imp.import_gse90546(tmp_path, output, parse=True)
        assert report["status"] == "parsed"
        saved = json.loads((output / "GSE90546_structure_report.json").read_text(encoding="utf-8"))
        assert saved["status"] == "parsed"


class TestCombinatorialPerturbationRequests:
    def test_combinatorial_labels_extract_matching_genes(self, tmp_path):
        """双基因扰动标签 GENE1_GENE2 应提取出两个基因（TD-H02 输入契约）。"""
        from scripts.import_norman_adamson import _requested_genes_from_labels

        genes = ["GENE1", "GENE2", "GENE3"]
        labels = ["GENE1", "GENE1_GENE2", "control", "GENE3_GENE_X"]
        assert _requested_genes_from_labels(labels, genes) == ["GENE1", "GENE2", "GENE3"]


class TestNormanGuideIdentity:
    """Norman 2019 guide_identity 列格式解析（真实数据验证中发现）。"""

    def test_single_gene_guide(self):
        target, is_control = imp._parse_norman_guide_identity("ARID1A_NegCtrl0__ARID1A_NegCtrl0")
        assert target == "ARID1A"
        assert is_control is True

    def test_combinatorial_guide(self):
        target, is_control = imp._parse_norman_guide_identity("SET_KLF1__SET_KLF1")
        assert target == "SET_KLF1"
        assert is_control is False

    def test_parse_identities_with_guide_identity_column(self, tmp_path):
        dataset = tmp_path / "GSE133344"
        dataset.mkdir(parents=True)
        import gzip as _gz

        path = dataset / "identities.csv.gz"
        with _gz.open(path, "wt", encoding="utf-8") as handle:
            handle.write(
                "cell_barcode,guide_identity,read_count,UMI_count\n"
                "b1,ARID1A_NegCtrl0__ARID1A_NegCtrl0,100,10\n"
                "b2,SET_KLF1__SET_KLF1,200,20\n"
            )
        rows = imp.read_tsv(path)
        annotations = imp.parse_identities(rows)
        assert annotations["b1"]["target"] == "ARID1A"
        assert annotations["b1"]["is_control"] == "1"
        assert imp.perturbation_id_from_annotation(annotations["b1"]) == "control"
        assert annotations["b2"]["target"] == "SET_KLF1"
        assert imp.perturbation_id_from_annotation(annotations["b2"]) == "SET_KLF1"


class TestMatrixOrientation:
    def test_gene_x_cell_matrix_is_transposed(self, tmp_path):
        """10x 惯例矩阵为 基因×细胞，导入时须转置为 细胞×基因。"""
        import gzip as _gz

        dataset = tmp_path / "GSE133344"
        dataset.mkdir(parents=True)
        with _gz.open(dataset / "GSE133344_filtered_matrix.mtx.gz", "wt", encoding="utf-8") as handle:
            handle.write(
                "%%MatrixMarket matrix coordinate real general\n%\n5 6 6\n1 1 3\n1 2 3\n1 3 2\n1 4 2\n2 5 5\n3 6 7\n"
            )
        with _gz.open(dataset / "GSE133344_filtered_genes.tsv.gz", "wt", encoding="utf-8") as handle:
            handle.write("ENSG1\tG1\nENSG2\tG2\nENSG3\tG3\nENSG4\tG4\nENSG5\tG5\n")
        with _gz.open(dataset / "GSE133344_filtered_barcodes.tsv.gz", "wt", encoding="utf-8") as handle:
            handle.write("b1\nb2\nb3\nb4\nb5\nb6\n")
        with _gz.open(dataset / "GSE133344_filtered_cell_identities.csv.gz", "wt", encoding="utf-8") as handle:
            handle.write("barcode,target_gene\nb1,NT\nb2,NT\nb3,NT\nb4,NT\nb5,NT\nb6,NT\n")

        output = tmp_path / "out"
        summary = imp.import_gse133344(tmp_path, output)
        assert summary["cells"] == 6
        assert summary["genes"] == 5
        with np.load(output / "GSE133344_expression.npz", allow_pickle=False) as archive:
            rebuilt = sp.csc_matrix(
                (archive["data"], (archive["row"], archive["col"])),
                shape=tuple(archive["shape"]),
            )
        assert rebuilt.shape == (6, 5)
        assert rebuilt[0, 0] == 3.0  # 细胞 b1 的基因 G1（原 mtx (1,1,3)）
        assert rebuilt[2, 0] == 2.0  # 细胞 b3 的基因 G1（原 mtx (1,3,2)）
