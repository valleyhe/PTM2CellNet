"""F-03 图数据导入器（scripts/import_ppi_graphs.py）单元测试。

覆盖 BioPlex / RegNetwork / STRING 三源的解析、契约校验、canonical 边表
输出与 manifest 登记；全部使用 tmp_path 合成小 fixture，不依赖在盘数据。
"""

import gzip
import json
from pathlib import Path

import pytest

import scripts.import_ppi_graphs as imp


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bioplex_raw(tmp_path):
    path = tmp_path / "bioplex_293T_v2_edges.tsv"
    path.write_text(
        "source\ttarget\n"
        "ADA\tPOTEF\n"
        "POTEF\tHSPBP1\n"
        "TP53\tMDM2\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def regnetwork_raw(tmp_path):
    source = tmp_path / "human.source"
    source.write_text(
        "USF1\t7391\tS100A6\t6277\n"
        "USF1\t7391\tDUSP1\t1843\n"
        "-\t7157\tTP73\t7161\n",  # TF 符号缺失 -> human.node 回填
        encoding="utf-8",
    )
    node = tmp_path / "human.node"
    node.write_text("7391\tUSF1\n6277\tS100A6\n1843\tDUSP1\n7157\tTP53\n7161\tTP73\n", encoding="utf-8")
    core = tmp_path / "human.core"
    core.write_text("7391\t6277\t+\n7391\t1843\t-\n", encoding="utf-8")
    return source, node, core


@pytest.fixture()
def string_raw(tmp_path):
    path = tmp_path / "9606.protein.physical.links.v12.0.txt.gz"
    lines = [
        "protein1 protein2 combined_score",
        "9606.ENSP00000000233 9606.ENSP00000356607 950",
        "9606.ENSP00000000233 9606.ENSP00000427567 400",  # 低于阈值
        "9606.ENSP00000356607 9606.ENSP00000427567 999",
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    mapping = tmp_path / "protein_info.tsv"
    mapping.write_text(
        "# string_protein_id\tgene_symbol\n"
        "9606.ENSP00000000233\tLCK\n"
        "9606.ENSP00000356607\tPTPRJ\n",
        encoding="utf-8",
    )
    return path, mapping


# ---------------------------------------------------------------------------
# BioPlex
# ---------------------------------------------------------------------------


class TestBioplex:
    def test_parse_and_cli_roundtrip(self, bioplex_raw, tmp_path, capsys):
        exit_code = imp.main(
            ["--source", "bioplex", "--raw-root", str(bioplex_raw.parent),
             "--output-root", str(tmp_path / "out")]
        )
        assert exit_code == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["n_edges"] == 3

        edges = (tmp_path / "out" / "bioplex" / "bioplex_edges.tsv").read_text(encoding="utf-8")
        rows = [line.split("\t") for line in edges.strip().split("\n")]
        assert rows[0] == list(imp.EDGE_TABLE_COLUMNS)
        assert rows[1][:2] == ["ADA", "POTEF"]
        assert rows[1][2] == "ppi:bioplex_apms"

        manifest = json.loads(
            (tmp_path / "out" / "bioplex" / "import_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["schema_version"] == "ptm2cellnet.ppi-graph.import.v1"
        assert manifest["edge_type_vocab"] == ["ppi:bioplex_apms"]
        assert manifest["summary"]["edge_type_counts"] == {"ppi:bioplex_apms": 3}
        assert manifest["source"]["files"][0]["sha256"]

    def test_missing_file_fails_fast(self, tmp_path):
        with pytest.raises(imp.ImportError_, match="缺少 BioPlex"):
            imp.parse_bioplex(tmp_path / "nope.tsv")

    def test_bad_header_rejected(self, tmp_path):
        path = tmp_path / "bad.tsv"
        path.write_text("a\tb\nX\tY\n", encoding="utf-8")
        with pytest.raises(imp.ImportError_, match="列头"):
            imp.parse_bioplex(path)


# ---------------------------------------------------------------------------
# RegNetwork
# ---------------------------------------------------------------------------


class TestRegnetwork:
    def test_parse_with_core_relations(self, regnetwork_raw):
        source, node, core = regnetwork_raw
        edges, note, n_resolved = imp.parse_regnetwork(source, node, core)
        assert len(edges) == 3
        assert n_resolved == 2
        assert note.startswith("human.core")
        by_pair = {(e["source_gene"], e["target_gene"]): e["edge_type"] for e in edges}
        assert by_pair[("USF1", "S100A6")] == "tf_regulation:activates"
        assert by_pair[("USF1", "DUSP1")] == "tf_regulation:inhibits"
        # TF 符号缺失行经 human.node 回填为 TP53
        assert by_pair[("TP53", "TP73")] == "tf_regulation:unspecified"

    def test_parse_without_core_marks_unspecified(self, regnetwork_raw):
        source, node, _ = regnetwork_raw
        edges, note, n_resolved = imp.parse_regnetwork(source, node, None)
        assert n_resolved == 0
        assert "unavailable" in note
        assert all(e["edge_type"] == "tf_regulation:unspecified" for e in edges)

    def test_cli_roundtrip(self, regnetwork_raw, tmp_path, capsys):
        source, node, core = regnetwork_raw
        exit_code = imp.main(
            ["--source", "regnetwork", "--raw-root", str(source.parent),
             "--output-root", str(tmp_path / "out"),
             "--regnetwork-core", str(core)]
        )
        assert exit_code == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["n_edges"] == 3
        manifest = json.loads(
            (tmp_path / "out" / "regnetwork" / "import_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["relation_vocab"] == {"+": "activates", "-": "inhibits", "?": "unspecified"}
        assert manifest["summary"]["n_relation_resolved"] == 2

    def test_bad_row_rejected(self, tmp_path):
        source = tmp_path / "human.source"
        source.write_text("ONLY\tTWO\n", encoding="utf-8")
        node = tmp_path / "human.node"
        node.write_text("1\tA\n", encoding="utf-8")
        with pytest.raises(imp.ImportError_, match="行列数不足"):
            imp.parse_regnetwork(source, node)


# ---------------------------------------------------------------------------
# STRING
# ---------------------------------------------------------------------------


class TestString:
    def test_parse_filters_by_score_and_maps(self, string_raw):
        path, mapping_path = string_raw
        mapping = imp.load_ensp_mapping(mapping_path)
        edges, n_below, unmapped = imp.parse_string(path, mapping, min_score=700)
        assert len(edges) == 2  # score=400 行被过滤
        assert n_below == 1
        # ENSP00000427567 未映射，原样保留（不丢弃）
        assert unmapped == {"9606.ENSP00000427567"}
        assert edges[0]["source_gene"] == "LCK"
        assert edges[0]["target_gene"] == "PTPRJ"
        assert edges[1]["target_gene"] == "9606.ENSP00000427567"
        assert edges[0]["score"] == "950"

    def test_mapping_accepts_bare_ensp_keys(self, tmp_path):
        mapping_file = tmp_path / "m.tsv"
        mapping_file.write_text("ENSP00000000233\tLCK\n", encoding="utf-8")
        mapping = imp.load_ensp_mapping(mapping_file)
        assert mapping["ENSP00000000233"] == "LCK"

    def test_mapping_missing_file_fails_fast(self, tmp_path):
        with pytest.raises(imp.ImportError_, match="不存在"):
            imp.load_ensp_mapping(tmp_path / "nope.tsv")

    def test_cli_roundtrip(self, string_raw, tmp_path, capsys):
        path, mapping_path = string_raw
        exit_code = imp.main(
            ["--source", "string", "--raw-root", str(path.parent),
             "--output-root", str(tmp_path / "out"),
             "--ensp-mapping", str(mapping_path)]
        )
        assert exit_code == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["n_edges"] == 2
        assert summary["n_below_threshold"] == 1
        assert 0.0 < summary["mapping_rate"] < 1.0
        manifest = json.loads(
            (tmp_path / "out" / "string" / "import_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["parameters"]["min_score"] == 700

    def test_all_filtered_fails_fast(self, string_raw):
        path, _ = string_raw
        with pytest.raises(imp.ImportError_, match="过滤后无数据行"):
            imp.parse_string(path, {}, min_score=10000)


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------


def test_invalid_source_rejected():
    with pytest.raises(SystemExit):
        imp.main(["--source", "unknown"])


def test_edge_table_columns_cover_datasets_yaml_contract():
    # 三源 canonical_schema required 列（source_gene/target_gene/score/relation）
    # 的并集由统一 7 列覆盖（relation 编入 edge_type 词汇表）
    columns = set(imp.EDGE_TABLE_COLUMNS)
    assert {"source_gene", "target_gene", "score", "evidence", "species", "release"} <= columns
    for vocab in imp.EDGE_TYPE_VOCAB.values():
        assert vocab
