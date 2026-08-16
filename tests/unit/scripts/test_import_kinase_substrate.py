"""Unit tests for scripts/import_kinase_substrate.py (synthetic fixtures)."""

import json
from pathlib import Path
from unittest import mock

import pytest

import scripts.import_kinase_substrate as imp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_enzsub(path: Path, rows: str, header: str = "enzyme\tsubstrate\tresidue_type\tresidue_offset\tmodification\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + rows, encoding="utf-8")
    return path


ENZSUB_ROWS = (
    "P06239\tO14543\tY\t204\tphosphorylation\n"
    "P12931\tP14746\tS\t707\tphosphorylation\n"
    "P06241\tO15117\tY\t651\tacetylation\n"
)


@pytest.fixture()
def enzsub_file(tmp_path: Path) -> Path:
    return _write_enzsub(tmp_path / "omnipath_enzsub.tsv", ENZSUB_ROWS)


@pytest.fixture()
def mapping_file(tmp_path: Path) -> Path:
    p = tmp_path / "uniprot_to_gene.tsv"
    p.write_text("# uniprot_id\tgene_symbol\nP06239\tLCK\nO14543\tPTPRJ\nP12931\tSRC\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseEnzsub:
    def test_parses_expected_records(self, enzsub_file: Path):
        records = imp.parse_enzsub(enzsub_file)
        assert len(records) == 3
        assert records[0] == {
            "enzyme": "P06239",
            "substrate": "O14543",
            "residue_type": "Y",
            "residue_offset": "204",
            "modification": "phosphorylation",
        }

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(imp.ImportError_, match="缺少输入文件"):
            imp.parse_enzsub(tmp_path / "nope.tsv")

    def test_wrong_header_raises(self, tmp_path: Path):
        p = _write_enzsub(tmp_path / "bad.tsv", "a\tb\n", header="x\ty\tz\n")
        with pytest.raises(imp.ImportError_, match="列头"):
            imp.parse_enzsub(p)

    def test_short_row_raises(self, tmp_path: Path):
        p = _write_enzsub(tmp_path / "short.tsv", "P06239\tO14543\tY\n")
        with pytest.raises(imp.ImportError_, match="列数不足"):
            imp.parse_enzsub(p)

    def test_empty_data_raises(self, tmp_path: Path):
        p = _write_enzsub(tmp_path / "empty.tsv", "")
        with pytest.raises(imp.ImportError_, match="无数据行"):
            imp.parse_enzsub(p)

    def test_max_rows_limits(self, enzsub_file: Path):
        records = imp.parse_enzsub(enzsub_file, max_rows=2)
        assert len(records) == 2

    def test_blank_lines_skipped(self, tmp_path: Path):
        p = _write_enzsub(tmp_path / "blank.tsv", "\n\nP06239\tO14543\tY\t204\tphosphorylation\n\n")
        assert len(imp.parse_enzsub(p)) == 1


class TestCollectUniprotIds:
    def test_unique_ids_across_columns(self, enzsub_file: Path):
        records = imp.parse_enzsub(enzsub_file)
        ids = imp.collect_uniprot_ids(records)
        assert ids == ["O14543", "O15117", "P06239", "P06241", "P12931", "P14746"]

    def test_empty_records(self):
        assert imp.collect_uniprot_ids([]) == []


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


class TestLoadMappingFile:
    def test_loads_two_column_tsv(self, mapping_file: Path):
        mapping = imp.load_mapping_file(mapping_file)
        assert mapping["P06239"] == "LCK"
        assert mapping["P12931"] == "SRC"

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert imp.load_mapping_file(tmp_path / "nope.tsv") == {}

    def test_malformed_row_raises(self, tmp_path: Path):
        p = tmp_path / "bad.tsv"
        p.write_text("P06239\n", encoding="utf-8")
        with pytest.raises(imp.ImportError_, match="行格式错误"):
            imp.load_mapping_file(p)

    def test_comments_skipped(self, tmp_path: Path):
        p = tmp_path / "c.tsv"
        p.write_text("# header\n\nP06239\tLCK\n", encoding="utf-8")
        assert imp.load_mapping_file(p) == {"P06239": "LCK"}


class TestWriteMappingFile:
    def test_roundtrip(self, tmp_path: Path):
        p = tmp_path / "cache.tsv"
        imp.write_mapping_file(p, {"P06239": "LCK", "P12931": "SRC"})
        assert imp.load_mapping_file(p) == {"P06239": "LCK", "P12931": "SRC"}


class TestResolveOnline:
    def _mock_session(self, mapping: dict, *, inline: bool = True):
        """Fake requests session: submit -> status FINISHED -> full results.

        ``inline=True`` mimics the real /status payload shape (results key
        present, truncated in reality), but the full result set must still
        be fetched from /results — mirroring the production code path.
        """
        session = mock.MagicMock()
        session.post.return_value = mock.MagicMock(
            status_code=200,
            text="job-1",
            json=lambda: {"jobId": "job-1"},
            raise_for_status=lambda: None,
        )
        results = [
            {"from": uid, "to": {"primaryAccession": gene}}
            for uid, gene in mapping.items()
        ]
        status_payload = (
            {"results": results, "failedIds": []}
            if inline
            else {"jobStatus": "FINISHED"}
        )
        session.get = mock.MagicMock(
            side_effect=[
                mock.MagicMock(status_code=200, json=lambda: status_payload, raise_for_status=lambda: None),
                mock.MagicMock(
                    status_code=200,
                    json=lambda: {"results": results},
                    raise_for_status=lambda: None,
                ),
            ]
        )
        return session

    def test_resolves_batch(self):
        session = self._mock_session({"P06239": "LCK", "O14543": "PTPRJ"})
        mapping = imp.resolve_online(["P06239", "O14543"], session=session)
        assert mapping == {"P06239": "LCK", "O14543": "PTPRJ"}
        # Status polled via /status/{job}; the full result set is ALWAYS
        # fetched from /results (inline /status results are truncated).
        assert session.get.call_args_list[0][0][0].endswith("/status/job-1")
        assert session.get.call_args_list[1][0][0].endswith("/results/job-1")

    def test_resolves_via_jobstatus_flavour(self):
        session = self._mock_session({"P06239": "LCK"}, inline=False)
        mapping = imp.resolve_online(["P06239"], session=session)
        assert mapping == {"P06239": "LCK"}
        assert session.get.call_args_list[1][0][0].endswith("/results/job-1")

    def test_empty_input(self, session=None):
        assert imp.resolve_online([], session=session) == {}

    def test_submission_failure_raises(self):
        session = mock.MagicMock()
        session.post.side_effect = RuntimeError("boom")
        with pytest.raises(imp.ImportError_, match="提交失败"):
            imp.resolve_online(["P06239"], session=session)

    def test_job_timeout_raises(self):
        session = mock.MagicMock()
        session.post.return_value = mock.MagicMock(
            status_code=200,
            text="job-1",
            json=lambda: {"jobId": "job-1"},
            raise_for_status=lambda: None,
        )
        # Status never reaches FINISHED -> poll budget exhausted.
        session.get.return_value = mock.MagicMock(
            status_code=200, json=lambda: {"jobStatus": "RUNNING"}, raise_for_status=lambda: None
        )
        with mock.patch.object(imp, "ONLINE_POLL_BUDGET_SECONDS", 0.1), mock.patch.object(
            imp, "ONLINE_POLL_INTERVAL_SECONDS", 0.0
        ), pytest.raises(imp.ImportError_, match="未在预算时间内完成"):
            imp.resolve_online(["P06239"], session=session)


class TestResolveGenes:
    def test_local_cache_only(self, tmp_path: Path, mapping_file: Path):
        resolved = imp.resolve_genes(["P06239", "P12931", "Q00000"], mapping_file=mapping_file)
        assert resolved == {"P06239": "LCK", "O14543": "PTPRJ", "P12931": "SRC"}

    def test_online_fills_missing(self, mapping_file: Path):
        session = mock.MagicMock()
        session.post.return_value = mock.MagicMock(
            status_code=200,
            text="job-1",
            json=lambda: {"jobId": "job-1"},
            raise_for_status=lambda: None,
        )
        session.get = mock.MagicMock(
            side_effect=[
                mock.MagicMock(status_code=200, json=lambda: {"jobStatus": "FINISHED"}, raise_for_status=lambda: None),
                mock.MagicMock(
                    status_code=200,
                    json=lambda: {"results": [{"from": "Q00000", "to": {"primaryAccession": "MYGENE"}}]},
                    raise_for_status=lambda: None,
                ),
            ]
        )
        resolved = imp.resolve_genes(["P06239", "Q00000"], mapping_file=mapping_file, online=True, session=session)
        assert resolved == {"P06239": "LCK", "O14543": "PTPRJ", "P12931": "SRC", "Q00000": "MYGENE"}

    def test_online_without_local_file(self):
        session = mock.MagicMock()
        session.post.return_value = mock.MagicMock(
            status_code=200,
            text="job-1",
            json=lambda: {"jobId": "job-1"},
            raise_for_status=lambda: None,
        )
        session.get = mock.MagicMock(
            side_effect=[
                mock.MagicMock(status_code=200, json=lambda: {"jobStatus": "FINISHED"}, raise_for_status=lambda: None),
                mock.MagicMock(
                    status_code=200,
                    json=lambda: {"results": [{"from": "P06239", "to": {"primaryAccession": "LCK"}}]},
                    raise_for_status=lambda: None,
                ),
            ]
        )
        resolved = imp.resolve_genes(["P06239"], online=True, session=session)
        assert resolved == {"P06239": "LCK"}

    def test_no_source_returns_empty(self):
        assert imp.resolve_genes(["P06239"]) == {}


# ---------------------------------------------------------------------------
# Edge table
# ---------------------------------------------------------------------------


class TestBuildEdgeTable:
    def test_maps_and_keeps_unmapped(self, enzsub_file: Path, mapping_file: Path):
        records = imp.parse_enzsub(enzsub_file)
        mapping = imp.load_mapping_file(mapping_file)
        edges = imp.build_edge_table(records, mapping)
        assert len(edges) == 3
        # Mapped pair.
        assert edges[0]["source_gene"] == "LCK"
        assert edges[0]["target_gene"] == "PTPRJ"
        assert edges[0]["edge_type"] == "kinase_substrate:phosphorylation"
        assert edges[0]["source_site"] == "Y204"
        assert edges[0]["species"] == "9606"
        assert edges[0]["release"] == "2026-08"
        # Unmapped substrate kept verbatim (accession), not dropped.
        assert edges[1]["source_gene"] == "SRC"
        assert edges[1]["target_gene"] == "P14746"
        # Edge type carries the modification vocabulary.
        assert edges[2]["edge_type"] == "kinase_substrate:acetylation"

    def test_empty_records(self):
        assert imp.build_edge_table([], {}) == []


class TestWriteEdgeTable:
    def test_roundtrip_columns(self, tmp_path: Path, enzsub_file: Path, mapping_file: Path):
        records = imp.parse_enzsub(enzsub_file)
        edges = imp.build_edge_table(records, imp.load_mapping_file(mapping_file))
        out = tmp_path / "edges.tsv"
        imp.write_edge_table(out, edges)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "\t".join(imp.EDGE_TABLE_COLUMNS)
        assert len(lines) == 4
        assert lines[1].startswith("LCK\tPTPRJ\tkinase_substrate:phosphorylation\tY204\t")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestWriteImportManifest:
    def test_manifest_content(self, tmp_path: Path, enzsub_file: Path):
        class Args:
            mapping_file = None
            online_mapping = False

        out = tmp_path / "out"
        path = imp.write_import_manifest(
            out,
            source_path=enzsub_file,
            source_sha256="abc123",
            n_records=3,
            n_unique_ids=6,
            n_mapped_ids=4,
            n_edges=3,
            edge_type_counts={"kinase_substrate:phosphorylation": 2, "kinase_substrate:acetylation": 1},
            args=Args(),
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "ptm2cellnet.kinase-substrate.import.v1"
        assert manifest["summary"]["n_records"] == 3
        assert manifest["summary"]["mapping_rate"] == round(4 / 6, 4)
        assert manifest["summary"]["edge_type_counts"]["kinase_substrate:phosphorylation"] == 2
        assert manifest["parameters"]["online_mapping"] is False


# ---------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------


class TestMain:
    def test_cli_offline_with_mapping_file(self, tmp_path: Path, enzsub_file: Path, mapping_file: Path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir()
        enzsub_file.rename(raw / "omnipath_enzsub.tsv")
        out = tmp_path / "processed"
        rc = imp.main(["--raw-root", str(raw), "--output", str(out), "--mapping-file", str(mapping_file)])
        assert rc == 0
        captured = json.loads(capsys.readouterr().out)
        assert captured["ok"] is True
        assert captured["n_records"] == 3
        assert captured["n_mapped_uniprot_ids"] == 3  # P06239, O14543, P12931
        assert (out / imp.EDGES_NAME).is_file()
        assert (out / imp.MANIFEST_NAME).is_file()
        assert (out / imp.MAPPING_CACHE_NAME).is_file()
        # Manifest provenance.
        manifest = json.loads((out / imp.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["source"]["sha256"]
        assert manifest["summary"]["n_edges"] == 3

    def test_cli_reuses_previous_cache(self, tmp_path: Path, enzsub_file: Path, mapping_file: Path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir()
        enzsub_file.rename(raw / "omnipath_enzsub.tsv")
        out = tmp_path / "processed"
        imp.main(["--raw-root", str(raw), "--output", str(out), "--mapping-file", str(mapping_file)])
        capsys.readouterr()  # drain first run's output
        # Second run without --mapping-file must reuse the written cache.
        rc = imp.main(["--raw-root", str(raw), "--output", str(out)])
        assert rc == 0
        captured = json.loads(capsys.readouterr().out)
        assert captured["n_mapped_uniprot_ids"] == 3

    def test_cli_missing_input_fails(self, tmp_path: Path, capsys):
        rc = imp.main(["--raw-root", str(tmp_path / "nope"), "--output", str(tmp_path / "out")])
        assert rc != 0
        assert "缺少输入文件" in capsys.readouterr().err
