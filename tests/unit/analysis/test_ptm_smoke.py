"""Unit tests for deterministic PTM smoke tables."""

from __future__ import annotations

from src.analysis.ptm_activity import PTM_INPUT_REQUIRED_COLUMNS, load_ptm_activity_table, load_ptm_site_quantification
from src.analysis.ptm_smoke import (
    KSTAR_METHOD_NAME,
    SMOKE_METHOD_VERSION,
    UNMAPPED_PROTEIN,
    build_activity_stub,
    build_kstar_evidence,
    build_site_quantification,
    write_smoke_bundle,
)


class TestSmokeTables:
    def test_site_table_covers_donors_replicate_and_unmapped(self):
        frame = build_site_quantification()
        assert list(frame.columns)[: len(PTM_INPUT_REQUIRED_COLUMNS)] == list(PTM_INPUT_REQUIRED_COLUMNS)
        assert set(frame["condition"]) == {"disease", "normal"}
        assert frame.loc[frame["donor_id"] != "", "donor_id"].nunique() == 6
        gsk = frame[(frame["gene_symbol"] == "GSK3B") & (frame["residue"] == "S9") & (frame["donor_id"] == "D1")]
        assert len(gsk) == 2
        assert UNMAPPED_PROTEIN in set(frame["protein_id"])
        assert (frame["donor_id"] == "").any()

    def test_kstar_evidence_labels_increased_and_decreased_without_running_kstar(self):
        evidence = build_kstar_evidence(build_site_quantification())
        by_site = evidence.set_index("site_id")
        assert int(by_site.loc["P49841_S9", "kstar_increased"]) == 1
        assert int(by_site.loc["Q00535_Y15", "kstar_decreased"]) == 1
        assert set(evidence["lineage_boundary"]) == {"smoke_only"}

    def test_activity_stub_keeps_kstar_method_name_but_smoke_version(self, tmp_path):
        stub = build_activity_stub()
        path = tmp_path / "activity.tsv"
        stub.to_csv(path, sep="\t", index=False)
        loaded = load_ptm_activity_table(path)
        assert set(loaded["method"]) == {KSTAR_METHOD_NAME}
        assert set(loaded["method_version"]) == {SMOKE_METHOD_VERSION}
        assert loaded["may_enter_lineage"].astype(str).str.lower().eq("false").all()


class TestSmokeBundle:
    def test_write_bundle_is_loadable(self, tmp_path):
        bundle = write_smoke_bundle(tmp_path / "smoke")
        assert bundle.lineage["biology_pass"] is False
        assert bundle.lineage["kstar_ran"] is False
        loaded = load_ptm_site_quantification(bundle.paths["ptm_site_quantification"])
        assert len(loaded) == len(build_site_quantification())
