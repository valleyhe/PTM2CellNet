"""Unit tests for scripts/integrate_data_v2.py provenance columns (P2-1).

Verifies the real-data integration pipeline now emits the recommended
provenance columns (protein_accession / gene_symbol / source_db /
evidence_level) so the data contract's leakage audit and release gate
are actually usable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# Load the script as a module (it is not part of the src package).
_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "integrate_data_v2.py"
_spec = importlib.util.spec_from_file_location("integrate_data_v2", _SCRIPT_PATH)
integrate_data_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(integrate_data_v2)

RAW_DIR = integrate_data_v2.RAW_DIR


def _sample_sequences():
    return {"P12345": "ACDEFGHIKLMNPQRSTVWY" * 2, "Q00001": "ACDEFGHIKLMNPQRSTVWY"}


def _sample_sites():
    return {
        "P12345": [
            {"position": 1, "type": "phosphorylation", "amino_acid": "A"},
            {"position": 5, "type": "acetylation", "amino_acid": "F"},
        ],
        "Q00001": [{"position": 2, "type": "methylation", "amino_acid": "C"}],
    }


class TestBuildDatasetProvenance:
    def test_emits_all_recommended_columns(self):
        df = integrate_data_v2.build_dataset(_sample_sequences(), _sample_sites())
        for col in (
            "protein_accession",
            "gene_symbol",
            "source_db",
            "evidence_level",
        ):
            assert col in df.columns, f"missing recommended column {col}"
        assert len(df) == 2

    def test_protein_accession_equals_uniprot_id(self):
        df = integrate_data_v2.build_dataset(_sample_sequences(), _sample_sites())
        assert (df["protein_accession"] == df["uniprot_id"]).all()

    def test_gene_symbol_mapped_from_idmapping(self):
        gene_map = {"P12345": "TP53"}
        df = integrate_data_v2.build_dataset(_sample_sequences(), _sample_sites(), gene_symbol_map=gene_map)
        by_id = df.set_index("uniprot_id")["gene_symbol"]
        assert by_id["P12345"] == "TP53"
        # 无映射的 accession 留空而非报错
        assert by_id["Q00001"] == ""

    def test_source_db_and_evidence_level_filled(self):
        df = integrate_data_v2.build_dataset(_sample_sequences(), _sample_sites())
        assert (df["source_db"] == "epsd+cplm+dbptm").all()
        assert (df["evidence_level"] == "database").all()

    def test_isoform_resolution_keeps_canonical_accession(self):
        sequences = {"P12345": "ACDEFGHIKLMNPQRSTVWY" * 2}
        sites = {"P12345-3": [{"position": 1, "type": "phosphorylation", "amino_acid": "A"}]}
        df = integrate_data_v2.build_dataset(sequences, sites)
        assert len(df) == 1
        assert df.iloc[0]["protein_accession"] == "P12345"


class TestGeneSymbolMapLoader:
    @pytest.mark.skipif(
        not (Path(RAW_DIR) / "uniprot" / "human_idmapping.gz").exists(),
        reason="local UniProt idmapping file not present",
    )
    def test_loads_gene_names_from_local_idmapping(self):
        gene_map = integrate_data_v2._load_gene_symbol_map()
        assert isinstance(gene_map, dict)
        assert len(gene_map) > 0
        # UniProt 官方 idmapping 的 Gene_Name 行格式
        sample = next(iter(gene_map.items()))
        assert isinstance(sample[0], str) and isinstance(sample[1], str)

    def test_returns_empty_when_idmapping_missing(self, monkeypatch):
        monkeypatch.setattr(integrate_data_v2, "RAW_DIR", str(Path("/nonexistent")))
        assert integrate_data_v2._load_gene_symbol_map() == {}


class TestIntegrateLabelsProvenance:
    def test_pmads_columns_are_propagated(self, tmp_path, monkeypatch):
        """PMADS 分支：protein→protein_accession、source→source_db。"""
        pmads = pd.DataFrame(
            {
                "id": ["p1"],
                "sequence": ["ACDEFGHIK"],
                "ptm_sites": ['[{"position":1,"type":"phosphorylation"}]'],
                "cell_state": ["apoptosis"],
                "protein": ["P12345"],
                "ptm_position": [1],
                "ptm_type": ["phosphorylation"],
                "source": ["pmads_db"],
            }
        )
        monkeypatch.setattr(integrate_data_v2, "PROCESSED_DIR", str(tmp_path))
        pmads.to_csv(tmp_path / "pmads_combined.csv", index=False)

        out = integrate_data_v2.integrate_labels(pd.DataFrame(), gene_symbol_map={"P12345": "TP53"})
        assert out.iloc[0]["protein_accession"] == "P12345"
        assert out.iloc[0]["gene_symbol"] == "TP53"
        assert out.iloc[0]["source_db"] == "pmads_db"
        assert out.iloc[0]["evidence_level"] == "database"

    def test_returns_input_when_pmads_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(integrate_data_v2, "PROCESSED_DIR", str(tmp_path))
        df = pd.DataFrame({"a": [1]})
        assert integrate_data_v2.integrate_labels(df) is df
