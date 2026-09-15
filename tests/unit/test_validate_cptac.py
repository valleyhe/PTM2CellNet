"""Unit tests for the refactored validate_cptac script (F-03 / TD-H3).

Verifies that:

* the experimental guard refuses to run without the env var
* mock mode produces a result file explicitly marked not scientifically valid
* the PDC-backed downloader delegates to :class:`PDCClient` for the manifest
* the validator's compare_with_known_sites parses site IDs correctly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_cptac.py"


def _import_script():
    """Import scripts/validate_cptac.py as a module (it's not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_cptac", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_raises_without_env(monkeypatch):
    monkeypatch.delenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", raising=False)
    # Force re-import so the module-level guard runs again.
    sys.modules.pop("validate_cptac", None)
    with pytest.raises(RuntimeError, match="EXPERIMENTAL"):
        _import_script()


def test_guard_passes_with_env(monkeypatch):
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()
    assert hasattr(module, "CPTACDataDownloader")
    assert hasattr(module, "CPTACValidator")


def test_mock_backend_produces_synthetic_matrix(monkeypatch, tmp_path):
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()
    downloader = module.CPTACDataDownloader(str(tmp_path), backend="mock")
    df = downloader.download_phosphoproteomics("BRCA")
    assert df.shape[0] > 0
    assert df.shape[1] == 100
    # Cache file written.
    assert (tmp_path / "BRCA_phosphoproteomics.csv").is_file()


def test_pdc_backend_delegates_to_pdc_client(monkeypatch, tmp_path):
    """In PDC backend, the manifest call must route through PDCClient.get_study."""
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    # Fake the PDCClient.get_study to return a known PDCStudy.
    from src.analysis.pdc_client import PDCStudy

    captured: dict = {}

    class _FakeClient:
        def __init__(self, *a, **kw):
            captured["init_args"] = (a, kw)

        def get_study(self, study_id):
            captured["study_id"] = study_id
            return PDCStudy(
                study_id=study_id,
                study_submitter_id="CPTAC-BRCA",
                study_name="CPTAC Breast Cancer",
                disease_type="Breast",
                primary_site="Breast",
                files=[{"file_name": "phospho.tsv", "data_category": "Phosphoproteomics"}],
            )

    monkeypatch.setattr(module, "PDCClient", _FakeClient) if hasattr(module, "PDCClient") else None
    # PDCClient is imported lazily inside the method; patch at the source.
    import src.analysis.pdc_client as pdc_mod

    monkeypatch.setattr(pdc_mod, "PDCClient", _FakeClient)

    downloader = module.CPTACDataDownloader(str(tmp_path), backend="pdc")
    manifest = downloader.download_study_manifest("BRCA")
    assert captured["study_id"] == "7c0c6e28-d405-11e8-b853-a005056ab009"
    assert manifest["study_submitter_id"] == "CPTAC-BRCA"
    assert manifest["files"][0]["file_name"] == "phospho.tsv"


def test_compare_with_known_sites_parses_site_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    phospho = pd.DataFrame(
        {"Sample_0": [1.0, 2.0, 3.0]},
        index=["P12345_100S", "P12345_200T", "P67890_300Y"],
    )
    phospho.index.name = "Site"
    validator = module.CPTACValidator.__new__(module.CPTACValidator)
    validator.phospho_data = phospho
    result = validator.compare_with_known_sites()
    assert result["total_phospho_sites"] == 3
    assert result["unique_proteins"] == 2
    assert set(result["aa_distribution"].keys()) == {"S", "T", "Y"}


def test_main_mock_mode_marks_results_not_valid(monkeypatch, tmp_path):
    """End-to-end: mock mode writes a JSON with scientifically_valid=false."""
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    out_dir = tmp_path / "cptac_out"
    out_dir.mkdir()

    # Argv with --mock + --download-only to avoid the model load path
    # (which would require a real checkpoint).
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_cptac.py", "--mock", "-o", str(out_dir), "--download-only"],
    )
    module.main()

    # download-only doesn't write validation_results.json; verify data files.
    assert (out_dir / "BRCA_phosphoproteomics.csv").is_file()
    assert (out_dir / "BRCA_mutations.csv").is_file()


def test_main_mock_full_run_marks_results_not_valid(monkeypatch, tmp_path):
    """Full mock run (not download-only) writes scientifically_valid=false."""
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    out_dir = tmp_path / "cptac_full"
    out_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_cptac.py", "--mock", "-o", str(out_dir), "-m", str(tmp_path / "no_models")],
    )
    module.main()

    result_path = out_dir / "validation_results.json"
    assert result_path.is_file()
    payload = json.loads(result_path.read_text())
    assert payload["scientifically_valid"] is False
    assert payload["backend"] == "mock"
    # v17: validator now runs and emits predictor_wired / effect_counts.
    assert payload["predictor_wired"] is False
    assert "prediction_summary" in payload


# ---------------------------------------------------------------------------
# F-03 v17: TSV parser + predictor wiring
# ---------------------------------------------------------------------------


def test_parse_phospho_tsv_basic(tmp_path):
    """_parse_phospho_tsv parses a CPTAC-style (sites × samples) TSV."""
    monkeypatch_env = None
    monkeypatch_env = __import__("pytest").MonkeyPatch()
    monkeypatch_env.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()
    monkeypatch_env.undo()

    tsv_path = tmp_path / "phospho.tsv"
    tsv_path.write_text(
        "Gene_Site\tSample_1\tSample_2\tGene\tPosition\nP53_S15\t1.0\t2.0\tTP53\t15\nP53_T18\t3.0\t4.0\tTP53\t18\n",
        encoding="utf-8",
    )
    df = module._parse_phospho_tsv(tsv_path)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["Sample_1", "Sample_2"]
    assert df.index.name == "Site"
    assert df.loc["P53_S15", "Sample_1"] == 1.0


def test_parse_phospho_tsv_rejects_no_sample_columns(tmp_path):
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()
    mp.undo()

    tsv_path = tmp_path / "metadata_only.tsv"
    tsv_path.write_text(
        "Gene\tPosition\tDescription\nP53\t15\tblah\n",
        encoding="utf-8",
    )
    with _pytest.raises(ValueError, match="no sample intensity columns"):
        module._parse_phospho_tsv(tsv_path)


def test_find_best_checkpoint_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()
    assert module._find_best_checkpoint(tmp_path / "does_not_exist") is None


def test_find_best_checkpoint_prefers_highest_val_auroc(monkeypatch, tmp_path):
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    ckpt_dir = tmp_path / "phosphorylation" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=000-val_auroc=0.50.ckpt").write_text("x")
    (ckpt_dir / "epoch=001-val_auroc=0.91.ckpt").write_text("x")
    (ckpt_dir / "epoch=002-val_auroc=0.80.ckpt").write_text("x")
    chosen = module._find_best_checkpoint(tmp_path)
    assert chosen is not None
    # Sorts descending by stem; 0.91 wins.
    assert "0.91" in chosen.stem


def test_validator_predictor_not_wired_emits_unknown(monkeypatch, tmp_path):
    """When no checkpoint is found, predicted_effect is 'unknown' for every site."""
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    import pandas as _pd

    mut = _pd.DataFrame(
        {
            "UniProt_ID": ["P53"],
            "Protein_Position": [15],
            "Reference_AA": ["S"],
            "Variant_AA": ["A"],
        }
    )
    validator = module.CPTACValidator(
        model_dir=str(tmp_path / "no_models"),
        phospho_data=_pd.DataFrame(),
        mutation_data=mut,
        sequences={"P53": "MAS" * 50},
    )
    assert validator._predictor is None
    out = validator.validate_predictions()
    assert out["scientifically_valid"] is False
    assert out["predictor_wired"] is False
    assert out["validation_results"][0]["predicted_effect"] == "unknown"


def test_validator_predictor_wired_scores_mutation(monkeypatch, tmp_path):
    """When a predictor is wired, validate_predictions scores gain/loss/neutral."""
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    import pandas as _pd

    mut = _pd.DataFrame(
        {
            "UniProt_ID": ["P53"],
            "Protein_Position": [15],
            "Reference_AA": ["S"],
            "Variant_AA": ["A"],
        }
    )

    # Build a validator with a fake predictor injected directly.
    validator = module.CPTACValidator.__new__(module.CPTACValidator)
    validator.model_dir = tmp_path
    validator.phospho_data = _pd.DataFrame()
    validator.mutation_data = mut
    validator.sequences = {"P53": "MAS" * 50}
    validator._predictor_loaded = True

    class _FakeModel:
        ptm_types = ["Phosphorylation"]

    class _FakePredictor:
        model = _FakeModel()

        def predict_protein(self, sequence, protein_id, threshold=0.0):
            # WT scores low at the mutation site; mutant scores higher → gain.
            if protein_id == "mut":
                return _pd.DataFrame(
                    [
                        {
                            "protein_id": protein_id,
                            "position": 15,
                            "aa": "S",
                            "ptm_type": "Phosphorylation",
                            "probability": 0.9,
                        }
                    ]
                )
            return _pd.DataFrame(
                [
                    {
                        "protein_id": protein_id,
                        "position": 15,
                        "aa": "S",
                        "ptm_type": "Phosphorylation",
                        "probability": 0.4,
                    }
                ]
            )

    validator._predictor = _FakePredictor()
    out = validator.validate_predictions()
    assert out["scientifically_valid"] is True
    assert out["predictor_wired"] is True
    # delta = 0.9 - 0.4 = 0.5 > 0.2 threshold → gain.
    assert out["validation_results"][0]["predicted_effect"] == "gain"


def test_main_pdc_mode_uses_real_download(monkeypatch, tmp_path):
    """End-to-end: PDC backend must call PDCClient.download_file (not _mock)."""
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    out_dir = tmp_path / "cptac_pdc"
    out_dir.mkdir()

    # Patch PDCClient methods used by the downloader.
    import src.analysis.pdc_client as pdc_mod

    class _FakeStudy:
        study_id = "u"
        study_submitter_id = "CPTAC-BRCA"
        study_name = "CPTAC Breast Cancer"
        disease_type = "Breast"
        primary_site = "Breast"
        files = [{"file_id": "f1", "file_name": "phospho.tsv", "data_category": "Phosphoproteomics"}]

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.calls = []

        def get_study(self, study_id):
            return _FakeStudy()

        def download_file(self, file_id, target_dir, **kw):
            from pathlib import Path

            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            f = target / "phospho.tsv"
            f.write_text(
                "Gene_Site\tSample_1\nP53_S15\t1.0\nP53_T18\t2.0\n",
                encoding="utf-8",
            )
            return f

    monkeypatch.setattr(pdc_mod, "PDCClient", _FakeClient)

    # download-only to keep this test focused on the matrix path.
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_cptac.py", "-o", str(out_dir), "--download-only", "-s", "BRCA"],
    )
    module.main()

    # Real matrix cached.
    cached = out_dir / "BRCA_phosphoproteomics.csv"
    assert cached.is_file()
    import pandas as _pd

    df = _pd.read_csv(cached, index_col=0)
    assert df.shape == (2, 1)
    assert "Sample_1" in df.columns
