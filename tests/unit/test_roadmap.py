"""Unit tests for src/models/roadmap.py (V2-01 .. V2-05 helpers)."""

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.models.roadmap import (
    DEFERRED_FEATURES,
    DeferredFeature,
    distributed_trainer,
    esm3_encoder,
    get_deferred_features,
    load_custom_ptm_database,
    launch_gui,
    mass_spec_stream,
)


class TestDeferredRegistry:
    def test_registry_contains_all_v2_features(self):
        ids = {f.requirement_id for f in DEFERRED_FEATURES}
        assert ids == {"V2-01", "V2-02", "V2-03", "V2-04", "V2-05"}

    def test_get_deferred_features_returns_copy(self):
        first = get_deferred_features()
        first.append("polluted")
        second = get_deferred_features()
        assert "polluted" not in second

    def test_each_feature_has_required_fields(self):
        for f in DEFERRED_FEATURES:
            assert isinstance(f, DeferredFeature)
            assert f.requirement_id.startswith("V2-")
            assert f.name and f.summary and f.rationale and f.recommended_path


class TestRoadmapHelpers:
    def test_esm3_encoder_falls_back_to_esm2(self, monkeypatch, caplog):
        class FakeESM2Encoder:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        import src.models.pretrained_encoders as pretrained_encoders

        monkeypatch.setattr(pretrained_encoders, "ESM2Encoder", FakeESM2Encoder)
        monkeypatch.delattr(pretrained_encoders, "ESM3Encoder", raising=False)

        with caplog.at_level("WARNING"):
            encoder = esm3_encoder(model_size="650M", freeze=True)

        assert isinstance(encoder, FakeESM2Encoder)
        assert encoder.kwargs["model_size"] == "650M"
        assert encoder.kwargs["freeze"] is True
        assert "Falling back to ESM2Encoder" in caplog.text

    def test_mass_spec_stream_parses_tsv_stream(self):
        stream = StringIO("position\tptm_type\tintensity\tconfidence\n12\tphospho\t0.8\t0.99\n")

        result = mass_spec_stream(stream)

        assert list(result.columns) == ["position", "ptm_type", "intensity", "confidence"]
        assert result.to_dict(orient="records") == [
            {"position": 12, "ptm_type": "phospho", "intensity": 0.8, "confidence": 0.99}
        ]

    def test_load_custom_ptm_database_standardizes_and_warns(self, tmp_path, caplog):
        csv_path = tmp_path / "custom_ptm.csv"
        pd.DataFrame(
            [
                {
                    "accession": "P12345",
                    "position": 17,
                    "ptm_type": "acetylation",
                    "amino_acid": None,
                    "confidence": 0.87,
                }
            ]
        ).to_csv(csv_path, index=False)

        with caplog.at_level("WARNING"):
            result = load_custom_ptm_database(csv_path, source="user-upload")

        assert list(result.columns) == [
            "protein_accession",
            "position",
            "ptm_type",
            "amino_acid",
            "source",
            "confidence",
        ]
        assert result.loc[0, "protein_accession"] == "P12345"
        assert result.loc[0, "source"] == "user-upload"
        assert "missing values" in caplog.text.lower()

    def test_load_custom_ptm_database_rejects_missing_required_columns(self):
        with pytest.raises(ValueError, match="Missing required columns"):
            load_custom_ptm_database(pd.DataFrame([{"protein_accession": "P1"}]))

    def test_distributed_trainer_falls_back_without_lightning(self, monkeypatch, caplog):
        class FakeTrainer:
            def __init__(self, model, config=None, device=None):
                self.model = model
                self.config = config
                self.device = device

        import src.models.roadmap as roadmap

        monkeypatch.setattr(roadmap, "_LIGHTNING_IMPORT_ERROR", RuntimeError("missing lightning"))
        monkeypatch.setattr(roadmap, "_LIGHTNING_MODULE", None)
        monkeypatch.setattr(roadmap, "Trainer", FakeTrainer)

        model = MagicMock()
        datamodule = MagicMock()

        with caplog.at_level("WARNING"):
            trainer = distributed_trainer(model, datamodule, config={"training": {}}, device="cpu")

        assert isinstance(trainer, FakeTrainer)
        assert trainer.model is model
        assert "falling back to plain Trainer" in caplog.text

    def test_launch_gui_returns_false_without_streamlit(self, monkeypatch, capsys):
        import src.models.roadmap as roadmap

        monkeypatch.setattr(roadmap, "_STREAMLIT_IMPORT_ERROR", RuntimeError("missing streamlit"))
        monkeypatch.setattr(roadmap, "_STREAMLIT_MODULE", None)

        launched = launch_gui(title="PTM2CellNet")

        captured = capsys.readouterr()
        assert launched is False
        assert "streamlit is not installed" in captured.out.lower()

    def test_launch_gui_runs_streamlit_when_available(self, monkeypatch, tmp_path):
        import src.models.roadmap as roadmap

        run_calls = []

        def fake_run(cmd, check):
            run_calls.append((cmd, check))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(roadmap, "_STREAMLIT_IMPORT_ERROR", None)
        monkeypatch.setattr(roadmap, "_STREAMLIT_MODULE", object())
        monkeypatch.setattr(roadmap, "subprocess", SimpleNamespace(run=fake_run))
        monkeypatch.setattr(roadmap.tempfile, "gettempdir", lambda: str(tmp_path))

        launched = launch_gui(title="PTM2CellNet GUI", port=8765)

        assert launched is True
        assert run_calls
        command, check = run_calls[0]
        assert command[:3] == ["streamlit", "run", str(Path(tmp_path) / "ptm2cellnet_streamlit_app.py")]
        assert "--server.port" in command
        assert check is False
