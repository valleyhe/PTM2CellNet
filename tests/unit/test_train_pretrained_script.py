"""
train_pretrained.py script behavior tests
"""

import types

import pandas as pd

from src.utils.config import Config
import scripts.train_pretrained as train_pretrained


def test_main_creates_model_before_datamodule_with_tokenizer(monkeypatch, tmp_path):
    config = Config(
        {
            "model": {
                "encoder_type": "esm2_150M",
                "freeze_encoder": False,
            },
            "data": {
                "num_workers": 0,
                "pin_memory": False,
            },
            "training": {
                "max_epochs": 1,
                "devices": 0,
                "precision": "32",
                "batch_size": 1,
                "drop_last": False,
                "checkpoint": {"enabled": False},
                "early_stopping": {"enabled": False},
                "logger": {
                    "save_dir": str(tmp_path / "logs"),
                    "name": "test",
                },
            },
            "paths": {
                "outputs_models": str(tmp_path / "models"),
            },
        }
    )

    args = types.SimpleNamespace(
        model="esm2_150M",
        config=None,
        data="dummy.csv",
        freeze=False,
        resume=None,
        max_epochs=None,
        batch_size=None,
        learning_rate=None,
        gpus=0,
        precision="32",
        seed=42,
    )

    monkeypatch.setattr(train_pretrained, "parse_args", lambda: args)
    monkeypatch.setattr(train_pretrained.Config, "from_yaml", lambda _: config)

    class DummyLoader:
        def load_from_csv(self, _):
            # P0-2: label derivation requires >=2 classes, so provide two.
            return pd.DataFrame(
                {
                    "sequence": ["AAAA", "CCCC"],
                    "ptm_sites": [[], []],
                    "cell_state": ["state_a", "state_b"],
                }
            )

    class DummyPreprocessor:
        def __init__(self, _):
            pass

        def preprocess_pipeline(self, df):
            return df, df, df

    monkeypatch.setattr(train_pretrained, "DataLoader", DummyLoader)
    monkeypatch.setattr(train_pretrained, "DataPreprocessor", DummyPreprocessor)

    calls = []
    seen = {}
    tokenizer_obj = object()

    class DummyEncoder:
        def __init__(self):
            self.tokenizer = tokenizer_obj

    class DummyParam:
        def __init__(self, numel, requires_grad):
            self._numel = numel
            self.requires_grad = requires_grad

        def numel(self):
            return self._numel

    class DummyModel:
        def __init__(self):
            self.encoder = DummyEncoder()

        def parameters(self):
            return [DummyParam(10, True), DummyParam(5, False)]

    def fake_from_config(_):
        calls.append("model")
        return DummyModel()

    monkeypatch.setattr(
        train_pretrained.PTM2CellNet,
        "from_config",
        staticmethod(fake_from_config),
    )

    def fake_lightning(_, __):
        calls.append("lightning")
        return object()

    monkeypatch.setattr(train_pretrained, "PTM2CellNetLightning", fake_lightning)

    def fake_datamodule(*, train_df, val_df, test_df, config, tokenizer=None, feature_extractor=None):
        calls.append("datamodule")
        seen["tokenizer"] = tokenizer
        seen["train_df"] = train_df
        seen["val_df"] = val_df
        seen["test_df"] = test_df
        seen["config"] = config
        return types.SimpleNamespace()

    monkeypatch.setattr(train_pretrained, "PTMLightningDataModule", fake_datamodule)

    class DummyTrainer:
        def __init__(self, *_, **__):
            pass

        def fit(self, *_, **__):
            return None

        def test(self, *_, **__):
            return [{"metric": 1.0}]

    class DummyTBLogger:
        def __init__(self, *_, **__):
            pass

    monkeypatch.setattr(train_pretrained.L, "Trainer", DummyTrainer)
    # lightning does not expose loggers as a top-level attribute;
    # inject a stub module so L.loggers.TensorBoardLogger resolves.
    import types as _types
    _stub_loggers = _types.ModuleType("lightning.loggers")
    _stub_loggers.TensorBoardLogger = DummyTBLogger
    monkeypatch.setattr(train_pretrained.L, "loggers", _stub_loggers, raising=False)

    # P0-2: the unified artifact export now runs at the end of training. This
    # unit test is scoped to the model-vs-datamodule ordering and tokenizer
    # injection only, so stub the export/manifest helpers (the full export path
    # is covered by tests/e2e/test_pretrained_train_predict_cli.py).
    monkeypatch.setattr(
        train_pretrained,
        "export_inference_artifact",
        lambda *a, **k: {"checkpoint_path": "x", "config_path": "y"},
    )
    monkeypatch.setattr(
        train_pretrained,
        "write_artifact_manifest",
        lambda *a, **k: {},
    )

    train_pretrained.main()

    assert calls.index("model") < calls.index("datamodule")
    assert seen["tokenizer"] is tokenizer_obj
