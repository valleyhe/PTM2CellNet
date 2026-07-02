"""Project roadmap helpers for deferred v1.0 requirements (V2-01 .. V2-05)."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import pandas as pd
import torch

from ..training.trainers import Trainer

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import lightning as _LIGHTNING_MODULE
    _LIGHTNING_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    _LIGHTNING_MODULE = None
    _LIGHTNING_IMPORT_ERROR = exc

try:  # pragma: no cover - optional dependency
    import streamlit as _STREAMLIT_MODULE  # noqa: F401
    _STREAMLIT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    _STREAMLIT_MODULE = None
    _STREAMLIT_IMPORT_ERROR = exc


@dataclass(frozen=True)
class DeferredFeature:
    """Metadata for a deferred roadmap feature."""

    requirement_id: str
    name: str
    summary: str
    rationale: str
    recommended_path: str


DEFERRED_FEATURES: List[DeferredFeature] = [
    DeferredFeature(
        requirement_id="V2-01",
        name="ESM-3 integration",
        summary="Integrate ESM-3 (Evolutionary Scale Model 3) protein language model.",
        rationale=(
            "Deferred at v1.0 pending stable public release and weights "
            "availability of ESM-3. The existing ESM-2 encoder "
            "(``src/models/encoders.py`` / ``src/models/pretrained_encoders.py``) "
            "covers current protein-LM needs."
        ),
        recommended_path=(
            "Add an ``ESM3Encoder`` subclass of ``SequenceEncoder`` once ESM-3 "
            "weights are stable; mirror the ESM-2 integration pattern."
        ),
    ),
    DeferredFeature(
        requirement_id="V2-02",
        name="Real-time mass-spec streaming",
        summary="Streaming ingestion / online prediction for mass-spectrometry data.",
        rationale=(
            "Deferred at v1.0; no streaming/mass-spec infrastructure is needed "
            "for the current batch-prediction use case served by the API "
            "(``src/api/routes/predictions.py``)."
        ),
        recommended_path=(
            "Introduce a ``src/data/streaming.py`` module (Kafka/queue consumer "
            "+ incremental PTM-site decoder) when real-time pipelines are required."
        ),
    ),
    DeferredFeature(
        requirement_id="V2-03",
        name="Custom PTM database support",
        summary="Load and serve user-supplied custom PTM databases.",
        rationale=(
            "Deferred at v1.0. PTM type knowledge is currently driven by the "
            "extensible registry in ``src/models/ptm_direction_mapper.py`` "
            "(see ``register_ptm_direction``), which covers the runtime "
            "extensibility need without a dedicated DB loader."
        ),
        recommended_path=(
            "Add a ``src/data/custom_ptm_db.py`` loader when on-disk custom "
            "PTM databases (beyond the registry) are required."
        ),
    ),
    DeferredFeature(
        requirement_id="V2-04",
        name="Distributed training support",
        summary="Multi-GPU / distributed (DDP) training.",
        rationale=(
            "Deferred at v1.0. Lightning (already a dependency) provides DDP "
            "via ``L.Trainer(devices=N, strategy='ddp')``; no dedicated module "
            "is required until distributed training is exercised in practice."
        ),
        recommended_path=(
            "Pass ``strategy='ddp'`` and ``devices>1`` to the Lightning "
            "Trainer in training scripts (e.g. ``scripts/finetune_davf.py``); "
            "optionally add a helper in ``src/training/`` to centralize it."
        ),
    ),
    DeferredFeature(
        requirement_id="V2-05",
        name="GUI interface",
        summary="Graphical user interface for model interaction.",
        rationale=(
            "Deferred at v1.0. The FastAPI service (``src/api/``) is the "
            "primary interaction surface; a GUI was not a v1.0 milestone."
        ),
        recommended_path=(
            "Build a thin web front-end over the existing REST API rather "
            "than a desktop GUI (e.g. Streamlit/Gradio), when needed."
        ),
    ),
]

_MASS_SPEC_COLUMNS = ["position", "ptm_type", "intensity", "confidence"]
_STANDARD_PTM_COLUMNS = [
    "protein_accession",
    "position",
    "ptm_type",
    "amino_acid",
    "source",
    "confidence",
]
_COLUMN_ALIASES = {
    "accession": "protein_accession",
    "protein_id": "protein_accession",
    "uniprot": "protein_accession",
    "uniprot_accession": "protein_accession",
    "site": "position",
    "ptm_position": "position",
    "modification": "ptm_type",
    "residue": "amino_acid",
    "residue_aa": "amino_acid",
    "database": "source",
    "origin": "source",
    "score": "confidence",
    "probability": "confidence",
}


def _ensure_dataframe(records: Any, *, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, (str, Path)):
        return pd.read_csv(records, sep="\t")
    if hasattr(records, "read"):
        return pd.read_csv(records, sep="\t")

    rows = list(records)
    if not rows:
        return pd.DataFrame(columns=columns)
    if isinstance(rows[0], Mapping):
        return pd.DataFrame(rows)
    return pd.DataFrame.from_records(rows, columns=columns)


def esm3_encoder(*args: Any, **kwargs: Any):
    """V2-01: return an ESM-3 encoder when available, else fall back to ESM-2."""
    from . import pretrained_encoders

    esm3_cls = getattr(pretrained_encoders, "ESM3Encoder", None)
    if esm3_cls is not None:
        return esm3_cls(*args, **kwargs)

    esm2_cls = getattr(pretrained_encoders, "ESM2Encoder")
    logger.warning("ESM-3 encoder is unavailable. Falling back to ESM2Encoder.")
    return esm2_cls(*args, **kwargs)


def mass_spec_stream(file_or_stream: Any, **kwargs: Any) -> pd.DataFrame:
    """V2-02: parse tab-delimited mass-spec PTM records into a DataFrame."""
    frame = _ensure_dataframe(file_or_stream, columns=_MASS_SPEC_COLUMNS)
    missing = [column for column in _MASS_SPEC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required mass-spec columns: {missing}")

    frame = frame.loc[:, _MASS_SPEC_COLUMNS].copy()
    frame["position"] = pd.to_numeric(frame["position"], errors="raise").astype(int)
    frame["intensity"] = pd.to_numeric(frame["intensity"], errors="coerce")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    if kwargs.get("dropna", False):
        frame = frame.dropna(subset=_MASS_SPEC_COLUMNS)
    return frame.reset_index(drop=True)


def load_custom_ptm_database(path_or_df: Any, **kwargs: Any) -> pd.DataFrame:
    """V2-03: load a user-supplied PTM table into the standard schema."""
    if isinstance(path_or_df, pd.DataFrame):
        frame = path_or_df.copy()
    else:
        path = Path(path_or_df)
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv", ".txt"}:
            separator = kwargs.get("sep", "\t" if suffix == ".tsv" else ",")
            frame = pd.read_csv(path, sep=separator)
        elif suffix in {".json", ".jsonl"}:
            frame = pd.read_json(path)
        elif suffix in {".xls", ".xlsx"}:
            frame = pd.read_excel(path, sheet_name=kwargs.get("sheet_name", 0))
        else:
            raise ValueError(f"Unsupported PTM database format: {suffix or '<no suffix>'}")

    frame = frame.rename(columns=_COLUMN_ALIASES)
    if "source" not in frame.columns:
        frame["source"] = kwargs.get("source", "custom")
    if "confidence" not in frame.columns:
        frame["confidence"] = kwargs.get("confidence", pd.NA)

    required_input_columns = {"protein_accession", "position", "ptm_type", "amino_acid"}
    missing = sorted(required_input_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")

    standardized = frame.reindex(columns=_STANDARD_PTM_COLUMNS).copy()
    missing_values = standardized.isna().sum()
    missing_columns = [name for name, count in missing_values.items() if int(count) > 0]
    if missing_columns:
        logger.warning(
            "Custom PTM database contains missing values in columns: %s",
            ", ".join(missing_columns),
        )

    standardized["position"] = pd.to_numeric(standardized["position"], errors="raise").astype(int)
    standardized["confidence"] = pd.to_numeric(standardized["confidence"], errors="coerce")
    return standardized.reset_index(drop=True)


def distributed_trainer(model: Any, datamodule: Any, **kwargs: Any):
    """V2-04: build a Lightning DDP trainer when possible, else fall back safely."""
    del datamodule

    if _LIGHTNING_MODULE is not None:
        trainer_kwargs = dict(kwargs)
        devices = int(trainer_kwargs.pop("devices", 1))
        if torch.cuda.is_available():
            available_devices = torch.cuda.device_count()
            if available_devices > 1:
                trainer_kwargs["accelerator"] = trainer_kwargs.get("accelerator", "gpu")
                trainer_kwargs["devices"] = max(devices, available_devices)
                trainer_kwargs["strategy"] = "ddp"
            elif devices > 1:
                logger.warning(
                    "Requested devices=%s but only %s GPU is available; using a normal Lightning Trainer.",
                    devices,
                    available_devices,
                )
                trainer_kwargs["devices"] = 1
        return _LIGHTNING_MODULE.Trainer(**trainer_kwargs)

    logger.warning(
        "lightning is unavailable (%s); falling back to plain Trainer.",
        _LIGHTNING_IMPORT_ERROR,
    )
    return Trainer(
        model,
        config=kwargs.get("config"),
        device=kwargs.get("device"),
    )


def launch_gui(**kwargs: Any):
    """V2-05: launch a minimal Streamlit UI if available."""
    title = kwargs.get("title", "PTM2CellNet")
    port = int(kwargs.get("port", 8501))
    app_path = Path(tempfile.gettempdir()) / "ptm2cellnet_streamlit_app.py"

    if _STREAMLIT_MODULE is None:
        print(
            "Streamlit is not installed. Install it with `pip install streamlit` "
            f"and run `streamlit run {app_path}` to launch the GUI."
        )
        logger.warning("Streamlit GUI launch skipped: %s", _STREAMLIT_IMPORT_ERROR)
        return False

    app_path.write_text(
        "\n".join(
            [
                "import streamlit as st",
                f"st.set_page_config(page_title={title!r}, layout='wide')",
                f"st.title({title!r})",
                "st.write('Minimal PTM2CellNet GUI placeholder.')",
                "sequence = st.text_area('Protein sequence')",
                "ptm_data = st.text_area('PTM records (tab-delimited)')",
                "if st.button('Predict'):",
                "    st.success('GUI scaffold launched. Connect this view to the API for predictions.')",
                "    st.write({'sequence': sequence, 'ptm_data': ptm_data})",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["streamlit", "run", str(app_path), "--server.port", str(port)],
        check=False,
    )
    return result.returncode == 0


def get_deferred_features() -> List[DeferredFeature]:
    """Return metadata for all deferred v1.0 roadmap features."""
    return list(DEFERRED_FEATURES)


__all__ = [
    "DeferredFeature",
    "DEFERRED_FEATURES",
    "get_deferred_features",
    "esm3_encoder",
    "mass_spec_stream",
    "load_custom_ptm_database",
    "distributed_trainer",
    "launch_gui",
]
