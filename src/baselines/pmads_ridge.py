"""Leakage-safe PMADS Ridge regression baseline.

The baseline deliberately uses transparent, deterministic features instead of
the deep model stack: amino-acid composition, sequence length, PTM count/type
and normalized site-position summaries, plus explicitly selected numeric
measurements.  It is a reference point for model gains, not a replacement for
the scientific cross-scale model.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    matthews_corrcoef,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data.data_contract import validate_data_contract
from ..data.data_manifest import DataManifestError, load_manifest, manifest_digest
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
_EXCLUDED_NUMERIC_COLUMNS = {
    "label",
    "target",
    "cell_state",
    "split_group",
    "protein_accession",
    "protein",
    "gene_symbol",
    "gene",
    "id",
    "uniprot",
}


def _parse_sites(value: Any) -> List[Dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"none", "nan", "[]"}:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ptm_sites must be valid JSON: {value!r}") from exc
    else:
        raise ValueError(f"Unsupported ptm_sites value type: {type(value).__name__}")
    if not isinstance(parsed, list):
        raise ValueError("ptm_sites must decode to a list")
    return [dict(item) for item in parsed if isinstance(item, Mapping)]


def prepare_pmads_frame(df: pd.DataFrame, *, target_col: str = "label") -> pd.DataFrame:
    """Normalize common PMADS columns without inventing sequence data.

    ``ptm_sites`` is derived from ``ptm_position``/``ptm_type`` when those
    columns exist.  A missing sequence or target is a hard input error because
    using a random/placeholder sequence would invalidate the baseline.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("PMADS input must be a pandas DataFrame")
    frame = df.copy().reset_index(drop=True)
    if "sequence" not in frame.columns:
        raise ValueError("PMADS baseline requires a 'sequence' column")
    if target_col not in frame.columns:
        raise ValueError(f"PMADS baseline target column not found: {target_col}")

    frame["sequence"] = frame["sequence"].astype(str).str.strip().str.upper()
    if (frame["sequence"].str.len() == 0).any() or frame["sequence"].isin({"NAN", "NONE"}).any():
        raise ValueError("PMADS baseline contains empty or missing sequences")

    if "ptm_sites" not in frame.columns:
        if "ptm_position" in frame.columns:
            ptm_types = frame.get("ptm_type", pd.Series("unknown", index=frame.index))
            frame["ptm_sites"] = [
                json.dumps(
                    [{"position": int(position), "type": str(ptm_type)}] if pd.notna(position) else [],
                    ensure_ascii=False,
                )
                for position, ptm_type in zip(frame["ptm_position"], ptm_types, strict=False)
            ]
        else:
            frame["ptm_sites"] = "[]"
    else:
        # Validate all rows now so malformed JSON cannot be silently converted
        # to an empty feature vector during fitting.
        frame["ptm_sites"] = [_parse_sites(value) for value in frame["ptm_sites"]]

    if frame[target_col].isna().any():
        raise ValueError(f"PMADS baseline target contains missing values: {target_col}")
    return cast(pd.DataFrame, frame)


def profile_pmads_frame(
    frame: pd.DataFrame,
    *,
    target_col: str = "label",
    strict: bool = False,
) -> Dict[str, Any]:
    """Return quality metrics using the project's canonical data contract."""

    prepared = prepare_pmads_frame(frame, target_col=target_col)
    contract_frame = prepared.copy()
    if "cell_state" not in contract_frame.columns:
        contract_frame["cell_state"] = contract_frame[target_col]
    contract = validate_data_contract(
        contract_frame,
        require_all_rows_valid_ptm=strict,
        require_recommended=False,
    )
    sequences = contract_frame["sequence"].astype(str)
    nonstandard = int(sum(any(char not in set(AMINO_ACIDS) for char in seq) for seq in sequences))
    profile: Dict[str, Any] = {
        "row_count": int(len(prepared)),
        "unique_sequences": int(sequences.nunique()),
        "duplicate_rate": float(sequences.duplicated().mean()) if len(sequences) else 0.0,
        "nonstandard_sequence_rows": nonstandard,
        "target_column": target_col,
        "target_unique_values": int(prepared[target_col].nunique(dropna=True)),
        "contract": contract,
    }
    if strict and not contract["ok"]:
        raise ValueError("PMADS quality contract failed: " + "; ".join(contract["hard_failures"]))
    return profile


def _stratify_or_none(values: pd.Series) -> Optional[np.ndarray]:
    counts = values.value_counts(dropna=False)
    if len(counts) < 2 or int(counts.min()) < 2:
        return None
    return cast(np.ndarray, values.to_numpy())


def _indices_to_frame(frame: pd.DataFrame, indices: Sequence[int]) -> pd.DataFrame:
    return cast(pd.DataFrame, frame.iloc[list(map(int, indices))].copy().reset_index(drop=True))


def deterministic_split(
    frame: pd.DataFrame,
    *,
    target_col: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 42,
    group_col: Optional[str] = None,
    stratify: bool = True,
) -> Dict[str, Any]:
    """Create deterministic train/validation/test partitions.

    When ``group_col`` is supplied, two seeded ``GroupShuffleSplit`` operations
    ensure a protein/accession group cannot occur in more than one partition.
    Otherwise stratification is used when every class has at least two rows.
    """

    if not 0 < test_size < 1 or not 0 <= validation_size < 1:
        raise ValueError("test_size must be in (0, 1) and validation_size in [0, 1)")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be smaller than 1")
    if target_col not in frame.columns:
        raise ValueError(f"split target column not found: {target_col}")
    if len(frame) < 3:
        raise ValueError("At least three rows are required for train/validation/test splitting")

    all_indices = np.arange(len(frame), dtype=int)
    if group_col is not None:
        if group_col not in frame.columns:
            raise ValueError(f"group column not found: {group_col}")
        groups = frame[group_col]
        if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
            raise ValueError(f"group column contains missing values: {group_col}")
        if groups.nunique() < 3:
            raise ValueError("At least three distinct groups are required for a grouped split")
        first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(first.split(all_indices, groups=groups.to_numpy()))
        relative_validation = validation_size / (1.0 - test_size)
        if relative_validation == 0:
            train_idx, val_idx = train_val_idx, np.array([], dtype=int)
        else:
            second = GroupShuffleSplit(
                n_splits=1,
                test_size=relative_validation,
                random_state=seed + 1,
            )
            train_rel, val_rel = next(second.split(train_val_idx, groups=groups.iloc[train_val_idx].to_numpy()))
            train_idx, val_idx = train_val_idx[train_rel], train_val_idx[val_rel]
        split_strategy = "group_shuffle_split"
    else:
        stratify_values = _stratify_or_none(frame[target_col]) if stratify else None
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_values,
        )
        relative_validation = validation_size / (1.0 - test_size)
        if relative_validation == 0:
            train_idx, val_idx = train_val_idx, np.array([], dtype=int)
        else:
            train_stratify = _stratify_or_none(frame.iloc[train_val_idx][target_col]) if stratify else None
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=relative_validation,
                random_state=seed + 1,
                stratify=train_stratify,
            )
        split_strategy = "stratified_shuffle_split" if stratify_values is not None else "shuffle_split"

    # A defensive invariant: every row is assigned exactly once.
    partitions = [set(map(int, train_idx)), set(map(int, val_idx)), set(map(int, test_idx))]
    if any(left & right for index, left in enumerate(partitions) for right in partitions[index + 1 :]):
        raise RuntimeError("deterministic split produced overlapping partitions")
    if len(set().union(*partitions)) != len(frame):
        raise RuntimeError("deterministic split did not assign every input row")

    return {
        "train": _indices_to_frame(frame, cast(Sequence[int], train_idx)),
        "validation": _indices_to_frame(frame, cast(Sequence[int], val_idx)),
        "test": _indices_to_frame(frame, cast(Sequence[int], test_idx)),
        "indices": {
            "train": [int(value) for value in train_idx],
            "validation": [int(value) for value in val_idx],
            "test": [int(value) for value in test_idx],
        },
        "strategy": split_strategy,
        "seed": int(seed),
        "test_size": float(test_size),
        "validation_size": float(validation_size),
        "group_col": group_col,
        "stratify": bool(stratify),
    }


class PMADSRidgeBaseline:
    """Transparent Ridge model for PMADS-style row-level supervision."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        task: str = "classification",
        feature_columns: Optional[Sequence[str]] = None,
        max_iter: int = 10_000,
    ) -> None:
        if alpha <= 0:
            raise ValueError("Ridge alpha must be positive")
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.alpha = float(alpha)
        self.task = task
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.max_iter = int(max_iter)
        self.pipeline: Optional[Pipeline] = None
        self.target_col: Optional[str] = None
        self.feature_names_: List[str] = []
        self.numeric_columns_: List[str] = []
        self.ptm_types_: List[str] = []
        self.classes_: Optional[np.ndarray] = None

    @staticmethod
    def _site_features(value: Any, sequence_length: int) -> Dict[str, Any]:
        sites = _parse_sites(value)
        positions: List[float] = []
        types: List[str] = []
        for site in sites:
            position = site.get("position")
            try:
                if position is not None and np.isfinite(float(position)):
                    positions.append(float(position))
            except (TypeError, ValueError):
                continue
            if site.get("type") is not None:
                types.append(str(site["type"]).strip().lower())
        normalized = np.asarray(positions, dtype=float) / max(float(sequence_length), 1.0)
        return {
            "ptm_count": float(len(sites)),
            "ptm_position_mean": float(normalized.mean()) if len(normalized) else 0.0,
            "ptm_position_std": float(normalized.std()) if len(normalized) else 0.0,
            "ptm_position_min": float(normalized.min()) if len(normalized) else 0.0,
            "ptm_position_max": float(normalized.max()) if len(normalized) else 0.0,
            "_types": types,  # internal value consumed by _feature_matrix
        }

    def _fit_feature_schema(self, frame: pd.DataFrame, target_col: str) -> None:
        if self.feature_columns is not None:
            missing = [col for col in self.feature_columns if col not in frame.columns]
            if missing:
                raise ValueError("Requested feature columns missing: " + ", ".join(missing))
            self.numeric_columns_ = list(self.feature_columns)
        else:
            self.numeric_columns_ = [
                column
                for column in frame.columns
                if column not in _EXCLUDED_NUMERIC_COLUMNS
                and column not in {"sequence", "ptm_sites", target_col}
                and pd.api.types.is_numeric_dtype(frame[column])
            ]
        observed_types = set()
        for value, sequence in zip(frame["ptm_sites"], frame["sequence"], strict=False):
            observed_types.update(self._site_features(value, len(sequence))["_types"])
        self.ptm_types_ = sorted(observed_types)
        self.feature_names_ = ["sequence_length"] + [f"aa_fraction_{aa}" for aa in AMINO_ACIDS]
        self.feature_names_ += [
            "ptm_count",
            "ptm_position_mean",
            "ptm_position_std",
            "ptm_position_min",
            "ptm_position_max",
        ]
        self.feature_names_ += [f"ptm_type__{ptm_type}" for ptm_type in self.ptm_types_]
        self.feature_names_ += [f"numeric__{column}" for column in self.numeric_columns_]

    def _feature_matrix(self, frame: pd.DataFrame, *, fit_schema: bool = False) -> np.ndarray:
        if fit_schema:
            if self.target_col is None:
                raise RuntimeError("target_col must be set before fitting feature schema")
            self._fit_feature_schema(frame, self.target_col)
        if not self.feature_names_:
            raise RuntimeError("Feature schema is not initialized; call fit first")

        if frame.empty:
            return np.asarray([], dtype=np.float64)

        records = frame.to_dict(orient="records")
        numeric_values = {
            column: pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            for column in self.numeric_columns_
        }
        rows: List[List[float]] = []
        for row_index, row in enumerate(records):
            sequence = str(row["sequence"])
            length = max(len(sequence), 1)
            counts = {aa: sequence.count(aa) / length for aa in AMINO_ACIDS}
            site_features = self._site_features(row["ptm_sites"], length)
            values = [float(length)] + [float(counts[aa]) for aa in AMINO_ACIDS]
            values += [
                site_features[name]
                for name in (
                    "ptm_count",
                    "ptm_position_mean",
                    "ptm_position_std",
                    "ptm_position_min",
                    "ptm_position_max",
                )
            ]
            type_set = set(site_features["_types"])
            values += [1.0 if ptm_type in type_set else 0.0 for ptm_type in self.ptm_types_]
            values += [float(numeric_values[column][row_index]) for column in self.numeric_columns_]
            rows.append(values)
        return np.asarray(rows, dtype=np.float64)

    def _encode_target(self, series: pd.Series, *, fit: bool) -> np.ndarray:
        if self.task == "regression":
            values = pd.to_numeric(series, errors="coerce")
            if values.isna().any():
                raise ValueError("Regression target must be numeric and finite")
            return cast(np.ndarray, values.to_numpy(dtype=np.float64))

        if fit:
            unique = list(pd.unique(series))
            self.classes_ = np.asarray(sorted(unique, key=lambda value: str(value)), dtype=object)
        if self.classes_ is None or len(self.classes_) == 0:
            raise RuntimeError("Classification classes are not initialized")
        mapping = {str(value): index for index, value in enumerate(self.classes_)}
        encoded = [mapping.get(str(value), -1) for value in series]
        if any(value < 0 for value in encoded):
            raise ValueError("Evaluation target contains a class not observed during fitting")
        return np.asarray(encoded, dtype=np.float64)

    def fit(self, frame: pd.DataFrame, *, target_col: str = "label") -> "PMADSRidgeBaseline":
        prepared = prepare_pmads_frame(frame, target_col=target_col)
        self.target_col = target_col
        x = self._feature_matrix(prepared, fit_schema=True)
        y = self._encode_target(prepared[target_col], fit=True)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, max_iter=self.max_iter)),
            ]
        )
        self.pipeline.fit(x, y)
        return self

    def _require_fitted(self) -> Pipeline:
        if self.pipeline is None or self.target_col is None:
            raise RuntimeError("PMADSRidgeBaseline must be fitted before prediction")
        return self.pipeline

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        pipeline = self._require_fitted()
        target_col = self.target_col
        if target_col is None:  # narrowed for static checkers
            raise RuntimeError("target_col is not initialized")
        prepared = prepare_pmads_frame(frame, target_col=target_col)
        return np.asarray(pipeline.predict(self._feature_matrix(prepared)), dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(frame)
        if self.task == "regression":
            return raw
        if self.classes_ is None:
            raise RuntimeError("Classification classes are not initialized")
        indices = np.rint(raw).astype(int)
        indices = np.clip(indices, 0, len(self.classes_) - 1)
        return cast(np.ndarray, self.classes_[indices])

    def evaluate(self, frame: pd.DataFrame) -> Dict[str, float]:
        if self.target_col is None:
            raise RuntimeError("target_col is not initialized")
        prepared = prepare_pmads_frame(frame, target_col=self.target_col)
        raw = self.predict_raw(prepared)
        if self.task == "regression":
            target = pd.to_numeric(prepared[self.target_col], errors="coerce").to_numpy(dtype=float)
            mse = float(mean_squared_error(target, raw))
            return {
                "mae": float(mean_absolute_error(target, raw)),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(target, raw)) if len(target) > 1 else float("nan"),
            }

        target = self._encode_target(prepared[self.target_col], fit=False).astype(int)
        predictions = self.predict(prepared)
        mapping = {str(value): index for index, value in enumerate(cast(np.ndarray, self.classes_))}
        predicted_indices = np.asarray([mapping[str(value)] for value in predictions], dtype=int)
        return {
            "accuracy": float(accuracy_score(target, predicted_indices)),
            "balanced_accuracy": float(balanced_accuracy_score(target, predicted_indices)),
            "macro_f1": float(f1_score(target, predicted_indices, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(target, predicted_indices)),
            "raw_score_mean": float(raw.mean()) if len(raw) else 0.0,
        }

    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "task": self.task,
            "feature_columns": self.feature_columns,
            "max_iter": self.max_iter,
            "target_col": self.target_col,
            "feature_names": list(self.feature_names_),
        }


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _frame_digest(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_pmads_ridge(
    frame: pd.DataFrame,
    *,
    target_col: str = "label",
    task: str = "classification",
    alpha: float = 1.0,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 42,
    group_col: Optional[str] = None,
    strict_quality: bool = False,
) -> Dict[str, Any]:
    """Fit and evaluate the baseline on a deterministic three-way split."""

    prepared = prepare_pmads_frame(frame, target_col=target_col)
    quality = profile_pmads_frame(prepared, target_col=target_col, strict=strict_quality)
    split = deterministic_split(
        prepared,
        target_col=target_col,
        test_size=test_size,
        validation_size=validation_size,
        seed=seed,
        group_col=group_col,
        stratify=task == "classification",
    )
    model = PMADSRidgeBaseline(alpha=alpha, task=task)
    model.fit(split["train"], target_col=target_col)

    metric_results: Dict[str, Dict[str, float]] = {}
    prediction_rows: List[Dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        partition = cast(pd.DataFrame, split[split_name])
        metric_results[split_name] = model.evaluate(partition)
        raw = model.predict_raw(partition)
        predictions = model.predict(partition)
        for local_index, (raw_score, prediction, target) in enumerate(
            zip(raw, predictions, partition[target_col].tolist(), strict=False)
        ):
            prediction_rows.append(
                {
                    "split": split_name,
                    "row_index": int(local_index),
                    "target": target.item() if isinstance(target, np.generic) else target,
                    "prediction": prediction.item() if isinstance(prediction, np.generic) else prediction,
                    "raw_score": float(raw_score),
                }
            )

    return {
        "model": model,
        "metrics": metric_results,
        "predictions": pd.DataFrame(prediction_rows),
        "split": split,
        "quality": quality,
        "input_digest": _frame_digest(prepared),
    }


def save_baseline_artifact(
    result: Mapping[str, Any],
    output_dir: Union[str, Path],
    *,
    input_path: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    is_demo_data: bool = False,
) -> Dict[str, str]:
    """Write model, metrics, predictions and an auditable manifest."""

    model = result.get("model")
    if not isinstance(model, PMADSRidgeBaseline):
        raise TypeError("result['model'] must be a fitted PMADSRidgeBaseline")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model_path = output / "ridge_model.joblib"
    metrics_path = output / "metrics.json"
    predictions_path = output / "predictions.csv"
    manifest_output = output / "baseline_manifest.json"
    joblib.dump(model, model_path)

    metrics = cast(Dict[str, Dict[str, float]], result["metrics"])
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    predictions = cast(pd.DataFrame, result["predictions"])
    predictions.to_csv(predictions_path, index=False)

    raw_split = result.get("split", {})
    split_metadata = (
        {key: value for key, value in raw_split.items() if key not in {"train", "validation", "test"}}
        if isinstance(raw_split, Mapping)
        else {}
    )
    manifest_info: Dict[str, Any] = {
        "schema_version": "ptm2cellnet.ridge-baseline.v1",
        "created_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "model_class": "sklearn.linear_model.Ridge",
        "baseline_class": "PMADSRidgeBaseline",
        "model_kind": "demo" if is_demo_data else "real_or_user_supplied",
        "not_for_biological_use": bool(is_demo_data),
        "model_parameters": model.get_params(),
        "run_parameters": dict(parameters or {}),
        "input_path": str(input_path) if input_path is not None else None,
        "input_digest": result.get("input_digest"),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_digest": None,
        "split": split_metadata,
        "quality": result.get("quality", {}),
        "metrics_path": metrics_path.name,
        "predictions_path": predictions_path.name,
        "model_path": model_path.name,
        "metrics": metrics,
    }
    if manifest_path is not None:
        try:
            manifest = load_manifest(manifest_path)
            manifest_info["manifest_digest"] = manifest_digest(manifest)
        except (DataManifestError, FileNotFoundError) as exc:
            raise ValueError(f"Unable to load supplied data manifest: {exc}") from exc
    manifest_output.write_text(
        json.dumps(manifest_info, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "manifest": str(manifest_output),
    }


__all__ = [
    "PMADSRidgeBaseline",
    "deterministic_split",
    "prepare_pmads_frame",
    "profile_pmads_frame",
    "run_pmads_ridge",
    "save_baseline_artifact",
]
