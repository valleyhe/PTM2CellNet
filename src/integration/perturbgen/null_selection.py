"""Deterministic matched-null selection and null-distribution manifests."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_ORDER = ("mean_expression", "detection_rate", "fc", "token_rank")
_FC_COLUMNS = ("log2fc", "fc", "fold_change")
_SCHEMA_VERSION = "perturbgen_null_selection/v1"
_DISTRIBUTION_SCHEMA_VERSION = "perturbgen_null_distribution/v1"
_ENSEMBL_RE = re.compile(r"^ENSG\d+$")
_ENSEMBL_VERSION_RE = re.compile(r"\.\d+$")
_VALID_PATHS = {"source_intervention", "within_state"}
_VALID_MODES = {"mask", "pad", "delete", "overexpress"}


def _canonical_ensembl_id(value: Any) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError("Ensembl ID must be a non-empty ENSG identifier")
    text = str(value).strip()
    canonical = _ENSEMBL_VERSION_RE.sub("", text)
    if not _ENSEMBL_RE.fullmatch(canonical):
        raise ValueError(f"invalid Ensembl gene id: {value!r}")
    return canonical


def _canonical_ids(values: Sequence[Any], *, name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of IDs")
    result = [_canonical_ensembl_id(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate Ensembl IDs")
    return result


def _cohort_gene_ids(cohort: Any, width: int) -> list[str]:
    var = getattr(cohort, "var", None)
    values: Any = None
    if isinstance(var, Mapping) and "ensembl_id" in var:
        values = var["ensembl_id"]
    elif var is not None:
        columns = getattr(var, "columns", ())
        if "ensembl_id" in columns:
            values = var["ensembl_id"]
    if values is None:
        raise ValueError("cohort must provide explicit var['ensembl_id']")
    ids = _canonical_ids(list(values), name="cohort gene IDs")
    if len(ids) != width:
        raise ValueError("cohort gene ID count does not match cohort.X columns")
    return ids


def _cohort_matrix(cohort: Any) -> np.ndarray:
    if not hasattr(cohort, "X"):
        raise ValueError("cohort must provide X")
    raw = cohort.X
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("cohort.X must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("cohort.X must contain at least one cell and one gene")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("cohort.X must contain finite non-negative values")
    return matrix


def _feature_table_values(feature_table: pd.DataFrame | Mapping[Any, Any] | None) -> tuple[dict[str, float], str]:
    if feature_table is None:
        return {}, "unavailable/default"
    if isinstance(feature_table, pd.DataFrame):
        id_values = feature_table["ensembl_id"] if "ensembl_id" in feature_table.columns else feature_table.index
        fc_column = next((column for column in _FC_COLUMNS if column in feature_table.columns), None)
        if fc_column is None:
            return {}, "unavailable/default"
        values: dict[str, float] = {}
        for gene, value in zip(id_values, feature_table[fc_column], strict=True):
            canonical = _canonical_ensembl_id(gene)
            if canonical in values:
                raise ValueError("feature_table contains duplicate Ensembl IDs")
            if isinstance(value, bool):
                raise ValueError("feature_table FC values must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("feature_table FC values must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError("feature_table FC values must be finite")
            values[canonical] = number
        return values, f"feature_table:{fc_column}"
    if isinstance(feature_table, Mapping):
        values = {}
        for gene, value in feature_table.items():
            canonical = _canonical_ensembl_id(gene)
            if canonical in values:
                raise ValueError("feature_table contains duplicate Ensembl IDs")
            if isinstance(value, bool):
                raise ValueError("feature_table FC values must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("feature_table FC values must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError("feature_table FC values must be finite")
            values[canonical] = number
        return values, "feature_table:mapping"
    raise ValueError("feature_table must be a DataFrame, mapping, or None")


def _token_values(
    token_vocabulary: Mapping[Any, Any] | Sequence[Any] | None,
) -> tuple[dict[str, float], str]:
    if token_vocabulary is None:
        return {}, "unavailable/default"
    values: dict[str, float] = {}
    if isinstance(token_vocabulary, Mapping):
        for gene, token in token_vocabulary.items():
            canonical = _canonical_ensembl_id(gene)
            if canonical in values:
                raise ValueError("token_vocabulary contains duplicate Ensembl IDs")
            if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
                raise ValueError("token_vocabulary mapping values must be integers")
            if int(token) < 1:
                raise ValueError("token_vocabulary mapping ranks must be >= 1")
            values[canonical] = float(int(token))
        return values, "token_vocabulary:mapping"
    if isinstance(token_vocabulary, (str, bytes)):
        raise ValueError("token_vocabulary must be a mapping or ordered sequence")
    for rank, gene in enumerate(token_vocabulary, start=1):
        canonical = _canonical_ensembl_id(gene)
        if canonical in values:
            raise ValueError("token_vocabulary contains duplicate Ensembl IDs")
        values[canonical] = float(rank)
    return values, "token_vocabulary:sequence"


def _feature_payload(raw: Mapping[str, float], vector: np.ndarray) -> dict[str, Any]:
    return {
        **{name: float(raw[name]) for name in FEATURE_ORDER},
        "feature_order": list(FEATURE_ORDER),
        "matching_vector": [float(value) for value in vector],
    }


def select_matched_nulls(
    cohort: Any,
    target_ensembl_id: str,
    candidate_ensembl_ids: Sequence[str],
    *,
    feature_table: pd.DataFrame | Mapping[Any, Any] | None = None,
    token_vocabulary: Mapping[Any, int] | Sequence[str] | None = None,
    required_count: int = 99,
) -> dict[str, Any]:
    """Select deterministic expression-matched null genes from a cohort."""
    if isinstance(required_count, bool) or not isinstance(required_count, int) or required_count < 99:
        raise ValueError("required_count must be an integer >= 99")
    target_id = _canonical_ensembl_id(target_ensembl_id)
    candidate_ids = _canonical_ids(candidate_ensembl_ids, name="candidate_ensembl_ids")
    if target_id in candidate_ids:
        raise ValueError("target and candidate IDs must be distinct")

    matrix = _cohort_matrix(cohort)
    cohort_ids = _cohort_gene_ids(cohort, matrix.shape[1])
    if len(cohort_ids) != len(set(cohort_ids)):
        raise ValueError("cohort gene IDs contain duplicates")
    if target_id not in cohort_ids:
        raise ValueError(f"target {target_id!r} is not present in cohort")
    missing_candidates = [gene for gene in candidate_ids if gene not in cohort_ids]
    if missing_candidates:
        raise ValueError(f"candidate IDs not present in cohort: {', '.join(missing_candidates)}")

    means = np.mean(matrix, axis=0)
    detection = np.mean(matrix > 0, axis=0)
    expression_features = {
        gene: {
            "mean_expression": float(means[index]),
            "detection_rate": float(detection[index]),
        }
        for index, gene in enumerate(cohort_ids)
    }
    fc_values, fc_source = _feature_table_values(feature_table)
    token_values, token_source = _token_values(token_vocabulary)
    audit_ids = (target_id, *candidate_ids)
    if any(gene not in fc_values for gene in audit_ids) and "unavailable/default" not in fc_source:
        fc_source = f"{fc_source}; unavailable/default=0.0"
    if any(gene not in token_values for gene in audit_ids) and "unavailable/default" not in token_source:
        token_source = f"{token_source}; unknown/default=0.0"

    raw_features = {
        gene: {
            **expression_features[gene],
            "fc": float(fc_values.get(gene, 0.0)),
            "token_rank": float(token_values.get(gene, 0.0)),
        }
        for gene in cohort_ids
    }
    excluded_ids = sorted({target_id, *candidate_ids})
    excluded_set = set(excluded_ids)
    eligible_ids = [gene for gene in cohort_ids if gene not in excluded_set]
    if len(eligible_ids) < required_count:
        raise ValueError(f"cohort has only {len(eligible_ids)} eligible nulls; required {required_count}")

    candidate_matrix = np.asarray([[raw_features[gene][name] for name in FEATURE_ORDER] for gene in eligible_ids])
    minimum = candidate_matrix.min(axis=0)
    span = candidate_matrix.max(axis=0) - minimum
    candidate_vectors = np.divide(
        candidate_matrix - minimum,
        span,
        out=np.zeros_like(candidate_matrix, dtype=float),
        where=span != 0,
    )
    target_vector = np.divide(
        np.asarray([raw_features[target_id][name] for name in FEATURE_ORDER]) - minimum,
        span,
        out=np.zeros(len(FEATURE_ORDER), dtype=float),
        where=span != 0,
    )
    distances = np.linalg.norm(candidate_vectors - target_vector, axis=1)
    ranked = sorted(
        zip(distances, eligible_ids, candidate_vectors, strict=True),
        key=lambda item: (float(item[0]), item[1]),
    )
    selected = [
        {
            "ensembl_id": gene,
            "match_features": _feature_payload(raw_features[gene], vector),
            "distance": float(distance),
        }
        for distance, gene, vector in ranked[:required_count]
    ]
    feature_sources = {
        "mean_expression": "cohort.X",
        "detection_rate": "cohort.X",
        "fc": fc_source,
        "token_rank": token_source,
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "source": {
            "cohort_matrix": "cohort.X",
            "cohort_ensembl_ids": "cohort.var['ensembl_id']",
            "feature_sources": feature_sources,
        },
        "feature_sources": feature_sources,
        "feature_order": list(FEATURE_ORDER),
        "target": {
            "ensembl_id": target_id,
            "features": _feature_payload(raw_features[target_id], target_vector),
        },
        "excluded_ids": excluded_ids,
        "excluded_candidate_ids": candidate_ids,
        "required_count": required_count,
        "selected_nulls": selected,
    }


def _write_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return resolved


def _validate_selection_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema_version",
        "source",
        "target",
        "excluded_ids",
        "excluded_candidate_ids",
        "required_count",
        "selected_nulls",
    )
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(f"selection manifest missing required fields: {', '.join(missing)}")
    if manifest["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(f"unsupported selection manifest schema_version {manifest['schema_version']!r}")
    required_count = manifest["required_count"]
    if isinstance(required_count, bool) or not isinstance(required_count, int) or required_count < 99:
        raise ValueError("selection manifest required_count must be an integer >= 99")
    if not isinstance(manifest["source"], Mapping) or not manifest["source"]:
        raise ValueError("selection manifest source must be a non-empty mapping")
    target = manifest["target"]
    if not isinstance(target, Mapping) or "ensembl_id" not in target or "features" not in target:
        raise ValueError("selection manifest target must contain ensembl_id and features")
    target_id = _canonical_ensembl_id(target["ensembl_id"])
    target_features = target["features"]
    if not isinstance(target_features, Mapping):
        raise ValueError("selection manifest target features must be a mapping")
    _validate_feature_payload(target_features, name="target features")
    excluded_ids = _canonical_ids(manifest["excluded_ids"], name="selection manifest excluded_ids")
    if target_id not in excluded_ids:
        raise ValueError("selection manifest excluded_ids must contain target")
    candidate_ids = _canonical_ids(
        manifest["excluded_candidate_ids"], name="selection manifest excluded_candidate_ids"
    )
    if target_id in candidate_ids:
        raise ValueError("selection manifest excluded_candidate_ids must not contain target")
    if set(excluded_ids) != {target_id, *candidate_ids}:
        raise ValueError("selection manifest excluded_ids must equal target plus all candidate IDs")
    selected = manifest["selected_nulls"]
    if not isinstance(selected, list) or len(selected) < required_count:
        raise ValueError(f"selection manifest must contain at least {required_count} selected nulls")
    selected_ids: list[str] = []
    for item in selected:
        if not isinstance(item, Mapping):
            raise ValueError("selection manifest selected_nulls entries must be mappings")
        if not all(name in item for name in ("ensembl_id", "match_features", "distance")):
            raise ValueError("selection manifest selected nulls need ensembl_id, match_features, and distance")
        null_id = _canonical_ensembl_id(item["ensembl_id"])
        selected_ids.append(null_id)
        if not isinstance(item["match_features"], Mapping):
            raise ValueError("selection manifest match_features must be a mapping")
        _validate_feature_payload(item["match_features"], name="match_features")
        try:
            distance = float(item["distance"])
        except (TypeError, ValueError) as exc:
            raise ValueError("selection manifest distances must be numeric") from exc
        if not math.isfinite(distance):
            raise ValueError("selection manifest distances must be finite")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selection manifest selected null IDs must be unique")
    if set(selected_ids) & set(excluded_ids):
        raise ValueError("selection manifest selected nulls must exclude target and candidate IDs")
    return dict(manifest)


def _validate_feature_payload(features: Mapping[str, Any], *, name: str) -> None:
    if features.get("feature_order") != list(FEATURE_ORDER):
        raise ValueError(f"selection manifest {name} has an invalid feature_order")
    for feature_name in FEATURE_ORDER:
        value = features.get(feature_name)
        if value is None or isinstance(value, bool):
            raise ValueError(f"selection manifest {name} values must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"selection manifest {name} values must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"selection manifest {name} values must be finite")
    vector = features.get("matching_vector")
    if not isinstance(vector, list) or len(vector) != len(FEATURE_ORDER):
        raise ValueError(f"selection manifest {name} matching_vector is invalid")
    for value in vector:
        if isinstance(value, bool):
            raise ValueError(f"selection manifest {name} matching_vector must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"selection manifest {name} matching_vector must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"selection manifest {name} matching_vector must be finite")


def write_null_selection_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    return _write_json(_validate_selection_manifest(manifest), output_path)


def _path_kind(value: Any, *, name: str = "path") -> str:
    text = str(value).strip()
    if text not in _VALID_PATHS:
        raise ValueError(f"{name} must be one of {sorted(_VALID_PATHS)}, got {text!r}")
    return text


def _mode(value: Any, *, name: str = "mode") -> str:
    text = str(value).strip()
    if text not in _VALID_MODES:
        raise ValueError(f"{name} must be one of {sorted(_VALID_MODES)}, got {text!r}")
    return text


def _seed(value: Any, *, name: str = "seed") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def summarize_null_distribution(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_ensembl_id: str,
    path: str | Path,
    mode: str,
    seed: int,
    required_count: int = 99,
    output_path: str | Path | None = None,
    selection_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if isinstance(required_count, bool) or not isinstance(required_count, int) or required_count < 99:
        raise ValueError("required_count must be an integer >= 99")
    seed = _seed(seed)
    candidate_id = _canonical_ensembl_id(candidate_ensembl_id)
    expected_path = _path_kind(path)
    mode = _mode(mode)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence")
    if len(records) < required_count:
        raise ValueError(f"at least {required_count} records are required")

    parsed: list[tuple[str, float]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("every null record must be a mapping")
        required = ("candidate_ensembl_id", "null_ensembl_id", "path", "mode", "seed", "rescue_excl_target")
        missing = [name for name in required if name not in record]
        if missing:
            raise ValueError(f"null record missing required fields: {', '.join(missing)}")
        if _canonical_ensembl_id(record["candidate_ensembl_id"]) != candidate_id:
            raise ValueError("null record candidate_ensembl_id does not match")
        if (
            _path_kind(record["path"]) != expected_path
            or _mode(record["mode"]) != mode
            or type(record["seed"]) is not int
            or record["seed"] < 0
            or record["seed"] != seed
        ):
            raise ValueError("null record binding does not match")
        null_id = _canonical_ensembl_id(record["null_ensembl_id"])
        if null_id == candidate_id:
            raise ValueError("null record must not select the candidate gene")
        if null_id in seen:
            raise ValueError("null records contain duplicate null_ensembl_id values")
        seen.add(null_id)
        if isinstance(record["rescue_excl_target"], bool):
            raise ValueError("null record rescue_excl_target must be numeric")
        try:
            rescue = float(record["rescue_excl_target"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("null record rescue_excl_target must be numeric") from exc
        if not math.isfinite(rescue):
            raise ValueError("null record rescue_excl_target must be finite")
        parsed.append((null_id, rescue))

    parsed.sort(key=lambda item: item[0])
    payload: dict[str, Any] = {
        "schema_version": _DISTRIBUTION_SCHEMA_VERSION,
        "candidate_ensembl_id": candidate_id,
        "path": expected_path,
        "mode": mode,
        "seed": seed,
        "required_count": required_count,
        "values": [value for _, value in parsed],
        "null_ensembl_ids": [gene for gene, _ in parsed],
    }
    if selection_manifest_path is not None:
        payload["selection_manifest_path"] = str(selection_manifest_path)
    if output_path is not None:
        _write_json(payload, output_path)
    return payload


def _validate_values(
    payload: Mapping[str, Any],
    *,
    required_count: int,
    require_null_ids: bool = False,
) -> dict[str, Any]:
    if isinstance(required_count, bool) or not isinstance(required_count, int) or required_count < 1:
        raise ValueError("required_count must be a positive integer")
    declared_count = payload.get("required_count")
    minimum_count = required_count
    if declared_count is not None:
        if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count < 99:
            raise ValueError("null distribution required_count must be an integer >= 99")
        minimum_count = max(minimum_count, declared_count)
    values = payload.get("values")
    if not isinstance(values, list) or len(values) < minimum_count:
        raise ValueError(f"null distribution must contain at least {minimum_count} values")
    converted = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("null distribution values must be numeric")
        try:
            converted.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("null distribution values must be numeric") from exc
    if not np.isfinite(converted).all():
        raise ValueError("null distribution values must be finite")
    result = dict(payload)
    result["values"] = converted
    null_ids = result.get("null_ensembl_ids")
    if require_null_ids and null_ids is None:
        raise ValueError("null distribution manifest must include null_ensembl_ids")
    if null_ids is not None:
        if not isinstance(null_ids, list) or len(null_ids) != len(converted):
            raise ValueError("null_ensembl_ids must match values length")
        canonical_ids = [_canonical_ensembl_id(value) for value in null_ids]
        if len(canonical_ids) != len(set(canonical_ids)):
            raise ValueError("null_ensembl_ids must be unique")
        result["null_ensembl_ids"] = canonical_ids
    return result


def _validate_distribution_manifest(payload: Mapping[str, Any], *, required_count: int) -> dict[str, Any]:
    required = (
        "schema_version",
        "candidate_ensembl_id",
        "path",
        "mode",
        "seed",
        "required_count",
        "values",
        "null_ensembl_ids",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"null distribution manifest missing required fields: {', '.join(missing)}")
    if payload["schema_version"] != _DISTRIBUTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported null distribution schema_version {payload['schema_version']!r}")
    result = _validate_values(payload, required_count=required_count, require_null_ids=True)
    candidate_id = _canonical_ensembl_id(result["candidate_ensembl_id"])
    path_kind = _path_kind(result["path"])
    mode = _mode(result["mode"])
    seed = _seed(result["seed"])
    if candidate_id in result["null_ensembl_ids"]:
        raise ValueError("null distribution must exclude the candidate gene")
    result.update(
        {
            "candidate_ensembl_id": candidate_id,
            "path": path_kind,
            "mode": mode,
            "seed": seed,
        }
    )
    return result


def _check_binding(
    payload: Mapping[str, Any],
    *,
    candidate_ensembl_id: str | None,
    path_name: str | Path | None,
    mode: str | None,
    seed: int | None,
) -> None:
    expected = {
        "candidate_ensembl_id": None if candidate_ensembl_id is None else _canonical_ensembl_id(candidate_ensembl_id),
        "path": None if path_name is None else _path_kind(path_name, name="path_name"),
        "mode": None if mode is None else _mode(mode),
        "seed": None if seed is None else _seed(seed),
    }
    for field, value in expected.items():
        if value is None:
            continue
        actual = payload.get(field)
        if field == "candidate_ensembl_id" and actual is not None:
            actual = _canonical_ensembl_id(actual)
        if actual != value:
            raise ValueError(f"null distribution {field} binding does not match")


def load_null_distribution_manifest(
    path: str | Path,
    *,
    candidate_ensembl_id: str | None = None,
    path_name: str | Path | None = None,
    mode: str | None = None,
    seed: int | None = None,
    required_count: int = 99,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if seed is not None:
        _seed(seed)
    expected_bound = any(value is not None for value in (candidate_ensembl_id, path_name, mode, seed))

    if isinstance(payload, list):
        if expected_bound:
            raise ValueError("old unbound null distribution cannot satisfy expected binding")
        return _validate_values({"values": payload}, required_count=required_count)
    if not isinstance(payload, Mapping):
        raise ValueError("null distribution manifest must be a mapping or value list")

    schema_version = payload.get("schema_version")
    if schema_version is None:
        if expected_bound:
            raise ValueError("old unbound null distribution cannot satisfy expected binding")
        return _validate_values({"values": payload.get("values")}, required_count=required_count)
    if schema_version != _DISTRIBUTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported null distribution schema_version {schema_version!r}")

    if isinstance(payload.get("distributions"), list):
        distributions = payload["distributions"]
        matches = []
        for distribution in distributions:
            if not isinstance(distribution, Mapping):
                raise ValueError("distributions entries must be mappings")
            validated = _validate_distribution_manifest(distribution, required_count=required_count)
            try:
                _check_binding(
                    validated,
                    candidate_ensembl_id=candidate_ensembl_id,
                    path_name=path_name,
                    mode=mode,
                    seed=seed,
                )
            except ValueError:
                continue
            matches.append(validated)
        if len(matches) != 1:
            raise ValueError("null distribution index must select exactly one distribution")
        return matches[0]

    validated = _validate_distribution_manifest(payload, required_count=required_count)
    _check_binding(
        validated,
        candidate_ensembl_id=candidate_ensembl_id,
        path_name=path_name,
        mode=mode,
        seed=seed,
    )
    return validated
