"""Boundary adapter for the external KSTAR 1.2.x environment.

This module deliberately has no KSTAR import.  It turns the stage-1
standardized PTM table into explicit increased/decreased evidence columns and
turns the two directional KSTAR result tables back into the project's signed
``ptm_activity.tsv`` contract.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, cast

import pandas as pd

# Keep this boundary importable in the independent KSTAR environment.  The
# core ``src`` package bootstrap requires optional project dependencies; the
# adapter only needs the frozen standard-table column names.
ACTIVITY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "activity_unit",
    "regulator_id",
    "regulator_type",
    "condition_or_contrast",
    "activity_score",
    "activity_direction",
    "activity_pvalue",
    "activity_qvalue",
    "n_substrates",
    "network_coverage",
    "method",
    "method_version",
    "input_manifest",
)

KSTAR_ADAPTER_SCHEMA_VERSION = "ptm2cellnet.kstar-adapter-manifest/v2"
KSTAR_INPUT_MANIFEST_SCHEMA_VERSION = "ptm2cellnet.ptm-input-manifest/v1"
KSTAR_METHOD = "KSTAR"
_DIRECTIONS = ("increased", "decreased")
_RESIDUE_PATTERN = re.compile(r"^[STY][1-9][0-9]*$")
_VERSION_PATTERN = re.compile(r"^1\.2\.\d+$")
_ENSEMBL_PATTERN = re.compile(r"^ENSG\d+$")
_ENSEMBL_VERSION_PATTERN = re.compile(r"\.\d+$")


class KSTARAdapterError(ValueError):
    """Raised when a KSTAR handover violates an explicit contract."""


@dataclass(frozen=True)
class KSTARInputAudit:
    """Counts and frozen choices recorded beside a KSTAR input handover."""

    input_rows: int
    donor_rows: int
    usable_rows: int
    complete_sites: int
    increased_sites: int
    decreased_sites: int
    unchanged_sites: int
    excluded_no_donor_rows: int
    excluded_nonfinite_rows: int
    excluded_incomplete_sites: int
    case_condition: str
    reference_condition: str
    contrast: str
    value_scale: str
    fold_threshold: float
    case_donors: tuple[str, ...]
    reference_donors: tuple[str, ...]
    source_datasets: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_table(source: str | Path | pd.DataFrame, *, name: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return cast(pd.DataFrame, source.copy())
    try:
        path = Path(source).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise KSTARAdapterError(f"{name} does not exist: {source}") from exc
    try:
        return pd.read_csv(path, sep="\t")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise KSTARAdapterError(f"could not read {name}: {path}: {exc}") from exc


def _require_manifest(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    """Require a structured stage-1 manifest, not just any JSON object.

    An empty ``{}`` (or a free-form JSON blob) can no longer pass as PTM
    provenance: the manifest must carry the frozen stage-1 schema version,
    source provenance with a hash, and standardization statistics.
    """

    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise KSTARAdapterError(f"input manifest does not exist: {path}") from exc
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KSTARAdapterError(f"input manifest is not valid JSON: {resolved}") from exc
    if not isinstance(payload, Mapping):
        raise KSTARAdapterError("input manifest must contain a JSON object")
    schema = str(payload.get("schema_version", "")).strip()
    if schema != KSTAR_INPUT_MANIFEST_SCHEMA_VERSION:
        raise KSTARAdapterError(
            f"input manifest schema_version must be {KSTAR_INPUT_MANIFEST_SCHEMA_VERSION!r}, got {schema!r}"
        )
    source = payload.get("source")
    if not isinstance(source, Mapping) or not str(source.get("sha256", "")).strip():
        raise KSTARAdapterError("input manifest must record source provenance with a sha256")
    standardization = payload.get("standardization")
    if not isinstance(standardization, Mapping) or not standardization:
        raise KSTARAdapterError("input manifest must record standardization statistics")
    return resolved, payload


def _nonempty(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype("string")
    return values.notna() & values.str.strip().ne("")


def _join_unique(values: pd.Series) -> str:
    return ";".join(sorted({str(value).strip() for value in values if str(value).strip()}))


def build_kstar_input(
    standardized_ptm: str | Path | pd.DataFrame,
    *,
    input_manifest: str | Path,
    case_condition: str,
    reference_condition: str,
    contrast: str,
    fold_threshold: float = 1.2,
    min_donors_per_state: int = 1,
) -> tuple[pd.DataFrame, KSTARInputAudit]:
    """Build explicit increased/decreased KSTAR evidence from stage 1 output.

    ``ptm_value_normalized`` is the only value consumed.  Rows without an
    explicit donor or finite normalized value stay out of KSTAR evidence and
    are counted in the audit; no raw-value fallback is allowed.  Formal runs
    raise ``min_donors_per_state`` so a one-donor-per-state table cannot enter
    formal KSTAR evidence.
    """

    if isinstance(min_donors_per_state, bool) or not isinstance(min_donors_per_state, int) or min_donors_per_state < 1:
        raise KSTARAdapterError("min_donors_per_state must be an integer >= 1")
    manifest_path, _ = _require_manifest(input_manifest)
    frame = _load_table(standardized_ptm, name="standardized PTM table")
    case_condition = case_condition.strip()
    reference_condition = reference_condition.strip()
    contrast = contrast.strip()
    required = (
        "sample_id",
        "donor_id",
        "condition",
        "protein_id",
        "residue",
        "ptm_type",
        "ptm_value_normalized",
        "value_scale",
        "source_dataset",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KSTARAdapterError(f"standardized PTM table is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise KSTARAdapterError("standardized PTM table is empty")
    if not case_condition.strip() or not reference_condition.strip() or not contrast.strip():
        raise KSTARAdapterError("case_condition, reference_condition, and contrast must be non-empty")
    if case_condition == reference_condition:
        raise KSTARAdapterError("case_condition and reference_condition must be different")
    if not math.isfinite(fold_threshold) or fold_threshold <= 1.0:
        raise KSTARAdapterError("fold_threshold must be finite and > 1")

    for column in ("sample_id", "condition", "protein_id", "residue", "ptm_type", "value_scale", "source_dataset"):
        if frame[column].isna().any():
            raise KSTARAdapterError(f"standardized PTM column {column!r} contains missing values")
    text_types = frame["ptm_type"].astype(str).str.strip().str.lower()
    if set(text_types) != {"phosphorylation"}:
        raise KSTARAdapterError("KSTAR adapter accepts only ptm_type=phosphorylation")
    if (~frame["residue"].astype(str).str.strip().str.upper().str.fullmatch(_RESIDUE_PATTERN)).any():
        raise KSTARAdapterError("residue must be a one-letter S/T/Y site such as S9 or Y123")
    if (~_nonempty(frame, "protein_id")).any() or (~_nonempty(frame, "sample_id")).any():
        raise KSTARAdapterError("sample_id and protein_id must not be empty")

    scales = sorted(set(frame["value_scale"].astype(str).str.strip()))
    if scales not in [["linear"], ["log2"]]:
        raise KSTARAdapterError(f"value_scale must be exactly one of linear or log2; got {scales}")
    value_scale = scales[0]
    normalized = pd.to_numeric(frame["ptm_value_normalized"], errors="coerce")
    donor = frame["donor_id"].fillna("").astype(str).str.strip()
    condition = frame["condition"].astype(str).str.strip()
    donor_mask = donor.ne("")
    state_mask = condition.isin((case_condition, reference_condition))
    finite_mask = normalized.notna() & normalized.map(math.isfinite)
    usable_mask = donor_mask & state_mask & finite_mask
    if not usable_mask.any():
        raise KSTARAdapterError("no donor-level rows with finite normalized values in case/reference conditions")

    usable = frame.loc[usable_mask].copy()
    usable["_donor"] = donor.loc[usable_mask]
    usable["_condition"] = condition.loc[usable_mask]
    usable["_value"] = normalized.loc[usable_mask].astype(float)
    usable["_residue"] = usable["residue"].astype(str).str.strip().str.upper()
    usable["_protein"] = usable["protein_id"].astype(str).str.strip()

    donor_level = usable.groupby(["_protein", "_residue", "_condition", "_donor"], sort=True, as_index=False)[
        "_value"
    ].mean()
    site_values = (
        donor_level.groupby(["_protein", "_residue", "_condition"], sort=True)["_value"].mean().unstack("_condition")
    )
    expected_states = {case_condition, reference_condition}
    observed_sites = set(site_values.index)
    complete = site_values.dropna(subset=[case_condition, reference_condition]).copy()
    incomplete_sites = len(observed_sites - set(complete.index))
    if complete.empty:
        raise KSTARAdapterError("no site has donor-level values in both case and reference conditions")

    case_values = complete[case_condition].astype(float)
    reference_values = complete[reference_condition].astype(float)
    if value_scale == "linear":
        if (reference_values <= 0).any():
            bad = complete.index[reference_values <= 0].tolist()
            raise KSTARAdapterError(f"linear reference values must be > 0 for fold evidence; bad sites: {bad}")
        effect = case_values / reference_values
        up_mask = effect >= fold_threshold
        down_mask = effect <= 1.0 / fold_threshold
    else:
        effect = case_values - reference_values
        delta = math.log2(fold_threshold)
        up_mask = effect >= delta
        down_mask = effect <= -delta

    if (up_mask & down_mask).any():
        raise KSTARAdapterError("a site was classified as both increased and decreased")

    metadata = (
        usable.groupby(["_protein", "_residue", "_condition"], sort=True)
        .agg(
            donors=("_donor", _join_unique),
            samples=("sample_id", _join_unique),
            source_datasets=("source_dataset", _join_unique),
        )
        .reset_index()
    )
    metadata = metadata.set_index(["_protein", "_residue", "_condition"])
    rows: list[dict[str, Any]] = []
    up_col = f"data:{contrast}:increased"
    down_col = f"data:{contrast}:decreased"
    for site, values in complete.iterrows():
        protein, residue = site
        site_up = bool(up_mask.loc[site])
        site_down = bool(down_mask.loc[site])
        evidence_class = "increased" if site_up else "decreased" if site_down else "unchanged"
        case_meta = metadata.loc[(protein, residue, case_condition)]
        reference_meta = metadata.loc[(protein, residue, reference_condition)]
        rows.append(
            {
                "accession": protein,
                "site": residue,
                up_col: int(site_up),
                down_col: int(site_down),
                "case_condition": case_condition,
                "reference_condition": reference_condition,
                "contrast": contrast,
                "value_scale": value_scale,
                "effect_value": float(values[case_condition] / values[reference_condition])
                if value_scale == "linear"
                else float(effect.loc[site]),
                "evidence_class": evidence_class,
                "case_mean": float(values[case_condition]),
                "reference_mean": float(values[reference_condition]),
                "case_donors": case_meta["donors"],
                "reference_donors": reference_meta["donors"],
                "case_samples": case_meta["samples"],
                "reference_samples": reference_meta["samples"],
                "source_datasets": ";".join(
                    sorted(
                        set(str(case_meta["source_datasets"]).split(";"))
                        | set(str(reference_meta["source_datasets"]).split(";"))
                    )
                ),
                "ptm_type": "phosphorylation",
                "input_manifest": str(manifest_path),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise KSTARAdapterError("KSTAR evidence table is empty")
    audit = KSTARInputAudit(
        input_rows=len(frame),
        donor_rows=int(donor_mask.sum()),
        usable_rows=len(usable),
        complete_sites=len(result),
        increased_sites=int(up_mask.sum()),
        decreased_sites=int(down_mask.sum()),
        unchanged_sites=int((~up_mask & ~down_mask).sum()),
        excluded_no_donor_rows=int((~donor_mask).sum()),
        excluded_nonfinite_rows=int((donor_mask & state_mask & ~finite_mask).sum()),
        excluded_incomplete_sites=incomplete_sites,
        case_condition=case_condition,
        reference_condition=reference_condition,
        contrast=contrast,
        value_scale=value_scale,
        fold_threshold=fold_threshold,
        case_donors=tuple(sorted(set(usable.loc[usable["_condition"] == case_condition, "_donor"]))),
        reference_donors=tuple(sorted(set(usable.loc[usable["_condition"] == reference_condition, "_donor"]))),
        source_datasets=tuple(sorted(set(usable["source_dataset"].astype(str).str.strip()))),
    )
    if not expected_states.issubset(set(condition)):  # pragma: no cover - guarded by complete check
        raise KSTARAdapterError("case/reference conditions are absent")
    if len(audit.case_donors) < min_donors_per_state or len(audit.reference_donors) < min_donors_per_state:
        raise KSTARAdapterError(
            f"each condition needs >= {min_donors_per_state} donors with usable evidence; "
            f"case has {len(audit.case_donors)} ({list(audit.case_donors)}), "
            f"reference has {len(audit.reference_donors)} ({list(audit.reference_donors)})"
        )
    return result, audit


def _read_result_table(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return cast(pd.DataFrame, source.copy())
    try:
        path = Path(source).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise KSTARAdapterError(f"KSTAR result does not exist: {source}") from exc
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    try:
        return pd.read_csv(path, sep=sep)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise KSTARAdapterError(f"could not read KSTAR result: {path}: {exc}") from exc


def read_kstar_result(
    source: str | Path | pd.DataFrame,
    *,
    direction: Literal["increased", "decreased"] | None,
    contrast: str,
) -> pd.DataFrame:
    """Read one real KSTAR result and retain its unsigned p-value explicitly.

    KSTAR 1.2.x writes p-values without a sign.  ``direction`` is therefore
    mandatory and is supplied only for a separately run increased/decreased
    evidence analysis; a single unsigned table is rejected.
    """

    if direction not in _DIRECTIONS:
        raise KSTARAdapterError("KSTAR result direction is required: run increased and decreased analyses separately")
    if not contrast.strip():
        raise KSTARAdapterError("contrast must be non-empty")
    frame = _read_result_table(source)
    if frame.index.name == "KSTAR_KINASE" and "KSTAR_KINASE" not in frame.columns:
        frame = frame.reset_index()
    id_candidates = ("KSTAR_KINASE", "kinase", "kinase_id", "regulator_id")
    id_column = next((column for column in id_candidates if column in frame.columns), None)
    if id_column is None:
        raise KSTARAdapterError("KSTAR result is missing KSTAR_KINASE; unsigned output cannot be mapped")
    p_column: str | None = None
    if "activity_pvalue" in frame.columns:
        p_column = "activity_pvalue"
    else:
        data_columns = [column for column in frame.columns if str(column).startswith("data:")]
        matching = [column for column in data_columns if contrast in str(column)]
        if len(matching) == 1:
            p_column = matching[0]
        elif len(matching) > 1:
            raise KSTARAdapterError("KSTAR result has multiple contrast columns; one directional result is required")
        elif len(data_columns) == 1:
            p_column = data_columns[0]
        elif "kinase_activity" in frame.columns and "data" in frame.columns:
            selected = frame[frame["data"].astype(str).str.contains(contrast, regex=False)].copy()
            if selected.empty:
                raise KSTARAdapterError(f"KSTAR result has no rows for contrast {contrast!r}")
            frame = selected
            p_column = "kinase_activity"
    if p_column is None:
        raise KSTARAdapterError(
            "KSTAR output has no explicit directional p-value column; an unsigned result cannot become signed activity"
        )
    identifiers = frame[id_column].astype(str).str.strip()
    if identifiers.eq("").any() or frame[id_column].isna().any():
        raise KSTARAdapterError("KSTAR result contains an empty kinase identifier")
    pvalues = pd.to_numeric(frame[p_column], errors="coerce")
    if pvalues.isna().any() or (~pvalues.map(math.isfinite)).any() or (~pvalues.between(0.0, 1.0)).any():
        raise KSTARAdapterError("KSTAR p-values must be finite numbers in [0, 1]")
    result = pd.DataFrame({"regulator_id": identifiers, "activity_pvalue": pvalues.astype(float)})
    for column in ("n_substrates", "network_coverage"):
        if column in frame.columns:
            result[column] = frame[column].to_numpy()
    if result["regulator_id"].duplicated().any():
        raise KSTARAdapterError("KSTAR result contains duplicate kinase identifiers")
    return result


def _validate_method_version(method_version: str) -> str:
    value = str(method_version).strip()
    if not _VERSION_PATTERN.fullmatch(value) or "smoke-stub" in value.lower():
        raise KSTARAdapterError(f"method_version must be a real KSTAR 1.2.x version, got {method_version!r}")
    return value


def canonicalize_ensembl_id(value: str) -> str:
    """Return a canonical ENSG id without a version suffix."""

    text = str(value).strip()
    if not text:
        raise KSTARAdapterError("Ensembl id must not be empty")
    text = _ENSEMBL_VERSION_PATTERN.sub("", text)
    if not _ENSEMBL_PATTERN.fullmatch(text):
        raise KSTARAdapterError(f"invalid Ensembl gene id: {value!r}")
    return text


def _add_symbol_ensembl(grouped: dict[str, set[str]], symbol: str, ensembl: str) -> None:
    gene = str(symbol).strip().upper()
    if not gene or gene.startswith("ENSG"):
        return
    grouped.setdefault(gene, set()).add(canonicalize_ensembl_id(ensembl))


def load_symbol_to_ensembl_lookup(path: str | Path) -> dict[str, set[str]]:
    """Load gene-symbol to Ensembl candidates from a TSV or pickle mapping."""

    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise KSTARAdapterError(f"Ensembl mapping does not exist: {path}") from exc
    grouped: dict[str, set[str]] = {}
    suffix = resolved.suffix.lower()
    if suffix == ".pkl":
        import pickle

        try:
            payload = pickle.loads(resolved.read_bytes())
        except (OSError, pickle.UnpicklingError, ValueError) as exc:
            raise KSTARAdapterError(f"could not read Ensembl pickle mapping: {resolved}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise KSTARAdapterError("Ensembl pickle mapping must contain a dict")
        for key, value in payload.items():
            left = str(key).strip()
            right = str(value).strip()
            if left.startswith("ENSG") and not right.startswith("ENSG"):
                _add_symbol_ensembl(grouped, right, left)
            elif right.startswith("ENSG") and not left.startswith("ENSG"):
                _add_symbol_ensembl(grouped, left, right)
    else:
        try:
            frame = pd.read_csv(resolved, sep="\t", dtype=str, keep_default_na=False)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise KSTARAdapterError(f"could not read Ensembl mapping: {resolved}: {exc}") from exc
        columns = {str(column).strip().lower(): column for column in frame.columns}
        symbol_column = next((columns[name] for name in ("gene_symbol", "symbol", "gene") if name in columns), None)
        ensembl_column = next(
            (columns[name] for name in ("ensembl_id", "ensembl", "gene_id") if name in columns),
            None,
        )
        if symbol_column is None or ensembl_column is None:
            raise KSTARAdapterError(
                "Ensembl mapping TSV must contain gene_symbol and ensembl_id columns "
                f"(or symbol/ensembl aliases); got {list(frame.columns)!r}"
            )
        for symbol, ensembl in zip(frame[symbol_column], frame[ensembl_column], strict=True):
            _add_symbol_ensembl(grouped, symbol, ensembl)
    if not grouped:
        raise KSTARAdapterError(f"Ensembl mapping contains no symbol-to-ENSG pairs: {resolved}")
    return grouped


def map_gene_symbols_to_ensembl(symbols: Iterable[str], mapping: str | Path) -> dict[str, str]:
    """Map requested gene symbols to canonical Ensembl ids.

    Unused mapping collisions are ignored.  A requested kinase with zero or
    more than one distinct Ensembl id hard-fails; no alias guessing is allowed.
    """

    requested = [str(symbol).strip() for symbol in symbols]
    if any(not symbol for symbol in requested):
        raise KSTARAdapterError("kinase symbol must not be empty")
    lookup = load_symbol_to_ensembl_lookup(mapping)
    mapped: dict[str, str] = {}
    missing: list[str] = []
    conflicts: dict[str, tuple[str, ...]] = {}
    for symbol in requested:
        key = symbol.upper()
        if key in mapped or symbol in mapped:
            continue
        hits = sorted(lookup.get(key, ()))
        if not hits:
            missing.append(symbol)
        elif len(hits) > 1:
            conflicts[symbol] = tuple(hits)
        else:
            mapped[symbol] = hits[0]
            mapped[key] = hits[0]
    if missing or conflicts:
        raise KSTARAdapterError(
            "KSTAR kinase symbols could not be mapped to a unique canonical Ensembl id; "
            f"missing={missing}; conflicts={conflicts}"
        )
    return mapped


def _bh_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 1.0
    size = len(values)
    for rank in range(size - 1, -1, -1):
        index = order[rank]
        running = min(running, values[index] * size / (rank + 1))
        adjusted[index] = min(1.0, running)
    return adjusted


def _load_metrics(metrics: str | Path | pd.DataFrame | None) -> pd.DataFrame | None:
    if metrics is None:
        return None
    frame = _load_table(metrics, name="KSTAR metrics") if not isinstance(metrics, pd.DataFrame) else metrics.copy()
    required = {"regulator_id", "n_substrates", "network_coverage"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KSTARAdapterError(f"KSTAR metrics are missing columns: {', '.join(missing)}")
    result = frame[["regulator_id", "n_substrates", "network_coverage"]].copy()
    result["regulator_id"] = result["regulator_id"].astype(str).str.strip()
    if result["regulator_id"].eq("").any() or result["regulator_id"].duplicated().any():
        raise KSTARAdapterError("KSTAR metrics must have one non-empty row per kinase")
    result["n_substrates"] = pd.to_numeric(result["n_substrates"], errors="coerce")
    result["network_coverage"] = pd.to_numeric(result["network_coverage"], errors="coerce")
    if result.isna().any().any() or (~result["n_substrates"].map(math.isfinite)).any():
        raise KSTARAdapterError("KSTAR metrics must be finite")
    if (result["n_substrates"] < 0).any() or (~result["network_coverage"].between(0.0, 1.0)).any():
        raise KSTARAdapterError("KSTAR metrics are out of range")
    if (result["n_substrates"] % 1 != 0).any():
        raise KSTARAdapterError("n_substrates must be integer-valued")
    result["n_substrates"] = result["n_substrates"].astype(int)
    return cast(pd.DataFrame, result)


def convert_kstar_outputs(
    increased_output: str | Path | pd.DataFrame,
    decreased_output: str | Path | pd.DataFrame,
    *,
    contrast: str,
    method_version: str,
    input_manifest: str | Path,
    increased_metrics: str | Path | pd.DataFrame | None = None,
    decreased_metrics: str | Path | pd.DataFrame | None = None,
    ensembl_mapping: str | Path | None = None,
) -> pd.DataFrame:
    """Convert paired KSTAR p-value outputs into signed standard activity rows.

    ``increased_metrics``/``decreased_metrics`` carry the direction-specific
    ``n_substrates``/``network_coverage`` computed from that direction's
    evidence set; both must be supplied together.  Each kinase row binds the
    metrics of its *winning* direction (the analysis with the smaller
    p-value), so legitimate differences between the two directions are
    preserved instead of being rejected.
    """

    if (increased_metrics is None) != (decreased_metrics is None):
        raise KSTARAdapterError("increased_metrics and decreased_metrics must be supplied together")

    manifest_path, _ = _require_manifest(input_manifest)
    version = _validate_method_version(method_version)
    increased = read_kstar_result(increased_output, direction="increased", contrast=contrast)
    decreased = read_kstar_result(decreased_output, direction="decreased", contrast=contrast)
    if set(increased["regulator_id"]) != set(decreased["regulator_id"]):
        raise KSTARAdapterError("increased and decreased KSTAR outputs have different kinase sets")

    result_ids = set(increased["regulator_id"])
    metrics_by_direction: dict[str, pd.DataFrame] = {}
    if increased_metrics is not None and decreased_metrics is not None:
        increased_frame = _load_metrics(increased_metrics)
        decreased_frame = _load_metrics(decreased_metrics)
        assert increased_frame is not None and decreased_frame is not None
        if set(increased_frame["regulator_id"]) != set(decreased_frame["regulator_id"]):
            raise KSTARAdapterError("directional KSTAR metrics cover different kinase sets")
        metrics_by_direction = {"increased": increased_frame, "decreased": decreased_frame}
        for direction, frame in metrics_by_direction.items():
            if not result_ids.issubset(set(frame["regulator_id"])):
                missing = sorted(result_ids - set(frame["regulator_id"]))
                raise KSTARAdapterError(f"{direction} KSTAR metrics are missing kinases present in results: {missing}")
    else:
        parts = [part for part in (increased, decreased) if {"n_substrates", "network_coverage"}.issubset(part.columns)]
        if len(parts) != 2:
            raise KSTARAdapterError(
                "direction-specific n_substrates/network_coverage metrics are required; "
                "KSTAR p-value outputs do not provide them"
            )
        fallback = parts[0][["regulator_id", "n_substrates", "network_coverage"]].copy()
        if not fallback.equals(parts[1][["regulator_id", "n_substrates", "network_coverage"]]):
            raise KSTARAdapterError("directional KSTAR outputs disagree on substrate metrics")
        loaded = _load_metrics(fallback)
        assert loaded is not None
        metrics_by_direction = {"increased": loaded, "decreased": loaded}

    merged = increased.merge(decreased, on="regulator_id", suffixes=("_increased", "_decreased"), validate="one_to_one")
    raw_pvalues = merged[["activity_pvalue_increased", "activity_pvalue_decreased"]].to_numpy().reshape(-1).tolist()
    qvalues = _bh_adjust([float(value) for value in raw_pvalues])
    q_frame = pd.DataFrame(
        {
            "regulator_id": merged["regulator_id"].repeat(2).to_numpy(),
            "direction": [direction for _ in range(len(merged)) for direction in _DIRECTIONS],
            "qvalue": qvalues,
        }
    )
    q_lookup = {
        (row.regulator_id, row.direction): float(cast(float, row.qvalue)) for row in q_frame.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        inc_p = float(row.activity_pvalue_increased)
        dec_p = float(row.activity_pvalue_decreased)
        if inc_p == 1.0 and dec_p == 1.0:
            continue
        if inc_p < 1.0 and dec_p < 1.0 and math.isclose(inc_p, dec_p, rel_tol=0.0, abs_tol=1e-15):
            raise KSTARAdapterError(f"kinase {row.regulator_id!r} has ambiguous increased/decreased evidence")
        direction = "increased" if inc_p < dec_p else "decreased"
        pvalue = inc_p if direction == "increased" else dec_p
        sign = 1.0 if direction == "increased" else -1.0
        score = sign * -math.log10(max(pvalue, math.nextafter(0.0, 1.0)))
        if not math.isfinite(score) or score == 0.0:
            raise KSTARAdapterError(f"kinase {row.regulator_id!r} did not produce a finite non-zero signed score")
        winning_metrics = metrics_by_direction[direction]
        metric_row = winning_metrics.loc[winning_metrics["regulator_id"] == row.regulator_id]
        if metric_row.empty:
            raise KSTARAdapterError(f"{direction} KSTAR metrics have no row for kinase {row.regulator_id!r}")
        rows.append(
            {
                "activity_unit": "signed_-log10_kstar_pvalue",
                "regulator_id": row.regulator_id,
                "regulator_type": "kinase",
                "condition_or_contrast": contrast,
                "activity_score": score,
                "activity_direction": "up" if direction == "increased" else "down",
                "activity_pvalue": pvalue,
                "activity_qvalue": q_lookup[(row.regulator_id, direction)],
                "n_substrates": int(metric_row["n_substrates"].iloc[0]),
                "network_coverage": float(metric_row["network_coverage"].iloc[0]),
                "method": KSTAR_METHOD,
                "method_version": version,
                "input_manifest": str(manifest_path),
            }
        )
    if not rows:
        raise KSTARAdapterError("paired KSTAR outputs contain no directional kinase evidence")
    activity = pd.DataFrame(rows, columns=list(ACTIVITY_REQUIRED_COLUMNS))
    if ensembl_mapping is not None:
        mapped = map_gene_symbols_to_ensembl(activity["regulator_id"].astype(str), ensembl_mapping)
        activity["regulator_id"] = activity["regulator_id"].astype(str).map(lambda symbol: mapped[str(symbol)])
        duplicated = activity["regulator_id"].duplicated(keep=False)
        if duplicated.any():
            raise KSTARAdapterError(
                "Ensembl mapping collapsed distinct KSTAR kinases onto the same gene: "
                f"{sorted(set(activity.loc[duplicated, 'regulator_id'].astype(str)))}"
            )
    return cast(pd.DataFrame, activity.sort_values("regulator_id").reset_index(drop=True))


def write_adapter_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write an adapter manifest without promoting the result to biology PASS."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    content = dict(payload)
    content.setdefault("schema_version", KSTAR_ADAPTER_SCHEMA_VERSION)
    content["lineage"] = {
        "lineage_boundary": content.get("lineage_boundary", "kstar_engineering"),
        "biology_pass": False,
        "may_enter_lineage": False,
        "formal_biology": False,
        "reason": "real PTM and registered kinase benchmark are still required",
    }
    resolved.write_text(json.dumps(content, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return resolved


__all__ = [
    "ACTIVITY_REQUIRED_COLUMNS",
    "KSTAR_ADAPTER_SCHEMA_VERSION",
    "KSTARInputAudit",
    "KSTARAdapterError",
    "build_kstar_input",
    "canonicalize_ensembl_id",
    "convert_kstar_outputs",
    "load_symbol_to_ensembl_lookup",
    "map_gene_symbols_to_ensembl",
    "read_kstar_result",
    "write_adapter_manifest",
]
