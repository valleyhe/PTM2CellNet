#!/usr/bin/env python3
"""Assemble a new signed kinase-to-TF/gene network release.

The kinase table contains only intermediate edges (for example
``kinase_substrate:*``); the TF table contains only ``tf_regulation``
gene-terminating edges.  Both inputs already use the nine-column signed
network contract.  This command never edits an input table, including the
frozen 2026-09-16 TF-only release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.signed_network import (  # noqa: E402
    SIGNED_NETWORK_REQUIRED_COLUMNS,
    SignedNetworkContractError,
    load_signed_network,
)
from src.models.gene_vocabulary import normalize_ensembl_id  # noqa: E402


DEFAULT_SPECIES = "9606"
TF_EDGE_TYPE = "tf_regulation"
TF_ONLY_RELEASE = "omnipath-2026-09-16"


class NetworkReleaseAssemblyError(ValueError):
    """Raised when an input or release violates the assembly contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kinase-network-tsv", type=Path, required=True)
    parser.add_argument("--tf-network-tsv", type=Path, required=True)
    parser.add_argument("--release", required=True, help="new frozen release identifier")
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--species", default=DEFAULT_SPECIES, help="NCBI taxonomy id (default: 9606)")
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_effect_sign(value: str, *, label: str, row_number: int) -> str:
    text = str(value).strip()
    if text in {"", "unsigned"}:
        return ""
    if text in {"+1", "+1.0", "1", "1.0", "+", "activation"}:
        return "+1"
    if text in {"-1", "-1.0", "-", "inhibition", "repression"}:
        return "-1"
    raise NetworkReleaseAssemblyError(
        f"{label} row {row_number}: effect_sign must be +1, -1, 1, or empty; got {value!r}"
    )


def _resolve_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise NetworkReleaseAssemblyError(f"{label} file does not exist: {path}") from exc
    if not resolved.is_file():
        raise NetworkReleaseAssemblyError(f"{label} is not a file: {path}")
    return resolved


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in SIGNED_NETWORK_REQUIRED_COLUMNS:
        frame[column] = frame[column].map(lambda value: str(value).strip())
    return frame


def _validate_ids(frame: pd.DataFrame, label: str) -> None:
    for column in ("source_id", "target_id"):
        for row_number, value in enumerate(frame[column], start=2):
            if not value:
                raise NetworkReleaseAssemblyError(f"{label} row {row_number}: {column} must not be empty")
            try:
                canonical = normalize_ensembl_id(value)
            except ValueError as exc:
                raise NetworkReleaseAssemblyError(
                    f"{label} row {row_number}: {column} must be a canonical Ensembl gene id, got {value!r}"
                ) from exc
            if canonical != value:
                raise NetworkReleaseAssemblyError(
                    f"{label} row {row_number}: {column} must be canonical without a version suffix, got {value!r}"
                )


def _validate_common(frame: pd.DataFrame, label: str, species: str) -> tuple[str, tuple[str, ...]]:
    if tuple(frame.columns) != SIGNED_NETWORK_REQUIRED_COLUMNS:
        raise NetworkReleaseAssemblyError(
            f"{label} must contain exactly the nine signed-network columns in canonical order; "
            f"got {list(frame.columns)!r}"
        )
    if frame.empty:
        raise NetworkReleaseAssemblyError(f"{label} has no data rows")
    frame = _clean_frame(frame)
    _validate_ids(frame, label)
    for column in ("edge_type", "species", "evidence", "release"):
        if (frame[column] == "").any():
            raise NetworkReleaseAssemblyError(f"{label} column {column!r} must not contain empty values")

    species_values = tuple(sorted(set(frame["species"])))
    if species_values != (species,):
        raise NetworkReleaseAssemblyError(
            f"{label} must contain one species matching {species!r}; got {species_values!r}"
        )
    release_values = tuple(sorted(set(frame["release"])))
    if len(release_values) != 1:
        raise NetworkReleaseAssemblyError(f"{label} must contain exactly one input release; got {release_values!r}")

    frame["effect_sign"] = [
        _canonical_effect_sign(value, label=label, row_number=index + 2)
        for index, value in enumerate(frame["effect_sign"])
    ]
    for row_number, value in enumerate(frame["confidence"], start=2):
        if not value:
            continue
        try:
            confidence = float(value)
        except ValueError as exc:
            raise NetworkReleaseAssemblyError(
                f"{label} row {row_number}: confidence must be numeric in (0, 1], got {value!r}"
            ) from exc
        if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
            raise NetworkReleaseAssemblyError(
                f"{label} row {row_number}: confidence must be numeric in (0, 1], got {value!r}"
            )
    return release_values[0], tuple(sorted(set(frame["edge_type"])))


def _load_table(path: Path, label: str, species: str, *, kinase: bool) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    resolved = _resolve_file(path, label)
    try:
        frame = pd.read_csv(resolved, sep="\t", dtype=str, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise NetworkReleaseAssemblyError(f"unable to read {label}: {resolved}: {exc}") from exc
    release, edge_types = _validate_common(frame, label, species)
    if kinase and any(
        edge_type == TF_EDGE_TYPE or edge_type.startswith(f"{TF_EDGE_TYPE}:") for edge_type in edge_types
    ):
        raise NetworkReleaseAssemblyError(
            f"{label} must contain intermediate edges only; {TF_EDGE_TYPE!r} is reserved for the TF table"
        )
    if not kinase and edge_types != (TF_EDGE_TYPE,):
        raise NetworkReleaseAssemblyError(
            f"{label} must contain only the terminal edge type {TF_EDGE_TYPE!r}; got {edge_types!r}"
        )
    return _clean_frame(frame), release, edge_types


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assemble_network_release(
    *,
    kinase_network_tsv: Path,
    tf_network_tsv: Path,
    release: str,
    output_tsv: Path,
    manifest_output: Path,
    species: str = DEFAULT_SPECIES,
) -> dict[str, Any]:
    species = str(species).strip()
    release = str(release).strip()
    if not species:
        raise NetworkReleaseAssemblyError("species must not be empty")
    if not release:
        raise NetworkReleaseAssemblyError("release must not be empty")
    if any(character in release for character in "\t\r\n"):
        raise NetworkReleaseAssemblyError("release must not contain tab or newline characters")

    kinase_path = _resolve_file(kinase_network_tsv, "kinase network")
    tf_path = _resolve_file(tf_network_tsv, "TF network")
    output_path = output_tsv.expanduser().resolve()
    manifest_path = manifest_output.expanduser().resolve()
    if output_path in {kinase_path, tf_path} or manifest_path in {kinase_path, tf_path}:
        raise NetworkReleaseAssemblyError("output paths must not overwrite either input network")
    if output_path == manifest_path:
        raise NetworkReleaseAssemblyError("output-tsv and manifest-output must be different paths")

    kinase_frame, kinase_release, kinase_edge_types = _load_table(kinase_path, "kinase network", species, kinase=True)
    tf_frame, tf_release, tf_edge_types = _load_table(tf_path, "TF network", species, kinase=False)
    input_releases = {kinase_release, tf_release}
    if release in input_releases or release == TF_ONLY_RELEASE:
        raise NetworkReleaseAssemblyError(
            f"release {release!r} must be a new release and cannot reuse an input/TF-only release"
        )

    merged = pd.concat([kinase_frame, tf_frame], ignore_index=True)
    merged = merged.drop_duplicates(keep="first")
    merged["release"] = release
    merged = merged.sort_values(list(SIGNED_NETWORK_REQUIRED_COLUMNS), kind="mergesort").reset_index(drop=True)
    if merged.empty:
        raise NetworkReleaseAssemblyError("assembled output has no rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, sep="\t", index=False)
    try:
        load_signed_network(output_path, expected_species=species, expected_release=release)
    except SignedNetworkContractError as exc:
        raise NetworkReleaseAssemblyError(f"assembled output failed signed-network validation: {exc}") from exc

    manifest: dict[str, object] = {
        "schema_version": "ptm2cellnet.signed-network-release/v1",
        "release": release,
        "species": species,
        "biology_pass": False,
        "inputs": {
            "kinase_network": {
                "path": str(kinase_path),
                "release": kinase_release,
                "edge_types": list(kinase_edge_types),
                "rows": len(kinase_frame),
                "sha256": _sha256_file(kinase_path),
            },
            "tf_network": {
                "path": str(tf_path),
                "release": tf_release,
                "edge_types": list(tf_edge_types),
                "rows": len(tf_frame),
                "sha256": _sha256_file(tf_path),
            },
        },
        "counts": {
            "input_rows": len(kinase_frame) + len(tf_frame),
            "duplicate_rows_removed": len(kinase_frame) + len(tf_frame) - len(merged),
            "output_rows": len(merged),
        },
        "output": {
            "path": str(output_path),
            "release": release,
            "rows": len(merged),
            "sha256": _sha256_file(output_path),
        },
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = assemble_network_release(
            kinase_network_tsv=args.kinase_network_tsv,
            tf_network_tsv=args.tf_network_tsv,
            release=args.release,
            output_tsv=args.output_tsv,
            manifest_output=args.manifest_output,
            species=args.species,
        )
    except (NetworkReleaseAssemblyError, OSError) as exc:
        print(f"[build_kinase_tf_network_release] 输入契约失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "output": manifest["output"], "rows": manifest["counts"]["output_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
