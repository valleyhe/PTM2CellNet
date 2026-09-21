#!/usr/bin/env python3
"""Build a signed kinase/signaling edge table from an OmniPath interactions export.

The output is an intermediate-edge table for ``build_kinase_tf_network_release.py``.
It never writes ``tf_regulation`` rows and never edits the 2026-09-16 TF-only freeze.

Sign and confidence rules match ``scripts/build_signed_network_release.py``:
consensus stimulation/inhibition first, then is_stimulation/is_inhibition;
``confidence = min(1.0, 0.5 + 0.1 * (n_unique_resources - 1))``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_signed_network_release import SIGNED_NETWORK_COLUMNS, _edge_sign  # noqa: E402
from src.analysis.kstar_adapter import canonicalize_ensembl_id, load_symbol_to_ensembl_lookup  # noqa: E402
from src.models.gene_vocabulary import normalize_ensembl_id  # noqa: E402

DEFAULT_SPECIES = "9606"
DEFAULT_EDGE_TYPE = "kinase_substrate:signaling"
TF_EDGE_TYPE = "tf_regulation"
OMNIPATH_QUERY = (
    "https://omnipathdb.org/interactions?datasets=omnipath,kinaseextra"
    "&genesymbols=1&fields=sources,evidences&organisms=9606"
)


class KinaseSignedEdgeError(ValueError):
    """Raised when kinase signed-edge construction violates its contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--omnipath-tsv", type=Path, required=True)
    parser.add_argument("--ensembl-mapping", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--edge-type", default=DEFAULT_EDGE_TYPE)
    parser.add_argument("--species", default=DEFAULT_SPECIES)
    parser.add_argument(
        "--enzyme-source-tsv",
        type=Path,
        help="optional TSV whose source_gene column whitelists enzyme/kinase sources",
    )
    parser.add_argument("--enzyme-source-column", default="source_gene")
    parser.add_argument(
        "--forbid-output-paths",
        type=Path,
        nargs="*",
        default=(),
        help="paths that this command must not write, including the TF-only freeze",
    )
    return parser.parse_args(argv)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise KinaseSignedEdgeError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise KinaseSignedEdgeError(f"{label} is not a file: {path}")
    return resolved


def _resolve_symbol(lookup: dict[str, set[str]], symbol: str) -> str | None:
    hits = lookup.get(str(symbol).strip().upper(), set())
    if len(hits) != 1:
        return None
    return canonicalize_ensembl_id(next(iter(hits)))


def _load_enzyme_sources(path: Path, column: str, lookup: dict[str, set[str]]) -> set[str]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if column not in frame.columns:
        raise KinaseSignedEdgeError(f"enzyme source table is missing column {column!r}")
    resolved: set[str] = set()
    for value in frame[column]:
        ensembl = _resolve_symbol(lookup, str(value))
        if ensembl is not None:
            resolved.add(ensembl)
    if not resolved:
        raise KinaseSignedEdgeError("no enzyme sources mapped to Ensembl")
    return resolved


def assemble_kinase_signed_edges(
    *,
    omnipath_tsv: Path,
    ensembl_mapping: Path,
    release: str,
    output_tsv: Path,
    manifest_output: Path,
    edge_type: str = DEFAULT_EDGE_TYPE,
    species: str = DEFAULT_SPECIES,
    enzyme_source_tsv: Path | None = None,
    enzyme_source_column: str = "source_gene",
    forbid_output_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    release = str(release).strip()
    edge_type = str(edge_type).strip()
    species = str(species).strip()
    if not release or not edge_type or not species:
        raise KinaseSignedEdgeError("release, edge_type, and species must not be empty")
    if edge_type == TF_EDGE_TYPE or edge_type.startswith(f"{TF_EDGE_TYPE}:"):
        raise KinaseSignedEdgeError(f"kinase edges cannot use the terminal type {TF_EDGE_TYPE!r}")
    if release == "omnipath-2026-09-16":
        raise KinaseSignedEdgeError("kinase release must be a new id; cannot reuse omnipath-2026-09-16")

    input_path = _resolve_file(omnipath_tsv, "OmniPath interactions")
    mapping_path = _resolve_file(ensembl_mapping, "Ensembl mapping")
    output_path = output_tsv.expanduser().resolve()
    manifest_path = manifest_output.expanduser().resolve()
    forbidden = {_resolve_file(path, "forbidden output") for path in forbid_output_paths}
    if output_path in forbidden or manifest_path in forbidden or output_path == input_path:
        raise KinaseSignedEdgeError("output paths must not overwrite the OmniPath input or a forbidden freeze file")

    frame = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
    if frame.empty:
        raise KinaseSignedEdgeError("OmniPath interactions table is empty")

    lookup = load_symbol_to_ensembl_lookup(mapping_path)
    enzyme_sources: set[str] | None = None
    if enzyme_source_tsv is not None:
        enzyme_sources = _load_enzyme_sources(
            _resolve_file(enzyme_source_tsv, "enzyme source"), enzyme_source_column, lookup
        )

    n_input = len(frame)
    n_undirected = 0
    n_unsigned = 0
    n_unmapped = 0
    n_self_loops = 0
    n_non_enzyme_source = 0
    rows: list[dict[str, object]] = []
    for record in frame.to_dict("records"):
        if "is_directed" in record and not _truthy(record.get("is_directed")):
            n_undirected += 1
            continue
        sign = _edge_sign(record)
        if sign is None:
            n_unsigned += 1
            continue
        source_symbol = str(record.get("source_genesymbol") or record.get("source", "")).strip()
        target_symbol = str(record.get("target_genesymbol") or record.get("target", "")).strip()
        source_id = _resolve_symbol(lookup, source_symbol)
        target_id = _resolve_symbol(lookup, target_symbol)
        if source_id is None or target_id is None:
            n_unmapped += 1
            continue
        source_id = normalize_ensembl_id(source_id)
        target_id = normalize_ensembl_id(target_id)
        if source_id == target_id:
            n_self_loops += 1
            continue
        if enzyme_sources is not None and source_id not in enzyme_sources:
            n_non_enzyme_source += 1
            continue
        resources = sorted({part.strip() for part in str(record.get("sources", "")).split(";") if part.strip()})
        if not resources:
            n_unsigned += 1
            continue
        confidence = min(1.0, 0.5 + 0.1 * (len(resources) - 1))
        rows.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
                "effect_sign": "+1" if sign > 0 else "-1",
                "site": "",
                "species": species,
                "evidence": ";".join(resources),
                "confidence": f"{confidence:.3f}",
                "release": release,
            }
        )

    output = pd.DataFrame(rows, columns=list(SIGNED_NETWORK_COLUMNS))
    if output.empty:
        raise KinaseSignedEdgeError("no signed kinase/signaling edges remained after filtering")
    output = output.sort_values(
        ["source_id", "target_id", "edge_type", "confidence"], ascending=[True, True, True, False]
    ).drop_duplicates(subset=["source_id", "target_id", "edge_type", "effect_sign"], keep="first")
    if (output["edge_type"] == TF_EDGE_TYPE).any():
        raise KinaseSignedEdgeError("kinase output unexpectedly contains tf_regulation rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep="\t", index=False)
    manifest = {
        "schema_version": "ptm2cellnet.signed-network-release/v1",
        "release": release,
        "biology_pass": False,
        "edge_type": edge_type,
        "species": species,
        "source": {
            "omnipath_export": str(input_path),
            "omnipath_sha256": _sha256_file(input_path),
            "query": OMNIPATH_QUERY,
            "ensembl_mapping": str(mapping_path),
            "enzyme_source_tsv": str(enzyme_source_tsv.expanduser().resolve()) if enzyme_source_tsv else None,
        },
        "sign_rule": "consensus_(stimulation|inhibition) first, then is_(stimulation|inhibition); both/neither -> dropped",
        "confidence_formula": "min(1.0, 0.5 + 0.1 * (n_unique_resources - 1))",
        "counts": {
            "input_rows": n_input,
            "undirected_dropped": n_undirected,
            "unsigned_or_conflicting_dropped": n_unsigned,
            "unmapped_dropped": n_unmapped,
            "self_loops_dropped": n_self_loops,
            "non_enzyme_source_dropped": n_non_enzyme_source,
            "released_edges": len(output),
            "activating": int((output["effect_sign"] == "+1").sum()),
            "inhibiting": int((output["effect_sign"] == "-1").sum()),
        },
        "output": {"path": str(output_path), "sha256": _sha256_file(output_path), "rows": len(output)},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = assemble_kinase_signed_edges(
            omnipath_tsv=args.omnipath_tsv,
            ensembl_mapping=args.ensembl_mapping,
            release=args.release,
            output_tsv=args.output_tsv,
            manifest_output=args.manifest_output,
            edge_type=args.edge_type,
            species=args.species,
            enzyme_source_tsv=args.enzyme_source_tsv,
            enzyme_source_column=args.enzyme_source_column,
            forbid_output_paths=args.forbid_output_paths,
        )
    except (KinaseSignedEdgeError, OSError) as exc:
        print(f"[build_kinase_signed_edges] 输入契约失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "output": manifest["output"], "rows": manifest["counts"]["released_edges"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
