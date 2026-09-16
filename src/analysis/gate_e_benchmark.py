"""Assemble the Gate-E vocabulary-migration benchmark from real PTM databases.

The Gate-E evaluator (``src/integration/perturbgen/gate_e.py``) consumes a
frozen benchmark CSV of ``(gene_symbol, ensembl_id, ptm_type[, position])``
rows with at least 200 samples. Every row here is a real annotated human PTM
site: phosphorylation comes from dbptm (UniProt-anchored) and the remaining
PTM types come from CPLM, which carries gene symbols directly. UniProt→symbol
resolution reuses the CPLM UniProt/symbol pairs; symbol→Ensembl uses the
PerturbGen ensembl mapping dict. The benchmark measures gene coverage, token
collisions and PTM action-code invariance across vocabulary migrations — it
is not a direction-prediction benchmark and carries no direction labels.
"""

from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.models.gene_vocabulary import normalize_ensembl_id

GATE_E_BENCHMARK_SCHEMA_VERSION = "ptm2cellnet.gate-e-benchmark/v1"
BENCHMARK_COLUMNS = ("gene_symbol", "ensembl_id", "ptm_type", "position")
_ENSG_PATTERN = re.compile(r"^ENSG\d{11}$")


class GateEBenchmarkBuildError(ValueError):
    """Raised when benchmark assembly inputs violate the contract."""


def _load_ensembl_mapping(path: str | Path) -> dict[str, str]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise GateEBenchmarkBuildError(f"ensembl mapping pickle must be a dict: {path}")
    mapping: dict[str, str] = {}
    for symbol, ensembl_id in payload.items():
        text_symbol = str(symbol).strip().upper()
        text_id = str(ensembl_id).strip()
        if not text_symbol or not _ENSG_PATTERN.match(text_id):
            continue
        mapping[text_symbol] = text_id
    if not mapping:
        raise GateEBenchmarkBuildError(f"ensembl mapping pickle yielded no usable entries: {path}")
    return mapping


def _load_cplm_sites(path: str | Path) -> tuple[list[tuple[str, int, str, str]], dict[str, str]]:
    """Return CPLM human sites and the UniProt→symbol pairs it carries."""

    sites: list[tuple[str, int, str, str]] = []
    acc_to_symbol: dict[str, str] = {}
    conflicts = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 6:
                continue
            _, uniprot_acc, position_text, ptm_type, gene_symbol, species = row[:6]
            if species.strip() != "Homo sapiens":
                continue
            symbol = gene_symbol.strip().upper()
            if not symbol or not uniprot_acc.strip():
                continue
            previous = acc_to_symbol.get(uniprot_acc)
            if previous is not None and previous != symbol:
                conflicts += 1
                continue
            acc_to_symbol[uniprot_acc] = symbol
            try:
                position = int(position_text)
            except ValueError:
                continue
            sites.append((symbol, position, ptm_type.strip().lower(), uniprot_acc))
    if conflicts:
        raise GateEBenchmarkBuildError(
            f"CPLM file carries conflicting UniProt→symbol pairs for {conflicts} accessions; resolve the input first"
        )
    return sites, acc_to_symbol


def _load_dbptm_phosphorylation(path: str | Path, acc_to_symbol: Mapping[str, str]) -> list[tuple[str, int, str, str]]:
    sites: list[tuple[str, int, str, str]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 4:
                continue
            _, uniprot_acc, position_text, ptm_type = row[:4]
            if ptm_type.strip().lower() != "phosphorylation":
                continue
            symbol = acc_to_symbol.get(uniprot_acc.strip())
            if symbol is None:
                continue
            try:
                position = int(position_text)
            except ValueError:
                continue
            sites.append((symbol, position, "phosphorylation", uniprot_acc))
    return sites


def build_gate_e_benchmark(
    *,
    cplm_path: str | Path,
    dbptm_phosphorylation_path: str | Path | None,
    ensembl_mapping_path: str | Path,
    min_rows: int = 200,
) -> tuple[pd.DataFrame, dict]:
    """Build the deduplicated benchmark frame and its provenance manifest."""

    ensembl_mapping = _load_ensembl_mapping(ensembl_mapping_path)
    cplm_sites, acc_to_symbol = _load_cplm_sites(cplm_path)
    phosphorylation_sites: list[tuple[str, int, str, str]] = []
    dbptm_input_rows = 0
    dbptm_unmapped_acc = 0
    if dbptm_phosphorylation_path is not None:
        raw_sites: list[tuple[str, int, str, str]] = []
        with Path(dbptm_phosphorylation_path).open("r", encoding="utf-8", errors="replace") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) >= 4:
                    dbptm_input_rows += 1
        phosphorylation_sites = _load_dbptm_phosphorylation(dbptm_phosphorylation_path, acc_to_symbol)
        dbptm_unmapped_acc = dbptm_input_rows - len(phosphorylation_sites)

    all_sites = cplm_sites + phosphorylation_sites
    unmapped_symbol = 0
    invalid_ensembl = 0
    deduplicated: dict[tuple[str, str, str], int] = {}
    for symbol, position, ptm_type, _uniprot_acc in all_sites:
        ensembl_id = ensembl_mapping.get(symbol)
        if ensembl_id is None:
            unmapped_symbol += 1
            continue
        try:
            canonical = normalize_ensembl_id(ensembl_id)
        except ValueError:
            invalid_ensembl += 1
            continue
        key = (symbol, canonical, ptm_type)
        deduplicated.setdefault(key, position)

    records = [
        {
            "gene_symbol": symbol,
            "ensembl_id": ensembl_id,
            "ptm_type": ptm_type,
            "position": position,
        }
        for (symbol, ensembl_id, ptm_type), position in deduplicated.items()
    ]
    records.sort(key=lambda record: (record["gene_symbol"], record["ptm_type"], record["ensembl_id"]))
    frame = pd.DataFrame(records, columns=list(BENCHMARK_COLUMNS))

    if len(frame) < min_rows:
        raise GateEBenchmarkBuildError(
            f"assembled benchmark has {len(frame)} rows; Gate-E requires at least {min_rows}"
        )

    manifest = {
        "schema_version": GATE_E_BENCHMARK_SCHEMA_VERSION,
        "sources": {
            "cplm": str(Path(cplm_path)),
            "dbptm_phosphorylation": None
            if dbptm_phosphorylation_path is None
            else str(Path(dbptm_phosphorylation_path)),
            "ensembl_mapping": str(Path(ensembl_mapping_path)),
        },
        "cplm_human_site_rows": len(cplm_sites),
        "cplm_uniprot_to_symbol_pairs": len(acc_to_symbol),
        "dbptm_phosphorylation_rows": dbptm_input_rows,
        "dbptm_phosphorylation_unmapped_accessions": dbptm_unmapped_acc,
        "unmapped_symbols": unmapped_symbol,
        "invalid_ensembl_ids": invalid_ensembl,
        "benchmark_rows": len(frame),
        "ptm_type_counts": frame["ptm_type"].value_counts().to_dict(),
        "min_rows": min_rows,
        "usage": (
            "Gate-E vocabulary-migration benchmark: gene coverage, token collisions and PTM action-code "
            "invariance; not a direction-prediction benchmark"
        ),
    }
    return frame, manifest


def write_gate_e_benchmark(
    frame: pd.DataFrame, manifest: dict, output_csv: str | Path, manifest_output: str | Path
) -> None:
    output_path = Path(output_csv).expanduser().resolve()
    manifest_path = Path(manifest_output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    payload = dict(manifest)
    payload["output_csv"] = str(output_path)
    import json

    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "BENCHMARK_COLUMNS",
    "GATE_E_BENCHMARK_SCHEMA_VERSION",
    "GateEBenchmarkBuildError",
    "build_gate_e_benchmark",
    "write_gate_e_benchmark",
]
