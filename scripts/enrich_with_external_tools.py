#!/usr/bin/env python
"""Enrich a protein of interest with external bioinformatics tool outputs.

This script bridges ``src/models/external_tools.py`` (AlphaFold / BLAST /
ClustalW / PSIPRED clients, already implemented) into the ``scripts/`` layer
so researchers can pull structure / homolog / secondary-structure annotations
from the command line without writing Python.

The script is intentionally network-tolerant: each tool falls back to a local
built-in implementation when its remote API or binary is unavailable, so the
script always produces a result file. This matches the design of the
underlying clients.

Usage::

    # Predict structure for a UniProt accession (uses AlphaFold EBI API).
    python scripts/enrich_with_external_tools.py \\
        --uniprot-id P00533 --output outputs/enrichment/P00533.json

    # Run BLAST + secondary-structure on a raw sequence.
    python scripts/enrich_with_external_tools.py \\
        --sequence "MKLAIV..." --output outputs/enrichment/seq.json \\
        --tools blast,psipred

    # Dry run: print which tools are available and exit.
    python scripts/enrich_with_external_tools.py --dry-run

Inputs (one of):
    --uniprot-id ID     Fetch sequence from UniProt and run tools.
    --sequence SEQ      Use a literal amino-acid sequence (1-letter codes).

Options:
    --tools LIST        Comma-separated subset of {alphafold,blast,clustalw,psipred}.
                        Default: all four.
    --output PATH       Output JSON path. Parent dirs are created.
    --dry-run           Skip API calls; only report tool availability.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_project_root() -> None:
    """Prepend the project root to ``sys.path`` so ``src.*`` resolves.

    Matches the convention used by the other scripts (e.g. ``predict.py``):
    insert the project root before any ``from src...`` import so the script
    works whether or not the package has been pip-installed.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from src.models.external_tools import (  # noqa: E402  (after path setup)
    AlphaFoldClient,
    BLASTClient,
    ClustalWClient,
    ChouFasmanClient,
)

logger = logging.getLogger("enrich_with_external_tools")

ALL_TOOLS = ("alphafold", "blast", "clustalw", "psipred")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_uniprot_sequence(uniprot_id: str) -> Optional[str]:
    """Fetch a canonical protein sequence from UniProt by accession.

    Returns ``None`` (and logs a warning) on any failure so the caller can
    decide whether to proceed without a sequence.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; cannot fetch UniProt sequence")
        return None

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        resp = requests.get(url, timeout=30)
        if not resp.ok:
            logger.warning("UniProt fetch %s returned HTTP %s", uniprot_id, resp.status_code)
            return None
        text = resp.text.strip()
        if not text:
            return None
        # FASTA: first line is the header, the rest is the sequence.
        lines = text.splitlines()
        return "".join(lines[1:]).strip()
    except Exception as exc:
        logger.warning("UniProt fetch %s failed: %s", uniprot_id, exc)
        return None


def _run_tool(name: str, sequence: str, uniprot_id: Optional[str]) -> Dict[str, Any]:
    """Dispatch to one external tool client and capture its result + errors.

    Each tool is wrapped so a single failure never aborts the whole enrichment
    run; the output records ``"error"`` alongside ``"available"`` so consumers
    can tell apart "tool unreachable" from "tool ran and returned empty".
    """
    entry: Dict[str, Any] = {"name": name}
    try:
        if name == "alphafold":
            client = AlphaFoldClient()
            entry["available"] = client.check_available()
            result = client.predict_structure(sequence, uniprot_id=uniprot_id or "")
            # pdb_string can be very large; store length + first lines for the
            # JSON summary, and write the full PDB to a sibling .pdb file.
            pdb = result.get("pdb_string", "")
            entry["confidence"] = result.get("confidence")
            entry["pdb_length"] = len(pdb)
            entry["pdb_preview"] = "\n".join(pdb.splitlines()[:5])
            entry["pdb_path_hint"] = "<set --output to capture full PDB>"
        elif name == "blast":
            client = BLASTClient()
            entry["available"] = client.check_available()
            result = client.search_homologs(sequence)
            hits = result.get("hits", [])
            entry["hit_count"] = len(hits)
            entry["top_hits"] = hits[:5]
        elif name == "clustalw":
            client = ClustalWClient()
            entry["available"] = client.check_available()
            # Self-alignment as a smoke check; real usage passes multiple seqs.
            result = client.align_sequences([sequence])
            entry["alignment_length"] = len(result.get("alignment", ""))
        elif name == "psipred":
            client = ChouFasmanClient()
            entry["available"] = client.check_available()
            result = client.predict_secondary_structure(sequence)
            entry["secondary_structure"] = result.get("secondary_structure", "")
            entry["helix_fraction"] = result.get("helix_fraction")
            entry["sheet_fraction"] = result.get("sheet_fraction")
            entry["coil_fraction"] = result.get("coil_fraction")
        else:
            entry["error"] = f"unknown tool: {name}"
    except Exception as exc:
        # Network failures / missing binaries must not crash the script.
        logger.exception("%s failed", name)
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enrich a protein with external tool outputs (AlphaFold / BLAST / ClustalW / PSIPRED).",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--uniprot-id", help="UniProt accession to fetch a sequence for.")
    grp.add_argument("--sequence", help="Literal amino-acid sequence (1-letter codes).")
    p.add_argument(
        "--tools",
        default=",".join(ALL_TOOLS),
        help=f"Comma-separated subset of {{{','.join(ALL_TOOLS)}}}. Default: all.",
    )
    p.add_argument(
        "--output",
        default="outputs/enrichment/enrichment.json",
        help="Output JSON path. Parent dirs are created.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report tool availability; do not call any tool.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve tools list (validate + dedupe, preserve order).
    requested = [t.strip().lower() for t in args.tools.split(",") if t.strip()]
    invalid = [t for t in requested if t not in ALL_TOOLS]
    if invalid:
        logger.error("Unknown tools: %s. Valid: %s", invalid, ALL_TOOLS)
        return 2
    tools = list(dict.fromkeys(requested))  # preserve order, dedupe

    # Resolve sequence (and uniprot id if provided).
    sequence: Optional[str] = args.sequence
    uniprot_id: Optional[str] = args.uniprot_id
    if uniprot_id and not sequence:
        sequence = _fetch_uniprot_sequence(uniprot_id)
        if not sequence:
            logger.error("Could not fetch sequence for %s; aborting.", uniprot_id)
            return 3
    if not sequence and not args.dry_run:
        logger.error("Either --uniprot-id or --sequence is required (unless --dry-run).")
        return 2

    # Dry run: just probe availability.
    if args.dry_run:
        probes = {}
        for name in tools:
            client_cls = {
                "alphafold": AlphaFoldClient,
                "blast": BLASTClient,
                "clustalw": ClustalWClient,
                "psipred": ChouFasmanClient,
            }[name]
            try:
                probes[name] = {"available": client_cls().check_available()}
            except Exception as exc:
                probes[name] = {"available": False, "error": str(exc)}
        print(json.dumps({"tools": probes}, indent=2))
        return 0

    assert sequence is not None  # for type checkers; guarded above

    # Run each requested tool.
    results: Dict[str, Any] = {
        "uniprot_id": uniprot_id,
        "sequence_length": len(sequence),
        "tools": {},
    }
    for name in tools:
        logger.info("Running tool: %s", name)
        results["tools"][name] = _run_tool(name, sequence, uniprot_id)

    # Persist JSON output (create parent dirs).
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Enrichment written to %s", out_path)

    # Also dump the full AlphaFold PDB next to the JSON when requested.
    af = results["tools"].get("alphafold") or {}
    if af.get("name") == "alphafold" and "error" not in af:
        try:
            client = AlphaFoldClient()
            pdb = client.predict_structure(sequence, uniprot_id=uniprot_id or "").get("pdb_string", "")
            if pdb:
                pdb_path = out_path.with_suffix(".pdb")
                pdb_path.write_text(pdb, encoding="utf-8")
                af["pdb_path"] = str(pdb_path)
                # Re-dump so the JSON records the .pdb path.
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Could not write sidecar PDB: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
