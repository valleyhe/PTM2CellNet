#!/usr/bin/env python3
"""Stage 5 of the PTM-activity mainline (方案 §5.5/§7 阶段 5).

For every frozen cell type, assemble the formal candidate spec consumed by
``run_davf_perturbgen_e2e.py --candidate-spec`` plus the
``downstream_targets_<cell_type>.json`` sidecar that carries the one-to-many
source → target relation (方案 §3.3: source and target roles never merge).

Research inputs are explicit and never generated here:

* ``--source-proposals-tsv`` — literature-anchored PTM-site proposals
  (protein, position, PTM type, direction hypothesis, proposal confidence);
* the source gene's own AD DEG row is required for the existing three-way
  direction gate — a source without DEG evidence in that cell type is
  listed as exploratory and does NOT become a formal candidate row
  (方案 §5.5: target-set concordance cannot substitute the source-gene gate);
* ``--context-h5ad`` binds a real ``context_cell_index`` (first cell of the
  cell type) instead of a default row 0 (方案 §4.6);
* with ``--embedding-vocab``, a source gene without a verified PerturbGen
  token fails hard (方案 §5.5).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.ptm_gene_score import PTMGeneScoreError, load_deg_table  # noqa: E402
from src.analysis.ptm_research_config import PTMResearchConfigError, load_ptm_research_config  # noqa: E402
from src.integration.perturbgen.downstream_target_evaluation import (  # noqa: E402
    DOWNSTREAM_TARGET_SIDECAR_SCHEMA_VERSION,
)
from src.models.gene_vocabulary import normalize_ensembl_id  # noqa: E402

PROPOSAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "protein_id",
    "position",
    "ptm_type",
    "gene_symbol",
    "ensembl_id",
    "source_activity_id",
    "proposed_direction",
    "site_probability",
    "provenance",
)


class ProposalContractError(ValueError):
    """Raised when the source-proposal research input violates the contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="frozen ptm_research_config.yaml")
    parser.add_argument("--source-proposals-tsv", type=Path, required=True, help="literature-anchored source proposals")
    parser.add_argument("--deg-table", type=Path, required=True, help="donor-level AD DEG table (source-gene evidence)")
    parser.add_argument("--target-set-manifest", type=Path, required=True, help="stage-4 target_set_manifest.json")
    parser.add_argument("--context-h5ad", type=Path, default=None, help="cohort h5ad (default: config.cohort_h5ad)")
    parser.add_argument("--cell-type-obs-column", default="cell_type")
    parser.add_argument(
        "--embedding-vocab",
        type=Path,
        default=None,
        help="PerturbGen gene vocabulary JSON (gene -> token id); required for formal runs (方案 §5.5)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def load_source_proposals(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the source-proposal research input."""

    resolved = Path(path).expanduser().resolve(strict=True)
    proposals: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in PROPOSAL_REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ProposalContractError(f"source proposals are missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            values: dict[str, Any] = {}
            for column in PROPOSAL_REQUIRED_COLUMNS:
                values[column] = (row.get(column) or "").strip()
            if any(not str(values[column]) for column in PROPOSAL_REQUIRED_COLUMNS):
                raise ProposalContractError(f"source proposal row {row_number} has an empty required field")
            position = int(values["position"])
            if position < 1:
                raise ProposalContractError(f"source proposal row {row_number} position must be >= 1")
            probability = float(values["site_probability"])
            if not 0.0 <= probability <= 1.0:
                raise ProposalContractError(f"source proposal row {row_number} site_probability must be within [0, 1]")
            direction = values["proposed_direction"].lower()
            if direction not in ("up", "down"):
                raise ProposalContractError(
                    f"source proposal row {row_number} proposed_direction must be 'up' or 'down'"
                )
            values["ensembl_id"] = normalize_ensembl_id(values["ensembl_id"])
            values["position"] = position
            values["proposed_direction"] = direction
            values["site_probability"] = probability
            values["aa"] = (row.get("aa") or "").strip()
            proposals.append(values)
    if not proposals:
        raise ProposalContractError("source proposals must contain at least one row")
    return proposals


def _first_cell_index_per_type(context_h5ad: Path, obs_column: str, cell_types: Sequence[str]) -> dict[str, int]:
    """Real per-cell-type context binding (方案 §4.6): first row of each type."""

    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("anndata is required to bind real context_cell_index values") from exc
    frame = ad.read_h5ad(context_h5ad, backed="r").obs
    if obs_column not in frame.columns:
        raise ProposalContractError(f"context h5ad has no obs column {obs_column!r}; available: {list(frame.columns)}")
    labels = frame[obs_column].astype(str)
    indices: dict[str, int] = {}
    for cell_type in cell_types:
        positions = np.flatnonzero((labels == cell_type).to_numpy())
        if len(positions) == 0:
            raise ProposalContractError(
                f"context h5ad has no cells of frozen cell type {cell_type!r} in column {obs_column!r}"
            )
        # Positional index: the E2E consumer indexes context rows by position, not label.
        indices[cell_type] = int(positions[0])
    return indices


def _load_embedding_vocab(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProposalContractError("embedding vocabulary must be a JSON object (gene -> token id)")
    return {str(gene): int(token) for gene, token in payload.items()}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_ptm_research_config(args.config)
        proposals = load_source_proposals(args.source_proposals_tsv)
        deg_frame = load_deg_table(args.deg_table)
        target_set_payload = json.loads(
            args.target_set_manifest.expanduser().resolve(strict=True).read_text(encoding="utf-8")
        )
        if not isinstance(target_set_payload, dict) or not isinstance(target_set_payload.get("cell_types"), dict):
            raise ProposalContractError("target-set manifest must contain a cell_types mapping")
        context_h5ad = (
            args.context_h5ad.expanduser().resolve(strict=True)
            if args.context_h5ad is not None
            else (PROJECT_ROOT / config.cohort_h5ad).resolve(strict=True)
        )
        context_indices = _first_cell_index_per_type(context_h5ad, args.cell_type_obs_column, config.cell_types)
        embedding_vocab = _load_embedding_vocab(args.embedding_vocab) if args.embedding_vocab is not None else None
    except (PTMResearchConfigError, PTMGeneScoreError, ProposalContractError, FileNotFoundError) as exc:
        print(f"[build_celltype_candidate_specs] 输入契约失败：{exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"[build_celltype_candidate_specs] 环境失败：{exc}", file=sys.stderr)
        return 1
    if embedding_vocab is not None:
        for proposal in proposals:
            if proposal["ensembl_id"] not in embedding_vocab:
                print(
                    f"[build_celltype_candidate_specs] source gene {proposal['ensembl_id']} "
                    f"({proposal['gene_symbol']}) has no verified PerturbGen token (方案 §5.5 硬失败)",
                    file=sys.stderr,
                )
                return 1
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_summary: dict[str, Any] = {
        "schema_version": "ptm2cellnet.celltype-candidate-build/v1",
        "context_h5ad": str(context_h5ad),
        "cell_types": {},
        "notes": {
            "exploratory": (
                "sources without their own DEG evidence in a cell type are exploratory only; "
                "target-set concordance cannot substitute the source-gene three-way gate (方案 §5.5)"
            ),
        },
    }
    exit_code = 0
    for cell_type in config.cell_types:
        manifest_cell = target_set_payload["cell_types"].get(cell_type)
        if manifest_cell is None:
            print(
                f"[build_celltype_candidate_specs] frozen cell type {cell_type!r} missing from the "
                "target-set manifest; run stage 4 first",
                file=sys.stderr,
            )
            exit_code = 1
            continue
        deg_cell = deg_frame[deg_frame["cell_type"].astype(str).str.strip() == cell_type]
        deg_by_ensembl = {str(row["ensembl_id"]): row for row in deg_cell.to_dict("records")}
        manifest_sources: Mapping[str, Any] = manifest_cell.get("sources", {})
        candidates: list[dict[str, Any]] = []
        sidecar_sources: dict[str, Any] = {}
        exploratory: list[dict[str, str]] = []
        unmatched_activities: list[str] = []
        for proposal in proposals:
            source_activity_id = proposal["source_activity_id"]
            entry = manifest_sources.get(source_activity_id)
            if entry is None:
                unmatched_activities.append(source_activity_id)
                continue
            deg_row = deg_by_ensembl.get(proposal["ensembl_id"])
            if deg_row is None:
                exploratory.append(
                    {
                        "source_activity_id": source_activity_id,
                        "ensembl_id": proposal["ensembl_id"],
                        "reason": "source gene has no AD DEG row in this cell type",
                    }
                )
                continue
            observed_direction = str(deg_row["observed_direction"]).strip().lower()
            gate_failures: list[str] = []
            if observed_direction not in {"up", "down"}:
                gate_failures.append("observed_direction must be 'up' or 'down'")
            observed_fdr = float(deg_row["fdr"])
            if observed_fdr > config.deg_max_fdr:
                gate_failures.append("fdr exceeds config.deg_max_fdr")
            normal_donors = int(deg_row["n_normal_donors"])
            if normal_donors < config.min_donors_per_state:
                gate_failures.append("n_normal_donors is below config.min_donors_per_state")
            disease_donors = int(deg_row["n_disease_donors"])
            if disease_donors < config.min_donors_per_state:
                gate_failures.append("n_disease_donors is below config.min_donors_per_state")
            if gate_failures:
                exploratory.append(
                    {
                        "source_activity_id": source_activity_id,
                        "ensembl_id": proposal["ensembl_id"],
                        "reason": "source DEG fails formal gate: " + "; ".join(gate_failures),
                    }
                )
                continue
            aa = proposal["aa"]
            candidates.append(
                {
                    "context_cell_index": context_indices[cell_type],
                    "gene_symbol": proposal["gene_symbol"].upper(),
                    "ensembl_id": proposal["ensembl_id"],
                    "position": proposal["position"],
                    "ptm_type": proposal["ptm_type"],
                    "proposed_direction": proposal["proposed_direction"],
                    "site_probability": proposal["site_probability"],
                    "provenance": proposal["provenance"],
                    "cell_type": cell_type,
                    "ptm_context": (
                        f"{proposal['gene_symbol'].upper()}:{aa}{proposal['position']}"
                        if aa
                        else f"{proposal['gene_symbol'].upper()}:{proposal['position']}"
                    ),
                    "observed_log2fc": float(deg_row["log2fc"]),
                    "observed_fdr": observed_fdr,
                    "observed_direction": observed_direction,
                    "direction_evidence_donor_counts": {
                        "normal": normal_donors,
                        "disease": disease_donors,
                    },
                    "semantic_context": config.semantic_context_for_cell_type(cell_type),
                    "source_activity_id": source_activity_id,
                }
            )
            sidecar_sources[proposal["ensembl_id"]] = {
                "source_activity_id": source_activity_id,
                "gene_symbol": proposal["gene_symbol"].upper(),
                "position": proposal["position"],
                "ptm_type": proposal["ptm_type"],
                "proposed_direction": proposal["proposed_direction"],
                "site_probability": proposal["site_probability"],
                "provenance": proposal["provenance"],
                "targets": entry["targets"],
            }
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in cell_type)
        cell_summary: dict[str, Any] = {
            "n_candidates": len(candidates),
            "context_cell_index": context_indices[cell_type],
            "exploratory_sources": exploratory,
            "proposals_without_target_set": sorted(set(unmatched_activities)),
            "candidate_spec": None,
            "downstream_targets": None,
        }
        if candidates:
            spec_payload = {
                "schema_version": "ptm2cellnet.candidate-spec/v1",
                "context_h5ad": str(context_h5ad),
                "candidates": candidates,
                "build_summary": {
                    "cell_type": cell_type,
                    "n_candidates": len(candidates),
                    "sources": {
                        "source_proposals": str(args.source_proposals_tsv.expanduser().resolve()),
                        "target_set_manifest": str(args.target_set_manifest.expanduser().resolve()),
                        "deg_table": str(args.deg_table.expanduser().resolve()),
                        "research_config": str(args.config.expanduser().resolve()),
                    },
                },
            }
            spec_path = output_dir / f"candidate_spec_{safe_name}.json"
            spec_path.write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            sidecar_path = output_dir / f"downstream_targets_{safe_name}.json"
            sidecar_payload = {
                "schema_version": DOWNSTREAM_TARGET_SIDECAR_SCHEMA_VERSION,
                "cell_type": cell_type,
                "sources": sidecar_sources,
                "lineage": {
                    "candidate_spec": str(spec_path),
                    "target_set_manifest": str(args.target_set_manifest.expanduser().resolve()),
                    "deg_table": str(args.deg_table.expanduser().resolve()),
                    "research_config": str(args.config.expanduser().resolve()),
                },
            }
            sidecar_path.write_text(json.dumps(sidecar_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            cell_summary["candidate_spec"] = str(spec_path)
            cell_summary["downstream_targets"] = str(sidecar_path)
        else:
            print(
                f"[build_celltype_candidate_specs] cell type {cell_type!r} produced no formal candidates "
                f"({len(exploratory)} exploratory, {len(set(unmatched_activities))} without target set)",
                file=sys.stderr,
            )
        build_summary["cell_types"][cell_type] = cell_summary
    summary_path = output_dir / "build_summary.json"
    summary_path.write_text(json.dumps(build_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": exit_code == 0,
                "output_dir": str(output_dir),
                "build_summary": str(summary_path),
            },
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
