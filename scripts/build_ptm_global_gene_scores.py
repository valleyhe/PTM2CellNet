#!/usr/bin/env python3
"""Stage 3 of the PTM-activity mainline (方案 §5.3/§7 阶段 3).

Propagate the *primary* activity method's signed regulator scores through
the frozen signed network (§4.3 contract) and emit the §4.4
``ptm_global_gene_scores.tsv`` plus a network/score manifest. A sensitivity
activity method (方案 §4.2) must be propagated by a separate run of this
script — methods are compared, never averaged.

No PTM-side significance is claimed: ``prediction_status`` stays
``direction_only`` until an independent null or external benchmark is
calibrated (方案 §4.4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.ptm_activity import (  # noqa: E402
    PTMActivityContractError,
    load_ptm_activity_table,
)
from src.analysis.ptm_activity_admission import select_activities_for_propagation  # noqa: E402
from src.analysis.ptm_activity_benchmark import (  # noqa: E402
    evaluate_activity_benchmark,
    load_activity_benchmark_table,
)
from src.analysis.ptm_gene_score import (  # noqa: E402
    PTMGeneScoreError,
    build_gene_score_table,
    load_network_id_map,
    write_gene_score_table,
)
from src.analysis.ptm_research_config import PTMResearchConfigError, load_ptm_research_config  # noqa: E402
from src.analysis.signed_network import (  # noqa: E402
    SignedNetworkContractError,
    load_signed_network,
    propagate_signed_scores,
    verify_network_release_binding,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="frozen ptm_research_config.yaml")
    parser.add_argument(
        "--activity-tsv", type=Path, required=True, help="ptm_activity.tsv (§4.2, external tool output)"
    )
    parser.add_argument(
        "--activity-benchmark",
        type=Path,
        default=None,
        help=(
            "optional kinase-perturbation benchmark TSV (方案 §5.2); requires pre-registered "
            "activity_benchmark criteria in the frozen config; FAIL keeps outputs exploratory"
        ),
    )
    parser.add_argument("--network-tsv", type=Path, required=True, help="signed network edge table (§4.3)")
    parser.add_argument("--network-id-map", type=Path, required=True, help="network_id/gene_symbol/ensembl_id TSV")
    parser.add_argument(
        "--condition-or-contrast",
        default=None,
        help="must match config.contrast when provided (default: frozen config.contrast)",
    )
    parser.add_argument("--output-tsv", type=Path, required=True, help="ptm_global_gene_scores.tsv output")
    parser.add_argument("--manifest-output", type=Path, required=True, help="gene-score manifest output")
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_ptm_research_config(args.config)
        if args.condition_or_contrast is not None and args.condition_or_contrast != config.contrast:
            raise PTMActivityContractError(
                "--condition-or-contrast must match frozen config.contrast "
                f"{config.contrast!r}; got {args.condition_or_contrast!r}"
            )
        frozen_contrast = config.contrast
        activity_frame = load_ptm_activity_table(args.activity_tsv)
        benchmark_report = None
        if args.activity_benchmark is not None:
            if config.activity_benchmark is None:
                raise PTMActivityContractError(
                    "--activity-benchmark requires pre-registered activity_benchmark criteria in the frozen "
                    "config (方案 §8.7 阈值不得事后挑选); refusing to invent thresholds"
                )
            benchmark_table = load_activity_benchmark_table(args.activity_benchmark)
            benchmark_report = evaluate_activity_benchmark(
                activity_frame,
                benchmark_table,
                criteria=config.activity_benchmark,
                method=config.primary_activity_method,
                condition_or_contrast=frozen_contrast,
            )
            if config.mode == "formal" and not benchmark_report.passed:
                raise PTMActivityContractError(
                    f"formal mode rejects a benchmark {benchmark_report.verdict} verdict; "
                    "outputs stay exploratory (方案 §5.2 第 5 条)"
                )
        selection = select_activities_for_propagation(
            activity_frame,
            method=config.primary_activity_method,
            condition_or_contrast=frozen_contrast,
            policy=config.activity_admission,
            benchmark_report=benchmark_report,
        )
        activities = dict(selection.admitted)
        network = load_signed_network(
            args.network_tsv,
            expected_species=config.species,
            expected_release=config.network_release,
        )
        release_binding: dict | None = None
        if config.network_release_manifest is not None:
            release_binding = verify_network_release_binding(
                args.network_tsv,
                config.network_release_manifest,
                expected_release=config.network_release,
            )
        network_id_map = load_network_id_map(args.network_id_map)
        propagation = propagate_signed_scores(network, activities, config=config.propagation)
        provenance = (
            f"activity={args.activity_tsv.expanduser().resolve()}[method={config.primary_activity_method}]; "
            f"network={args.network_tsv.expanduser().resolve()}[release={config.network_release}]"
        )
        score_frame, score_audit = build_gene_score_table(
            propagation,
            network_id_map=network_id_map,
            config=config,
            provenance=provenance,
        )
        output_path = write_gene_score_table(score_frame, args.output_tsv)
    except (
        PTMResearchConfigError,
        PTMActivityContractError,
        PTMGeneScoreError,
        SignedNetworkContractError,
        FileNotFoundError,
    ) as exc:
        print(f"[build_ptm_global_gene_scores] 输入契约失败：{exc}", file=sys.stderr)
        return 1
    manifest = {
        "schema_version": "ptm2cellnet.ptm-gene-score-manifest/v1",
        "sources": {
            "activity_table": {
                "file": str(args.activity_tsv.expanduser().resolve()),
                "sha256": _sha256_file(args.activity_tsv.expanduser().resolve()),
                "primary_method": config.primary_activity_method,
                "condition_or_contrast": frozen_contrast,
                "admission": selection.as_dict(),
            },
            "signed_network": {
                "file": str(args.network_tsv.expanduser().resolve()),
                "sha256": _sha256_file(args.network_tsv.expanduser().resolve()),
                "release_binding": release_binding,
                "audit": {
                    "n_rows_input": network.audit.n_rows_input,
                    "n_unsigned_rows": network.audit.n_unsigned_rows,
                    "n_self_loops": network.audit.n_self_loops,
                    "n_sign_conflict_edges": network.audit.n_sign_conflict_edges,
                    "n_default_confidence": network.audit.n_default_confidence,
                    "n_edges_propagation": network.audit.n_edges_propagation,
                    "species": list(network.audit.species),
                    "releases": list(network.audit.releases),
                    "edge_types": list(network.audit.edge_types),
                },
            },
            "network_id_map": {
                "file": str(args.network_id_map.expanduser().resolve()),
                "sha256": _sha256_file(args.network_id_map.expanduser().resolve()),
            },
            **(
                {
                    "activity_benchmark": {
                        "file": str(args.activity_benchmark.expanduser().resolve()),
                        "sha256": _sha256_file(args.activity_benchmark.expanduser().resolve()),
                    }
                }
                if args.activity_benchmark is not None
                else {}
            ),
        },
        "propagation": {
            "max_depth": config.propagation.max_depth,
            "decay": config.propagation.decay,
            "gene_edge_types": list(config.propagation.gene_edge_types),
            "max_paths_per_seed": config.propagation.max_paths_per_seed,
            "seeds_matched": list(propagation.seeds_matched),
            "seeds_without_node": list(propagation.seeds_without_node),
            "n_gene_paths_total": propagation.n_gene_paths_total,
            "per_seed_path_counts": propagation.diagnostics["per_seed_path_counts"],
            "path_count_distribution": propagation.diagnostics["path_count_distribution"],
        },
        "score_table_audit": {
            "n_rows": len(score_frame),
            "n_propagation_pairs": score_audit.n_propagation_pairs,
            "n_targets_unmapped": score_audit.n_targets_unmapped,
            "unmapped_targets": list(score_audit.unmapped_targets),
            "n_direction_indeterminate": score_audit.n_direction_indeterminate,
        },
        "notes": {
            "significance": (
                "gene_score lives on the propagation-internal scale; no q-value is emitted and "
                "prediction_status stays direction_only until an independent null or external benchmark "
                "is calibrated (方案 §4.4)"
            ),
            "sensitivity_method": (
                "a sensitivity activity method must be propagated by a separate run of this script; "
                "primary and sensitivity results are compared, never averaged (方案 §4.2)"
            ),
        },
        "output": {"file": str(output_path)},
    }
    manifest_path = args.manifest_output.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output_path),
                "manifest": str(manifest_path),
                "n_score_rows": len(score_frame),
                "n_seeds_matched": len(propagation.seeds_matched),
                "n_seeds_without_node": len(propagation.seeds_without_node),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
