#!/usr/bin/env python3
"""Run official KSTAR 1.2.x in the isolated ``kstar`` environment.

The CLI has two explicit modes. ``mapping`` only proves the KSTAR mapping API
and writes the mapped handover. ``analysis`` runs the official KSTAR API twice,
once for increased and once for decreased evidence, then writes the signed
standard ``ptm_activity.tsv`` contract. Neither mode is a biology PASS.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = PROJECT_ROOT / "src" / "analysis"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

from kstar_adapter import (  # noqa: E402
    KSTAR_ADAPTER_SCHEMA_VERSION,
    KSTARAdapterError,
    build_kstar_input,
    convert_kstar_outputs,
    write_adapter_manifest,
)
from kstar_resources import (  # noqa: E402
    KSTAR_PINNED_PYTHON,
    KSTAR_PINNED_VERSION,
    KSTARResourceError,
    default_resource_hash_manifest,
    verify_kstar_network_dir,
    verify_kstar_resource_files,
    verify_pinned_python_packages,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--standardized-ptm", type=Path, required=True, help="stage-1 standardized_ptm.tsv")
    parser.add_argument("--input-manifest", type=Path, required=True, help="stage-1 ptm_input_manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="mapped TSV or standard ptm_activity.tsv")
    parser.add_argument("--adapter-manifest", type=Path, required=True, help="adapter manifest JSON")
    parser.add_argument("--case-condition", required=True, help="case condition in standardized PTM table")
    parser.add_argument("--reference-condition", required=True, help="reference condition in standardized PTM table")
    parser.add_argument("--contrast", required=True, help="frozen contrast label")
    parser.add_argument("--network-dir", type=Path, required=True, help="KSTAR network root containing ST/Y")
    parser.add_argument("--kstar-output-dir", type=Path, required=True, help="directory for official KSTAR outputs")
    parser.add_argument("--phospho-type", choices=("ST", "Y"), required=True, help="KSTAR network type for this run")
    parser.add_argument("--mode", choices=("mapping", "analysis"), required=True)
    parser.add_argument("--name", default="ptm_activity", help="KSTAR run name")
    parser.add_argument("--network-name", default="Default", help="network name under the selected type")
    parser.add_argument("--fold-threshold", type=float, default=1.2)
    parser.add_argument(
        "--min-donors-per-state",
        type=int,
        default=1,
        help="minimum usable donors per condition for KSTAR evidence; formal runs must raise this explicitly",
    )
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument(
        "--ensembl-mapping",
        type=Path,
        help="symbol-to-Ensembl TSV or pickle; required for --mode analysis so regulator_id matches the signed network",
    )
    parser.add_argument("--smoke", action="store_true", help="mark this run smoke_only; never formal biology")
    return parser.parse_args(argv)


def _require_isolated_environment() -> str:
    environment = os.environ.get("CONDA_DEFAULT_ENV", "")
    if environment != "kstar":
        raise RuntimeError(
            "run_kstar_activity.py must run inside the isolated conda environment 'kstar'; "
            f"CONDA_DEFAULT_ENV={environment!r}"
        )
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        raise RuntimeError("kstar environment is missing CONDA_PREFIX")
    expected_prefix = Path(conda_prefix).expanduser().resolve()
    interpreter = Path(sys.executable).expanduser().resolve()
    actual_prefix = Path(sys.prefix).expanduser().resolve()
    if actual_prefix != expected_prefix:
        raise RuntimeError(
            f"kstar interpreter prefix mismatch: sys.prefix={actual_prefix}, CONDA_PREFIX={expected_prefix}"
        )
    try:
        interpreter.relative_to(expected_prefix)
    except ValueError as exc:
        raise RuntimeError(
            f"kstar interpreter path mismatch: sys.executable={interpreter}, CONDA_PREFIX={expected_prefix}"
        ) from exc
    version = importlib.metadata.version("kstar")
    actual_python = ".".join(str(part) for part in sys.version_info[:2])
    expected_python = ".".join(KSTAR_PINNED_PYTHON.split(".")[:2])
    if actual_python != expected_python:
        raise RuntimeError(f"kstar Python {KSTAR_PINNED_PYTHON} is required, found {sys.version.split()[0]}")
    try:
        verify_pinned_python_packages(
            {
                "kstar": version,
                "pandas": importlib.metadata.version("pandas"),
                "numpy": importlib.metadata.version("numpy"),
                "scipy": importlib.metadata.version("scipy"),
                "requests": importlib.metadata.version("requests"),
                "tqdm": importlib.metadata.version("tqdm"),
                "matplotlib": importlib.metadata.version("matplotlib"),
                "seaborn": importlib.metadata.version("seaborn"),
                "fpdf2": importlib.metadata.version("fpdf2"),
            }
        )
    except KSTARResourceError as exc:
        raise RuntimeError(str(exc)) from exc
    if version != KSTAR_PINNED_VERSION:
        raise RuntimeError(f"official KSTAR {KSTAR_PINNED_VERSION} is required, found {version}")
    return version


def _metrics_from_kinact(kinact: Any, data_column: str, kstar_config: Any) -> pd.DataFrame:
    """Compute substrate counts/coverage from the loaded official networks."""

    evidence = kinact.evidence_binary
    evidence_sites = set(
        zip(
            evidence.loc[evidence[data_column] >= 1, kstar_config.KSTAR_ACCESSION],
            evidence.loc[evidence[data_column] >= 1, kstar_config.KSTAR_SITE],
            strict=True,
        )
    )
    annotated: dict[str, set[tuple[str, str]]] = {}
    for network in kinact.networks.values():
        for kinase, group in network.groupby(kstar_config.KSTAR_KINASE, sort=False):
            annotated.setdefault(str(kinase), set()).update(
                zip(group[kstar_config.KSTAR_ACCESSION], group[kstar_config.KSTAR_SITE], strict=True)
            )
    rows = []
    for kinase, sites in sorted(annotated.items()):
        matched = sites & evidence_sites
        rows.append(
            {
                "regulator_id": kinase,
                "n_substrates": len(matched),
                "network_coverage": len(matched) / len(sites) if sites else 0.0,
            }
        )
    if not rows:
        raise RuntimeError("KSTAR produced no kinase network metrics")
    return pd.DataFrame(rows)


def _normalize_kstar_handover(
    kinact_dict: Any,
    *,
    phospho_type: str,
    run_name: str,
    odir: Path,
    data_column: str,
    kstar_calculate: Any,
    kstar_config: Any,
) -> None:
    """Repair KSTAR 1.2.0's unnamed result indices before ``from_kstar``."""

    expected_index = getattr(kstar_config, "KSTAR_KINASE", None)
    if expected_index != "KSTAR_KINASE":
        raise KSTARAdapterError(f"KSTAR configuration has unexpected kinase index name: {expected_index!r}")
    if not isinstance(kinact_dict, dict) or phospho_type not in kinact_dict:
        raise KSTARAdapterError(f"official KSTAR result has no {phospho_type} KinaseActivity object")

    kinact = kinact_dict[phospho_type]
    frames: list[pd.DataFrame] = []
    needs_resave = False
    for attribute in ("activities_mann_whitney", "fpr_mann_whitney"):
        frame = getattr(kinact, attribute, None)
        if not isinstance(frame, pd.DataFrame):
            raise KSTARAdapterError(f"official KSTAR result is missing {attribute}")
        if data_column not in frame.columns:
            raise KSTARAdapterError(f"official KSTAR {attribute} is missing data column {data_column!r}")
        if frame.empty or frame.index.has_duplicates:
            raise KSTARAdapterError(f"official KSTAR {attribute} has an empty or duplicate kinase index")
        if frame.index.name not in (None, expected_index):
            raise KSTARAdapterError(
                f"official KSTAR {attribute} has unexpected index name {frame.index.name!r}; "
                f"expected {expected_index!r}"
            )
        if any(bool(value) for value in frame.index.isna()) or any(str(value).strip() == "" for value in frame.index):
            raise KSTARAdapterError(f"official KSTAR {attribute} has an empty kinase identifier")
        if frame.index.name is None:
            frame.index.name = expected_index
            needs_resave = True
        setattr(kinact, attribute, frame)
        frames.append(frame)

    if not frames[0].index.equals(frames[1].index):
        raise KSTARAdapterError("official KSTAR activity and FPR outputs have different kinase indices")
    if needs_resave:
        save_kstar = getattr(kstar_calculate, "save_kstar", None)
        if not callable(save_kstar):
            raise KSTARAdapterError("official KSTAR calculate module cannot re-save normalized outputs")
        save_kstar(kinact_dict, run_name, str(odir), minimal=True, ftype="tsv", param_format="json")


def _run_direction(
    *,
    mapped: pd.DataFrame,
    direction: str,
    data_column: str,
    args: argparse.Namespace,
    kstar_calculate: Any,
    kstar_config: Any,
) -> tuple[Path, pd.DataFrame]:
    run_name = f"{args.name}_{direction}"
    args.kstar_output_dir.mkdir(parents=True, exist_ok=True)
    kinact_dict = kstar_calculate.run_kstar_analysis(
        mapped,
        odir=str(args.kstar_output_dir),
        name=run_name,
        phospho_types=[args.phospho_type],
        data_columns=[data_column],
        threshold=1.0,
        evidence_size=None,
        greater=True,
        agg="mean",
        min_evidence_size=0,
        allow_column_loss=False,
        save_output=True,
        show_taskbar=False,
        PROCESSES=args.processes,
        network_dir=str(args.network_dir),
        network_name=args.network_name,
    )
    _normalize_kstar_handover(
        kinact_dict,
        phospho_type=args.phospho_type,
        run_name=run_name,
        odir=args.kstar_output_dir,
        data_column=data_column,
        kstar_calculate=kstar_calculate,
        kstar_config=kstar_config,
    )
    result_path = (
        args.kstar_output_dir
        / "RESULTS"
        / args.phospho_type
        / f"{run_name}_{args.phospho_type}_mann_whitney_activities.tsv"
    )
    if not result_path.is_file():
        raise RuntimeError(f"official KSTAR run did not write its result: {result_path}")
    # Reconstruct the object to obtain the network metrics from the same run.
    # The saved result is still read by convert_kstar_outputs below.
    kinact_dict = kstar_calculate.from_kstar(run_name, str(args.kstar_output_dir))
    if not kinact_dict or args.phospho_type not in kinact_dict:
        raise RuntimeError(f"official KSTAR result cannot be reloaded for {args.phospho_type}: {run_name}")
    metrics = _metrics_from_kinact(kinact_dict[args.phospho_type], data_column, kstar_config)
    return result_path, metrics


def _run(args: argparse.Namespace) -> int:
    version = _require_isolated_environment()
    if args.processes < 1:
        raise KSTARAdapterError("--processes must be >= 1")
    if not args.network_dir.expanduser().is_dir():
        raise FileNotFoundError(f"KSTAR network directory does not exist: {args.network_dir}")
    if args.mode == "analysis" and args.ensembl_mapping is None:
        raise KSTARAdapterError("--ensembl-mapping is required for --mode analysis")

    input_frame, audit = build_kstar_input(
        args.standardized_ptm,
        input_manifest=args.input_manifest,
        case_condition=args.case_condition,
        reference_condition=args.reference_condition,
        contrast=args.contrast,
        fold_threshold=args.fold_threshold,
        min_donors_per_state=args.min_donors_per_state,
    )
    args.kstar_output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.kstar_output_dir / f"{args.name}_kstar_input.tsv"
    input_frame.to_csv(input_path, sep="\t", index=False)

    # KSTAR is imported only after the environment/version gate.
    from kstar import calculate
    from kstar import config as kstar_config
    from kstar import mapping

    network_audit = None
    try:
        verify_kstar_resource_files(kstar_config.RESOURCE_DIR, manifest_path=default_resource_hash_manifest())
        if args.mode == "analysis":
            network_audit = verify_kstar_network_dir(args.network_dir)
    except KSTARResourceError as exc:
        raise RuntimeError(str(exc)) from exc

    if args.mode == "analysis":
        kstar_config.update_network_directory(
            network_dir=str(args.network_dir),
            y_network_name=args.network_name,
            st_network_name=args.network_name,
        )

    mapper = mapping.ExperimentMapper(
        input_frame,
        columns={"accession_id": "accession", "site": "site"},
        odir=str(args.kstar_output_dir),
        name=args.name,
        data_columns=[f"data:{args.contrast}:increased", f"data:{args.contrast}:decreased"],
        show_taskbar=False,
        auto_convert_ids=False,
        auto_format_peptides=False,
    )
    mapped = mapper.get_experiment()
    mapped_path = args.kstar_output_dir / f"{args.name}_mapped.tsv"
    mapped.to_csv(mapped_path, sep="\t", index=False)
    lineage_boundary = "smoke_only" if args.smoke else "kstar_engineering"
    base_manifest: dict[str, Any] = {
        "schema_version": KSTAR_ADAPTER_SCHEMA_VERSION,
        "tool": {
            "name": "KSTAR",
            "version": version,
            "api": "kstar.mapping.ExperimentMapper",
        },
        "input": {
            "standardized_ptm": str(args.standardized_ptm.expanduser().resolve(strict=True)),
            "input_manifest": str(args.input_manifest.expanduser().resolve(strict=True)),
            "kstar_input": str(input_path),
            "mapped_output": str(mapped_path),
        },
        "design": {
            "case_condition": args.case_condition,
            "reference_condition": args.reference_condition,
            "contrast": args.contrast,
            "phospho_type": args.phospho_type,
            "fold_threshold": args.fold_threshold,
            "min_donors_per_state": args.min_donors_per_state,
            "directional_evidence": ["increased", "decreased"],
        },
        "input_audit": audit.as_dict(),
        "network_audit": network_audit.as_dict() if network_audit is not None else None,
        "ensembl_mapping": (
            str(args.ensembl_mapping.expanduser().resolve(strict=True)) if args.ensembl_mapping is not None else None
        ),
        "lineage_boundary": lineage_boundary,
        "mode": args.mode,
    }
    if args.mode == "mapping":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mapped.to_csv(args.output, sep="\t", index=False)
        base_manifest["output"] = {"file": str(args.output.expanduser().resolve()), "kind": "mapped_kstar_input"}
        base_manifest["notes"] = "mapping-only engineering check; no kinase activity was inferred"
        write_adapter_manifest(args.adapter_manifest, base_manifest)
        return 0

    increased_col = f"data:{args.contrast}:increased"
    decreased_col = f"data:{args.contrast}:decreased"
    increased_path, increased_metrics = _run_direction(
        mapped=mapped,
        direction="increased",
        data_column=increased_col,
        args=args,
        kstar_calculate=calculate,
        kstar_config=kstar_config,
    )
    decreased_path, decreased_metrics = _run_direction(
        mapped=mapped,
        direction="decreased",
        data_column=decreased_col,
        args=args,
        kstar_calculate=calculate,
        kstar_config=kstar_config,
    )
    # Each direction's evidence set yields its own n_substrates/coverage; the
    # adapter binds per-kinase metrics to the winning (smaller-p) direction.
    increased_metrics_path = args.kstar_output_dir / f"{args.name}_increased_network_metrics.tsv"
    decreased_metrics_path = args.kstar_output_dir / f"{args.name}_decreased_network_metrics.tsv"
    increased_metrics.to_csv(increased_metrics_path, sep="\t", index=False)
    decreased_metrics.to_csv(decreased_metrics_path, sep="\t", index=False)
    activity = convert_kstar_outputs(
        increased_path,
        decreased_path,
        contrast=args.contrast,
        method_version=version,
        input_manifest=args.input_manifest,
        increased_metrics=increased_metrics_path,
        decreased_metrics=decreased_metrics_path,
        ensembl_mapping=args.ensembl_mapping,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    activity.to_csv(args.output, sep="\t", index=False)
    base_manifest["tool"]["api"] = "kstar.calculate.run_kstar_analysis"
    base_manifest["directional_analysis"] = {
        "increased": str(increased_path),
        "decreased": str(decreased_path),
        "increased_metrics": {
            "file": str(increased_metrics_path),
            "n_kinases": int(len(increased_metrics)),
            "n_substrates_total": int(increased_metrics["n_substrates"].sum()),
        },
        "decreased_metrics": {
            "file": str(decreased_metrics_path),
            "n_kinases": int(len(decreased_metrics)),
            "n_substrates_total": int(decreased_metrics["n_substrates"].sum()),
        },
        "metrics_binding": (
            "per-kinase n_substrates/network_coverage are taken from the winning direction "
            "(the directional analysis with the smaller p-value)"
        ),
        "score_semantics": "sign from separate evidence analysis; magnitude is -log10(raw KSTAR p-value)",
        "qvalue_semantics": "Benjamini-Hochberg adjustment across both directional KSTAR p-value outputs",
    }
    base_manifest["output"] = {
        "file": str(args.output.expanduser().resolve()),
        "kind": "ptm_activity.tsv",
        "n_rows": len(activity),
    }
    base_manifest["notes"] = (
        "KSTAR p-values are unsigned; sign is assigned only after paired increased/decreased analyses. "
        "biology_pass remains false until real PTM and a registered benchmark pass."
    )
    write_adapter_manifest(args.adapter_manifest, base_manifest)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _run(args)
    except (KSTARAdapterError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[run_kstar_activity] hard failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
