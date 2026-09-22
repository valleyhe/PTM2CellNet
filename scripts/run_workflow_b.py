#!/usr/bin/env python3
"""Workflow B unified replayable lifecycle (REQUIREMENTS A-07, TD-08 方案 A).

Chains the existing Workflow B components into a single orchestrated run with
an immutable run manifest:

    verify_assets → build_pairs → train → evaluate → gate_e

- ``verify_assets`` re-validates the frozen embedding asset (hashes/schema via
  ``load_perturbgen_embedding_asset``), the scVI model directory and the
  explicit donor split before anything heavy runs.
- ``build_pairs``/``train``/``evaluate`` invoke the existing stage CLIs
  (``scripts.build_davf_latent_pairs`` / ``scripts.train_latent_davf`` /
  ``scripts.evaluate_latent_davf``) with their stdout JSON captured into the
  manifest.
- ``gate_e`` is opt-in via ``--stages`` (the ≥200 real direction benchmark is
  an external frozen asset; without it Gate-E must not be claimed).

The run directory must not exist or must be empty; the manifest is written
once, then sealed with a ``run_manifest.sha256`` sidecar.  Exit codes: 0 all
stages passed, 1 stage execution error, 2 Gate-E evaluated to an explicit
non-pass verdict (recorded, not hidden).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN_MANIFEST_SCHEMA = "ptm2cellnet.workflow_b_run/v1"
STAGE_ORDER: tuple[str, ...] = ("verify_assets", "build_pairs", "train", "evaluate", "gate_e")
DEFAULT_STAGES = "verify_assets,build_pairs,train,evaluate"
GATE_E_FAIL_EXIT_CODE = 2


class WorkflowBError(RuntimeError):
    """Raised when the orchestrated run cannot proceed (never for gate verdicts)."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise WorkflowBError(f"expected stage output missing: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _invoke_cli(main: Callable[[Iterable[str] | None], int], name: str, argv: Sequence[str]) -> dict[str, Any]:
    """Run a stage CLI entry point, capturing its stdout JSON payload.

    Component exceptions (invalid assets, contract violations) are recorded in
    the returned entry and re-raised so the caller can persist the failure in
    the run manifest before crashing.
    """

    captured = io.StringIO()
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(captured):
            code = main(list(argv))
    except Exception as exc:  # noqa: BLE001 -- record, then re-raise (let it crash)
        return {
            "stage": name,
            "argv": list(argv),
            "status": "crashed",
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": captured.getvalue().strip(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "re_raise": exc,
        }
    return {
        "stage": name,
        "argv": list(argv),
        "exit_code": int(code),
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout": captured.getvalue().strip(),
    }


def build_stage_argv(stage: str, args: argparse.Namespace, run_root: Path) -> list[str]:
    """Assemble the CLI argv for one stage (pure; unit-testable)."""

    pairs_dir = run_root / "latent_pairs"
    checkpoint_dir = run_root / "davf_checkpoint"
    if stage == "build_pairs":
        argv = [
            "--scvi-model",
            str(args.scvi_model),
            "--embedding-asset",
            str(args.embedding_asset),
            "--output-dir",
            str(pairs_dir),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
        ]
        if args.norman_dir:
            argv = ["--norman-dir", str(args.norman_dir)] + argv
        elif args.geo_root:
            argv = ["--geo-root", str(args.geo_root)] + argv
        return argv
    if stage == "train":
        argv = [
            "--train-data",
            str(pairs_dir / "train.npz"),
            "--val-data",
            str(pairs_dir / "val.npz"),
            "--scvi-model",
            str(args.scvi_model),
            "--intervention-type",
            str(args.intervention_type),
            "--embedding-asset",
            str(args.embedding_asset),
            "--output",
            str(checkpoint_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
        ]
        if args.train_donors or args.held_out_donors:
            argv += [
                "--train-donors",
                str(args.train_donors),
                "--held-out-donors",
                str(args.held_out_donors),
            ]
        elif args.require_donor_split:
            argv.append("--require-donor-split")
        return argv
    if stage == "evaluate":
        return [
            "--test-data",
            str(pairs_dir / "test.npz"),
            "--checkpoint",
            str(checkpoint_dir / "best_model.pt"),
            "--scvi-model",
            str(args.scvi_model),
            "--embedding-asset",
            str(args.embedding_asset),
            "--intervention-type",
            str(args.intervention_type),
            "--output",
            str(run_root / "evaluation.json"),
            "--device",
            str(args.device),
        ]
    if stage == "gate_e":
        if not args.benchmark:
            raise WorkflowBError("gate_e stage requires --benchmark (frozen Gate-E direction benchmark)")
        argv = [
            "--benchmark",
            str(args.benchmark),
            "--output",
            str(run_root / "gate_e_report.json"),
        ]
        if args.old_vocabulary:
            argv += ["--old-vocabulary", str(args.old_vocabulary)]
        if args.new_vocabulary:
            argv += ["--new-vocabulary", str(args.new_vocabulary)]
        if args.davf_paired_results:
            argv += ["--davf-paired-results", str(args.davf_paired_results)]
        return argv
    raise WorkflowBError(f"unknown stage: {stage}")


def stage_output_paths(stage: str, run_root: Path) -> list[Path]:
    """Files a successful stage must have produced (recorded with sha256)."""

    pairs_dir = run_root / "latent_pairs"
    checkpoint_dir = run_root / "davf_checkpoint"
    return {
        "build_pairs": [pairs_dir / "train.npz", pairs_dir / "val.npz", pairs_dir / "test.npz"],
        "train": [checkpoint_dir / "best_model.pt", checkpoint_dir / "training_metrics.json"],
        "evaluate": [run_root / "evaluation.json"],
        "gate_e": [run_root / "gate_e_report.json"],
    }.get(stage, [])


def verify_assets(args: argparse.Namespace, stages: Sequence[str]) -> dict[str, Any]:
    """Fail-fast validation of the shared inputs before heavy stages run."""

    from src.models.perturbgen_embedding import load_perturbgen_embedding_asset

    checks: dict[str, Any] = {}
    asset = load_perturbgen_embedding_asset(args.embedding_asset)
    checks["embedding_asset"] = {
        "path": str(Path(args.embedding_asset).expanduser().resolve()),
        "vocab_size": asset.vocab_size,
        "embedding_dim": asset.embedding_dim,
        "manifest_sha256": _sha256(Path(args.embedding_asset).expanduser().resolve() / "manifest.json"),
    }
    scvi_dir = Path(args.scvi_model).expanduser().resolve()
    if not scvi_dir.is_dir():
        raise WorkflowBError(f"scVI model directory not found: {scvi_dir}")
    checks["scvi_model"] = {"path": str(scvi_dir)}
    donor_split_explicit = bool(args.train_donors or args.held_out_donors)
    checks["donor_split"] = {
        "explicit": donor_split_explicit,
        "train_donors": args.train_donors,
        "held_out_donors": args.held_out_donors,
    }
    if "train" in stages and not donor_split_explicit and args.require_donor_split:
        raise WorkflowBError(
            "--require-donor-split set but no --train-donors/--held-out-donors provided "
            "(Workflow B requires an explicit donor split)"
        )
    if donor_split_explicit and not (args.train_donors and args.held_out_donors):
        raise WorkflowBError("--train-donors and --held-out-donors must be provided together")
    if "gate_e" in stages:
        if not args.benchmark:
            raise WorkflowBError("gate_e selected without --benchmark")
        benchmark = Path(args.benchmark).expanduser().resolve()
        if not benchmark.is_file():
            raise WorkflowBError(f"Gate-E benchmark file not found: {benchmark}")
    return checks


def _environment_snapshot() -> dict[str, Any]:
    import numpy
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": numpy.__version__,
    }


def run_workflow_b(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the selected stages and write the immutable run manifest."""

    run_root = Path(args.output_root).expanduser().resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise WorkflowBError(f"run directory is not empty (immutable lifecycle): {run_root} — use a fresh directory")
    run_root.mkdir(parents=True, exist_ok=True)

    if args.stages.strip().lower() == "all":
        stages = list(STAGE_ORDER)
    else:
        requested = [part.strip() for part in args.stages.split(",") if part.strip()]
        unknown = [stage for stage in requested if stage not in STAGE_ORDER]
        if unknown:
            raise WorkflowBError(f"unknown stages: {', '.join(unknown)} (valid: {', '.join(STAGE_ORDER)})")
        stages = [stage for stage in STAGE_ORDER if stage in requested]

    manifest: dict[str, Any] = {
        "schema": RUN_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_root),
        "config": {
            "intervention_type": args.intervention_type,
            "seed": args.seed,
            "device": args.device,
            "stages": stages,
            "norman_dir": str(args.norman_dir) if args.norman_dir else None,
            "geo_root": str(args.geo_root) if args.geo_root else None,
        },
        "environment": _environment_snapshot(),
        "inputs": {"embedding_asset": None, "scvi_model": None, "donor_split": None},
        "stages": {},
    }

    run_status = "completed"
    main_by_stage: dict[str, Callable[[Iterable[str] | None], int]] = {}
    if not args.dry_run:
        from scripts import build_davf_latent_pairs, evaluate_gate_e, evaluate_latent_davf, train_latent_davf

        main_by_stage = {
            "build_pairs": build_davf_latent_pairs.main,
            "train": train_latent_davf.main,
            "evaluate": evaluate_latent_davf.main,
            "gate_e": evaluate_gate_e.main,
        }

    for stage in stages:
        if stage == "verify_assets":
            checks = verify_assets(args, stages)
            manifest["inputs"] = {
                "embedding_asset": checks["embedding_asset"],
                "scvi_model": checks["scvi_model"],
                "donor_split": checks["donor_split"],
            }
            manifest["stages"][stage] = {"status": "passed", "checks": checks}
            continue
        if args.dry_run:
            manifest["stages"][stage] = {"status": "planned", "argv": build_stage_argv(stage, args, run_root)}
            continue

        entry: dict[str, Any] = {"status": "running"}
        manifest["stages"][stage] = entry
        _write_manifest(run_root, manifest)
        argv = build_stage_argv(stage, args, run_root)
        record = _invoke_cli(main_by_stage[stage], stage, argv)
        crash = record.pop("re_raise", None)
        entry.update(record)
        if crash is not None:
            entry["status"] = "failed"
            run_status = "failed"
            manifest["run_status"] = run_status
            _write_manifest(run_root, manifest)
            raise crash
        if record["exit_code"] != 0:
            if stage == "gate_e" and record["exit_code"] == 1 and (run_root / "gate_e_report.json").is_file():
                # evaluate_gate_e returns 1 for an explicit non-pass verdict, not a crash.
                entry["status"] = "gate_failed"
                entry["outputs"] = [_record_file(run_root / "gate_e_report.json")]
                run_status = "gate_failed"
                break
            entry["status"] = "failed"
            run_status = "failed"
            break
        entry["status"] = "passed"
        entry["outputs"] = [_record_file(path) for path in stage_output_paths(stage, run_root)]
        _write_manifest(run_root, manifest)

    manifest["run_status"] = run_status
    _write_manifest(run_root, manifest)
    return manifest


def _write_manifest(run_root: Path, manifest: dict[str, Any]) -> None:
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seal_run_manifest(run_root: Path) -> Path:
    """Write the sha256 sidecar that seals the immutable run manifest."""

    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise WorkflowBError(f"run manifest missing: {manifest_path}")
    sidecar = run_root / "run_manifest.sha256"
    sidecar.write_text(f"{_sha256(manifest_path)}  run_manifest.json\n", encoding="utf-8")
    return sidecar


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scvi-model", required=True, help="trained scVI model directory")
    parser.add_argument("--embedding-asset", required=True, help="verified PerturbGen embedding asset directory")
    parser.add_argument("--output-root", required=True, help="fresh run directory for the immutable manifest")
    parser.add_argument("--intervention-type", required=True, choices=("KO", "KD", "OE"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--norman-dir", help="directory containing the four GSE133344 10x files")
    source.add_argument("--geo-root", help="GEO root containing the GSE133344 directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--train-donors", default=None, help="comma-separated training donor IDs")
    parser.add_argument("--held-out-donors", default=None, help="comma-separated held-out donor IDs")
    parser.add_argument(
        "--require-donor-split",
        action="store_true",
        help="fail the train stage when no explicit donor split is provided (Workflow B default posture)",
    )
    parser.add_argument(
        "--stages",
        default=DEFAULT_STAGES,
        help=f"comma-separated subset of {','.join(STAGE_ORDER)} in canonical order, or 'all'",
    )
    parser.add_argument("--benchmark", default=None, help="frozen Gate-E benchmark CSV (required by the gate_e stage)")
    parser.add_argument("--old-vocabulary", default=None, help="old vocabulary JSON for Gate-E migration check")
    parser.add_argument("--new-vocabulary", default=None, help="new vocabulary JSON for Gate-E migration check")
    parser.add_argument("--davf-paired-results", default=None, help="paired old/new DAVF predictions CSV for Gate-E")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the manifest with planned argv only; do not execute stages",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    try:
        manifest = run_workflow_b(args)
        if not args.dry_run:
            seal_run_manifest(Path(args.output_root).expanduser().resolve())
    except WorkflowBError as exc:
        print(f"[run_workflow_b] {exc}", file=sys.stderr)
        return 1
    summary = {
        "run_directory": manifest["run_directory"],
        "run_status": manifest.get("run_status"),
        "stages": {name: record.get("status") for name, record in manifest["stages"].items()},
    }
    print(json.dumps(summary, ensure_ascii=False))
    if manifest.get("run_status") == "gate_failed":
        return GATE_E_FAIL_EXIT_CODE
    if manifest.get("run_status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
