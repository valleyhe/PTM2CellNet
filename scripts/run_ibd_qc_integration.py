#!/usr/bin/env python3
"""Build IBD metadata, run sample-wise QC, and optionally run scVI integration."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _ensure_project_root() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("metadata", "qc", "integrate", "all"),
        default="metadata",
        help="Workflow stage to run (default: metadata).",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ibd"))
    parser.add_argument(
        "--scope",
        choices=("core", "validation", "core_plus_validation", "all"),
        default="core",
        help="QC scope; core is the documented discovery atlas.",
    )
    parser.add_argument(
        "--run-doublet",
        action="store_true",
        help="Run Scrublet separately for each sample if the optional package is installed.",
    )
    parser.add_argument("--max-cells-per-sample", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser.parse_args(argv)


def _metadata_paths(output_dir: Path):
    return output_dir / "metadata_master.tsv", output_dir / "sample_inclusion.tsv"


def _run_metadata(args):
    from src.analysis.ibd_dataset import build_metadata_master, write_metadata_outputs

    metadata = build_metadata_master(args.raw_root)
    paths = write_metadata_outputs(metadata, args.output_dir)
    logging.info("Metadata written: %s, %s, %s", *paths)
    return paths[0]


def _ensure_metadata(args) -> Path:
    master_path, _ = _metadata_paths(args.output_dir)
    if not master_path.exists():
        return _run_metadata(args)
    return master_path


def _run_qc(args):
    from src.analysis.ibd_qc import load_and_qc, write_qc_outputs

    metadata_path = _ensure_metadata(args)
    adata, summary = load_and_qc(
        metadata_path,
        raw_root=args.raw_root,
        scope=args.scope,
        run_doublet=args.run_doublet,
    )
    prefix = f"ibd_{args.scope}_qc"
    h5ad_path, summary_path = write_qc_outputs(
        adata,
        summary,
        output_dir=args.output_dir,
        prefix=prefix,
    )
    logging.info(
        "QC completed: %d cells x %d genes; outputs=%s, %s",
        adata.n_obs,
        adata.n_vars,
        h5ad_path,
        summary_path,
    )
    return h5ad_path


def _run_integration(args):
    import anndata as ad

    from src.analysis.ibd_qc import integrate_scvi

    qc_path = args.output_dir / f"ibd_{args.scope}_qc.h5ad"
    if not qc_path.exists():
        raise FileNotFoundError(
            f"QC object not found: {qc_path}; run --stage qc before --stage integrate"
        )
    qc_adata = ad.read_h5ad(qc_path)
    integrated, model = integrate_scvi(
        qc_adata,
        max_cells_per_sample=args.max_cells_per_sample,
        max_epochs=args.epochs,
        seed=args.seed,
    )
    integrated_path = args.output_dir / f"ibd_{args.scope}_scvi.h5ad"
    model_dir = args.output_dir / f"ibd_{args.scope}_scvi_model"
    integrated.write_h5ad(integrated_path, compression="gzip")
    model.save(model_dir, overwrite=True, save_anndata=False)
    logging.info(
        "scVI integration completed: %d cells x %d genes; outputs=%s, %s",
        integrated.n_obs,
        integrated.n_vars,
        integrated_path,
        model_dir,
    )
    return integrated_path


def main(argv=None) -> int:
    _ensure_project_root()
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.stage == "metadata":
        _run_metadata(args)
    elif args.stage == "qc":
        _run_qc(args)
    elif args.stage == "integrate":
        _ensure_metadata(args)
        _run_integration(args)
    else:
        _run_metadata(args)
        _run_qc(args)
        _run_integration(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
