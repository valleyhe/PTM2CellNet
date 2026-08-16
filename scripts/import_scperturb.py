#!/usr/bin/env python3
"""Register the scPerturb h5ad snapshots into PTM2CellNet.

Consumes the 30 study h5ad files downloaded to ``data/raw/scperturb/``
(scPerturb consortium release; ~21 GB) and produces a *registration
report* — a structural probe per study (cell x gene shape, obs/var
columns, perturbation-identity columns, gene-id type) written to
``scperturb_study_manifest.json``.

This script closes the manifest gap recorded in
``data/manifests/datasets.yaml`` (``scperturb`` was ``controlled`` with
``loader: null`` and ``path: null`` although the files were on disk).
It deliberately does NOT parse expression matrices end to end — the
full ``load_controlled_dataset`` pipeline that turns each study into
standardized bundles is a separate step (F-05 in the 2026-08-16
analysis); this registration makes every snapshot addressable,
hash-verifiable and inspectable.

Design notes
------------
* ``anndata`` backed mode — only metadata is read, expression matrices
  are never loaded into memory.
* Fail-soft per file: a corrupt h5ad is reported with its error and the
  remaining studies still register (mirrors the GSE90546 probe pattern).
* Pure read-only on ``data/raw/``; writes only under the output dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_RAW_ROOT = "data/raw/scperturb"
DEFAULT_OUTPUT = "data/processed/scperturb"
STUDY_MANIFEST_NAME = "scperturb_study_manifest.json"

#: obs columns that indicate the perturbation identity (any of them present
#: satisfies ``perturbation_metadata_present`` in datasets.yaml).
PERTURBATION_COLUMN_MARKERS = (
    "perturbation",
    "target",
    "guide",
    "guide_id",
    "grna",
    "condition",
    "identity",
)


class ScPerturbImportError(RuntimeError):
    """Raised when the scPerturb root directory is missing or empty."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_study(path: Path, *, max_preview: int = 5) -> Dict[str, Any]:
    """Probe one h5ad study in backed mode (metadata only)."""
    import anndata

    ad = anndata.read_h5ad(path, backed="r")
    try:
        obs_columns = list(ad.obs.columns)
        var_columns = list(ad.var.columns)
        perturbation_cols = [c for c in obs_columns if c.lower() in PERTURBATION_COLUMN_MARKERS]
        n_unique_perturbations: Optional[int] = None
        if perturbation_cols:
            try:
                n_unique_perturbations = int(ad.obs[perturbation_cols[0]].nunique(dropna=True))
            except Exception:  # noqa: BLE001 - exotic column types degrade gracefully
                n_unique_perturbations = None
        return {
            "file": path.name,
            "sha256": _sha256_file(path),
            "n_cells": int(ad.n_obs),
            "n_genes": int(ad.n_vars),
            "obs_columns": obs_columns,
            "var_columns": var_columns,
            "var_index_head": [str(v) for v in ad.var.index[:max_preview]],
            "perturbation_columns": perturbation_cols,
            "n_unique_perturbations": n_unique_perturbations,
        }
    finally:
        try:
            ad.file.close()
        except Exception:  # noqa: BLE001 - closing is best-effort
            pass


def build_registration_report(raw_root: Path, studies: Sequence[Path]) -> Dict[str, Any]:
    """Probe every study and aggregate the registration report."""
    probes: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    total_cells = 0
    total_genes = 0
    for path in sorted(studies):
        try:
            probe = probe_study(path)
        except Exception as exc:  # noqa: BLE001 - fail-soft per file
            errors.append({"file": path.name, "error": str(exc)})
            continue
        probes.append(probe)
        total_cells += probe["n_cells"]
        total_genes += probe["n_genes"]
    return {
        "schema_version": "ptm2cellnet.scperturb.registration.v1",
        "summary": {
            "n_studies_registered": len(probes),
            "n_studies_failed": len(errors),
            "total_cells": total_cells,
            "total_genes": total_genes,
            "perturbation_metadata_present": sum(1 for p in probes if p["perturbation_columns"]),
        },
        "studies": probes,
        "errors": errors,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="scPerturb 下载根目录")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出目录")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅探测前 N 个研究（调试用）",
    )
    args = parser.parse_args(argv)

    raw_root = Path(args.raw_root)
    if not raw_root.is_dir():
        print(f"[scperturb] 注册失败：{raw_root} 不存在", file=sys.stderr)
        return 1
    studies = sorted(raw_root.glob("*.h5ad"))
    if args.limit is not None:
        studies = studies[: args.limit]
    if not studies:
        print(f"[scperturb] 注册失败：{raw_root} 下无 h5ad 文件", file=sys.stderr)
        return 1

    report = build_registration_report(raw_root, studies)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / STUDY_MANIFEST_NAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "report": str(report_path),
                "n_studies_registered": report["summary"]["n_studies_registered"],
                "n_studies_failed": report["summary"]["n_studies_failed"],
                "total_cells": report["summary"]["total_cells"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
