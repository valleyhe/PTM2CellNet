#!/usr/bin/env python3
"""Validate the versioned project data inventory and optional local snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manifest import DataManifestError, load_manifest, validate_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/datasets.yaml",
    )
    parser.add_argument("--check-files", action="store_true", help="check registered local snapshot paths")
    parser.add_argument("--verify-hashes", action="store_true", help="require and verify SHA-256 for present files")
    parser.add_argument("--strict", action="store_true", help="treat warnings as validation errors")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "activate a named asset profile declared under manifest.profiles "
            "(e.g. standard_training, cross_scale_training, cross_scale_inference). "
            "Required datasets in the profile are enforced as mandatory: missing "
            "path, absent file (with --check-files) or hash drift (with --verify-hashes) "
            "become hard errors regardless of per-file required flags."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except (FileNotFoundError, DataManifestError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, ensure_ascii=False, indent=2))
        return 2

    report = validate_manifest(
        manifest,
        manifest_path=args.manifest,
        check_files=args.check_files,
        verify_hashes=args.verify_hashes,
        strict_warnings=args.strict,
        profile=args.profile,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
