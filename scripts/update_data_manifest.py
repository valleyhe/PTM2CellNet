#!/usr/bin/env python3
"""Register a user-provided local data snapshot and its SHA-256 digest.

This command only edits the YAML inventory. It never downloads, uploads or
copies the source file, which keeps controlled data outside version control.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manifest import DataManifestError, get_dataset, sha256_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="dataset id from the manifest")
    parser.add_argument("--path", type=Path, required=True, help="local snapshot path")
    parser.add_argument("--format", required=True, help="format label, e.g. csv or h5ad")
    parser.add_argument("--required", action="store_true", help="mark this local file as required for strict checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    if not args.path.is_file():
        print(f"snapshot not found: {args.path}", file=sys.stderr)
        return 2

    try:
        payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise DataManifestError("manifest root must be a mapping")
        get_dataset(payload, args.dataset)
    except (yaml.YAMLError, DataManifestError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        relative_path = args.path.resolve().relative_to(PROJECT_ROOT.resolve())
        recorded_path = relative_path.as_posix()
    except ValueError:
        recorded_path = str(args.path.resolve())

    for dataset in payload["datasets"]:
        if dataset.get("id") != args.dataset:
            continue
        files = dataset.setdefault("files", [])
        if not isinstance(files, list):
            files = [files]
            dataset["files"] = files
        replacement: Dict[str, Any] = {
            "path": recorded_path,
            "format": args.format,
            "required": bool(args.required),
            "sha256": sha256_file(args.path),
            "bytes": args.path.stat().st_size,
            "recorded_at": date.today().isoformat(),
        }
        matching_index = next(
            (
                index
                for index, entry in enumerate(files)
                if isinstance(entry, dict) and entry.get("path") == recorded_path
            ),
            None,
        )
        if matching_index is None:
            files.append(replacement)
        else:
            files[matching_index] = replacement
        dataset["status"] = "local"
        break

    args.manifest.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"registered {args.dataset}: {recorded_path} sha256={replacement['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
