#!/usr/bin/env python3
"""Validate local cross-scale pLM directories without loading model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.plm_assets import DEFAULT_LOCAL_PLM_DIRS, resolve_local_plm_assets  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path("data/weights/plm"))
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        metavar="BACKBONE=DIRECTORY",
        help="覆盖默认目录映射；可重复指定",
    )
    parser.add_argument(
        "--required-backbone",
        action="append",
        default=[],
        help="必需 backbone；缺省时所有配置项均为必需",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _parse_model_dirs(values: list[str]) -> dict[str, str]:
    result = dict(DEFAULT_LOCAL_PLM_DIRS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model-dir 必须使用 BACKBONE=DIRECTORY 格式: {value}")
        name, directory = value.split("=", 1)
        if not name.strip() or not directory.strip():
            raise ValueError(f"--model-dir 名称和目录不能为空: {value}")
        result[name.strip().lower()] = directory.strip()
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model_dirs = _parse_model_dirs(args.model_dir)
        report = resolve_local_plm_assets(
            args.asset_root,
            model_dirs=model_dirs,
            required_backbones=args.required_backbone or None,
        )
    except (OSError, ValueError) as exc:
        report = {"ok": False, "errors": [str(exc)]}
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
