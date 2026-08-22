#!/usr/bin/env python3
"""Run or inspect the isolated PerturbGen six-stage pipeline."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.config_builder import (  # noqa: E402
    STAGE_ORDER,
    build_stage_plans,
    load_pipeline_config,
)
from src.integration.perturbgen.runner import PerturbGenRunner  # noqa: E402


def _apply_path(config: dict[str, Any], path: str | None) -> None:
    if path is None:
        return
    perturb = config["stages"]["perturb"]["perturb_config"]
    trainer = perturb["trainer"]
    datamodule = perturb["datamodule"]
    if path == "source_intervention":
        trainer["perturbation_sequence"] = ["src"]
        trainer.pop("pert_tps", None)
        datamodule.pop("pert_tps", None)
    elif path == "within_state":
        trainer["perturbation_sequence"] = ["tgt"]
        if not trainer.get("pert_tps") or not datamodule.get("pert_tps"):
            raise ValueError("within_state requires trainer/datamodule pert_tps")
    else:  # argparse choices make this defensive branch unreachable from CLI.
        raise ValueError(f"unknown path: {path}")


def _serialize_result(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return {
            key: str(item) if isinstance(item, Path) else item
            for key, item in value.__dict__.items()
        }
    return value


def _build_selected_plans(
    config: dict[str, Any], selected: set[str], path: str | None
):
    if path != "both":
        _apply_path(config, path)
        return tuple(
            plan for plan in build_stage_plans(config, project_root=PROJECT_ROOT)
            if plan.name in selected
        )

    base_plans = build_stage_plans(config, project_root=PROJECT_ROOT)
    path_plans = []
    for path_name in ("source_intervention", "within_state"):
        path_config = deepcopy(config)
        _apply_path(path_config, path_name)
        stage = path_config["stages"]["perturb"]
        stage["output_subdir"] = f"perturb/{path_name}"
        stage["perturb_config"]["trainer"]["output_dir"] = str(
            Path(path_config["pipeline"]["output_root"]) / "perturb" / path_name / "results"
        )
        perturb_plan = build_stage_plans(path_config, project_root=PROJECT_ROOT)[3]
        path_plans.append(replace(perturb_plan, name=path_name))

    output = []
    for plan in base_plans:
        if plan.name not in selected:
            continue
        if plan.name == "perturb":
            output.extend(path_plans)
        else:
            output.append(plan)
    return tuple(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=list(STAGE_ORDER))
    parser.add_argument(
        "--path",
        choices=("source_intervention", "within_state", "both"),
        help="Materialize the perturb stage as PerturbGen src or tgt+pert_tps",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-lock-file", type=Path, default=Path("outputs/perturbgen/.gpu.lock"))
    args = parser.parse_args(argv)

    config = load_pipeline_config(args.config)
    selected = set(args.stages)
    plans = _build_selected_plans(config, selected, args.path)
    runner = PerturbGenRunner(gpu_lock_file=args.gpu_lock_file)
    results = runner.run_pipeline(plans, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps([_serialize_result(item) for item in results], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
