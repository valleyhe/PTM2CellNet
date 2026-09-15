#!/usr/bin/env python
"""Collect M0 environment evidence for the isolated PerturbGen environment.

Run inside the ``perturbgen`` conda env (Python 3.11). Writes a JSON evidence
file covering the four items required by project_repair_report_20260822.md §5
(2026-08-23 window): python version, locked source commit, GPU/CUDA, frozen
dependency list — plus offline-import checks for the HF/evaluate stack.

Usage:
    /home/scu/anaconda3/envs/perturbgen/bin/python \
        scripts/collect_perturbgen_env_evidence.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PERTURBGEN_SRC = REPO_ROOT / "ref" / "Perturbgen-src"
OUTPUT_PATH = REPO_ROOT / "outputs" / "perturbgen" / ("env_evidence_" + _dt.date.today().strftime("%Y%m%d") + ".json")

# Force offline semantics for the import checks: M0 requires that the stack
# initializes without network access.
for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[var] = "1"


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"available": False}
    return {"available": True, "nvidia_smi": out}


def _import_check(module: str) -> dict:
    try:
        mod = __import__(module)
        return {"ok": True, "version": getattr(mod, "__version__", None)}
    except Exception as exc:  # noqa: BLE001 - evidence must record failures
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    import torch

    evidence = {
        "collected_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "perturbgen_source": {
            "path": str(PERTURBGEN_SRC),
            "commit": _git_commit(PERTURBGEN_SRC),
        },
        "gpu": _gpu_info(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        },
        "offline_imports": {
            # HF/evaluate components must initialize under offline semantics.
            name: _import_check(name)
            for name in (
                "transformers",
                "datasets",
                "evaluate",
                "tokenizers",
                "safetensors",
                "anndata",
                "scanpy",
                "h5py",
                "pytorch_lightning",
            )
        },
        "perturbgen_import": _import_check("perturbgen"),
    }

    # pip freeze of the isolated env (dependency manifest evidence).
    pip_bin = Path(sys.executable).with_name("pip")
    try:
        evidence["pip_freeze"] = subprocess.run(
            [str(pip_bin), "freeze"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        evidence["pip_freeze_error"] = str(exc)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(evidence, indent=2, ensure_ascii=False))

    ok = (
        evidence["python"]["version"].startswith("3.11")
        and evidence["perturbgen_import"]["ok"]
        and all(v["ok"] for v in evidence["offline_imports"].values())
    )
    print(
        json.dumps(
            {
                "evidence_file": str(OUTPUT_PATH),
                "python": evidence["python"]["version"],
                "commit": evidence["perturbgen_source"]["commit"],
                "gpu": evidence["gpu"].get("nvidia_smi"),
                "torch_cuda_available": evidence["torch"]["cuda_available"],
                "offline_imports_ok": all(v["ok"] for v in evidence["offline_imports"].values()),
                "perturbgen_import_ok": evidence["perturbgen_import"]["ok"],
                "overall": "OK" if ok else "INCOMPLETE",
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
