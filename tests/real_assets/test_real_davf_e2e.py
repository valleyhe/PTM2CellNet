"""Real-asset acceptance for DAVF E2E finetune (F-02, TD-H2).

Gated behind ``PTM2CELLNET_RUN_REAL_ASSET_TESTS=1``. When enabled, this
test runs ``scripts/finetune_davf_e2e.py`` as a subprocess against a real
(or user-supplied) DAVF checkpoint + CSV, and records the subprocess
exit code, stdout/stderr tail, and wall time as evidence.

Configuration via env vars:

* ``PTM2CELLNET_DAVF_CHECKPOINT`` — path to a real DAVF checkpoint.
* ``PTM2CELLNET_DAVF_CSV`` — path to a real PTM-site CSV for finetuning.
* ``PTM2CELLNET_DAVF_EXTRA_ARGS`` — extra CLI args (e.g. ``--epochs 1``).

If the gate is unset, the test is SKIPPED with a clear reason.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.real_assets import real_assets_enabled, record_evidence


pytestmark = pytest.mark.skipif(
    not real_assets_enabled(),
    reason=(
        "DAVF E2E real-asset test is opt-in. Set "
        "PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 and provide "
        "PTM2CELLNET_DAVF_CHECKPOINT + PTM2CELLNET_DAVF_CSV to run a real "
        "subprocess finetune."
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_davf_e2e_real_subprocess_run(tmp_path):
    """Run finetune_davf_e2e.py as a subprocess against real assets."""
    ckpt = os.environ.get("PTM2CELLNET_DAVF_CHECKPOINT")
    csv = os.environ.get("PTM2CELLNET_DAVF_CSV")
    extra = os.environ.get("PTM2CELLNET_DAVF_EXTRA_ARGS", "")

    if not ckpt or not csv:
        pytest.skip(
            "PTM2CELLNET_DAVF_CHECKPOINT and PTM2CELLNET_DAVF_CSV must both be set "
            "for the DAVF E2E real-asset test."
        )

    script = _repo_root() / "scripts" / "finetune_davf_e2e.py"
    assert script.is_file(), f"{script} not found"

    output_dir = tmp_path / "davf_real_out"
    cmd = [
        sys.executable,
        str(script),
        "--checkpoint", ckpt,
        "--data", csv,
        "--output-dir", str(output_dir),
    ] + (extra.split() if extra else [])

    start = time.time()
    outcome = "pass"
    err: str | None = None
    tail_stdout = ""
    tail_stderr = ""
    returncode: int | None = None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("PTM2CELLNET_DAVF_TIMEOUT_S", "1800")),
            cwd=str(_repo_root()),
        )
        returncode = proc.returncode
        tail_stdout = proc.stdout[-2000:]
        tail_stderr = proc.stderr[-2000:]
        if returncode != 0:
            outcome = "fail"
            err = f"subprocess exited {returncode}"
            pytest.fail(
                f"DAVF E2E subprocess exited {returncode}.\nstderr tail:\n{tail_stderr}"
            )
    except subprocess.TimeoutExpired as exc_:
        outcome = "fail"
        err = f"timeout after {exc_.timeout}s"
        raise
    except Exception as exc_:  # pragma: no cover
        outcome = "fail"
        err = f"{type(exc_).__name__}: {exc_}"
        raise
    finally:
        record_evidence(
            "test_real_davf_e2e_subprocess",
            asset=f"checkpoint={ckpt}, csv={csv}",
            outcome=outcome,
            duration_s=time.time() - start,
            extra={
                "returncode": returncode,
                "stdout_tail": tail_stdout,
                "stderr_tail": tail_stderr,
                "error": err,
            },
        )
