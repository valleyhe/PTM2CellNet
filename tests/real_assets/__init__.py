"""Opt-in real-assets acceptance suite (TD-H2, F-01, F-02, F-05).

These tests do **NOT** run by default. They are gated behind the
``PTM2CELLNET_RUN_REAL_ASSET_TESTS=1`` environment variable so that CI and
local dev environments never accidentally hit the network, download
multi-GB weights, or require a GPU.

Purpose
-------

The default test suite (``tests/unit``, ``tests/integration``, ``tests/e2e``)
proves the code is internally correct using mocks, stubs, and synthetic data.
That is necessary but not sufficient for production claims. These
real-assets tests close that gap:

* **F-01 / ESM-3** — actually load ``esm3_sm_open_v1`` weights (or a
  user-supplied local checkpoint) and run a forward pass on real protein
  sequences.
* **F-02 / DAVF E2E** — actually run the ``finetune_davf_e2e.py`` script
  as a subprocess against a real (or user-supplied) DAVF checkpoint + CSV.
* **F-05 / external services** — actually hit UniProt / KEGG / Reactome
  endpoints and assert non-fallback responses.

How to run
----------

::

    # All real-assets tests (requires network + sufficient disk/GPU):
    PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 pytest tests/real_assets -v

    # Just ESM-3:
    PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 \\
        PTM2CELLNET_ESM3_CHECKPOINT=/path/to/esm3_sm_open_v1.pth \\
        pytest tests/real_assets/test_real_esm3.py -v

    # Just DAVF E2E:
    PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 \\
        PTM2CELLNET_DAVF_CHECKPOINT=/path/to/davf.ckpt \\
        PTM2CELLNET_DAVF_CSV=/path/to/sites.csv \\
        pytest tests/real_assets/test_real_davf_e2e.py -v

Evidence capture
----------------

Each test writes a JSON record into ``outputs/real_assets/<test>.json``
containing the host, the asset path/source, cache hit/miss, wall time, and
the pass/fail outcome. This makes the "we ran against real assets on
<date>" claim auditable rather than implicit.
"""

import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any, Optional


GATE_ENV = "PTM2CELLNET_RUN_REAL_ASSET_TESTS"


def real_assets_enabled() -> bool:
    """True only when the operator explicitly opted in via the gate env var."""
    return os.environ.get(GATE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _output_dir() -> Path:
    out = Path("outputs") / "real_assets"
    out.mkdir(parents=True, exist_ok=True)
    return out


def record_evidence(
    test_name: str,
    *,
    asset: str,
    outcome: str,
    duration_s: float,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Write a JSON evidence record for a real-assets test run.

    Kept intentionally small so it can be appended to a release artifact
    bundle and audited later.
    """
    record: dict[str, Any] = {
        "test": test_name,
        "asset": asset,
        "outcome": outcome,
        "duration_s": round(duration_s, 3),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "timestamp_unix": int(time.time()),
    }
    if extra:
        record["extra"] = extra
    target = _output_dir() / f"{test_name}.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["GATE_ENV", "real_assets_enabled", "record_evidence"]
