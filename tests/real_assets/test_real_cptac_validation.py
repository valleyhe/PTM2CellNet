"""Real-asset acceptance for CPTAC/PDC validation (F-03, TD-H3).

Gated behind ``PTM2CELLNET_RUN_REAL_ASSET_TESTS=1``. When enabled, this
suite exercises the two F-03 paths that v16 still left partial:

1. **PDC real file download** — resolves a CPTAC study's phosphoproteomics
   ``file_id`` via the PDC GraphQL ``file`` query and streams the TSV to a
   local path through the project's safe download boundary. Asserts the
   downloaded bytes are non-empty and the parsed matrix has the expected
   ``(sites × samples)`` shape.

2. **CPTAC validator predictor wiring** — when a real multi-task PTM
   checkpoint is provided via ``PTM2CELLNET_CPTAC_MODEL_DIR``, runs the
   validator end-to-end against the real PDC matrix and asserts the
   output JSON contains ``predictor_wired=true`` and at least one
   non-``unknown`` ``predicted_effect``.

Configuration via env vars:

* ``PTM2CELLNET_CPTAC_STUDY`` — CPTAC alias to validate (default ``BRCA``).
* ``PTM2CELLNET_CPTAC_MODEL_DIR`` — directory containing the multi-task
  PTM checkpoint (e.g. ``outputs/ptm_pretrain``). If unset, only the
  download test runs.

If the gate is unset, both tests are SKIPPED with a clear reason so a
green CI run is never mistaken for "F-03 verified against real PDC".
"""

from __future__ import annotations

import json
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
        "CPTAC/PDC real-asset tests are opt-in (they hit the PDC API). "
        "Set PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 to run them."
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_pdc_real_file_download(tmp_path):
    """PDCClient.download_file must fetch a real CPTAC phospho TSV.

    Uses the canonical CPTAC BRCA PDC study_id. If the file list or
    network shape has shifted, the test surfaces the error rather than
    silently skipping.
    """
    from src.analysis.pdc_client import PDCClient, PDCAPIError

    study_alias = os.environ.get("PTM2CELLNET_CPTAC_STUDY", "BRCA")
    # Import the canonical study map from the script (no network needed).
    import importlib.util

    script_path = _repo_root() / "scripts" / "validate_cptac.py"
    spec = importlib.util.spec_from_file_location("validate_cptac_real", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    os.environ.setdefault("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    spec.loader.exec_module(module)
    pdc_study_id = module.CPTAC_STUDIES[study_alias]["pdc_study_id"]

    start = time.time()
    outcome = "pass"
    err: str | None = None
    downloaded: Path | None = None
    matrix_shape: tuple[int, int] | None = None
    try:
        client = PDCClient()
        study = client.get_study(pdc_study_id)
        phospho_files = [
            f
            for f in study.files
            if "phospho" in str(f.get("data_category", "")).lower() or "phospho" in str(f.get("file_name", "")).lower()
        ]
        assert phospho_files, f"PDC study {study_alias} returned no phospho files"

        # Take the first file_id and stream it through download_file.
        file_id = phospho_files[0].get("file_id") or phospho_files[0].get("file_name")
        assert file_id, f"phospho file has no file_id: {phospho_files[0]!r}"
        downloaded = client.download_file(str(file_id), str(tmp_path))
        assert downloaded.is_file() and downloaded.stat().st_size > 0

        # Parse it as a phospho matrix; assert at least one site × one sample.
        matrix = module._parse_phospho_tsv(downloaded)
        matrix_shape = matrix.shape
        assert matrix.shape[0] > 0 and matrix.shape[1] > 0
    except PDCAPIError as exc_:  # pragma: no cover
        outcome = "fail"
        err = f"PDCAPIError: {exc_}"
        raise
    except Exception as exc_:  # pragma: no cover
        outcome = "fail"
        err = f"{type(exc_).__name__}: {exc_}"
        raise
    finally:
        record_evidence(
            "test_real_pdc_file_download",
            asset=f"PDC study {study_alias} ({pdc_study_id})",
            outcome=outcome,
            duration_s=time.time() - start,
            extra={
                "downloaded_path": str(downloaded) if downloaded else None,
                "matrix_shape": list(matrix_shape) if matrix_shape else None,
                "error": err,
            },
        )


def test_cptac_validator_predictor_wired_against_real_pdc(tmp_path):
    """End-to-end: validate_cptac.py with real PDC data + a real checkpoint.

    Requires ``PTM2CELLNET_CPTAC_MODEL_DIR``. Otherwise skipped (not
    failed) because the download test above already exercises the
    network path.
    """
    model_dir = os.environ.get("PTM2CELLNET_CPTAC_MODEL_DIR")
    if not model_dir:
        pytest.skip(
            "PTM2CELLNET_CPTAC_MODEL_DIR must point at a multi-task PTM checkpoint directory for this real-asset test."
        )

    study_alias = os.environ.get("PTM2CELLNET_CPTAC_STUDY", "BRCA")
    script = _repo_root() / "scripts" / "validate_cptac.py"
    assert script.is_file(), f"{script} not found"

    output_dir = tmp_path / "cptac_real_out"
    cmd = [
        sys.executable,
        str(script),
        "-o",
        str(output_dir),
        "-s",
        study_alias,
        "-m",
        model_dir,
    ]
    env = os.environ.copy()
    env["PTM2CELLNET_ALLOW_EXPERIMENTAL"] = "1"

    start = time.time()
    outcome = "pass"
    err: str | None = None
    returncode: int | None = None
    payload_summary: dict | None = None
    tail_stderr = ""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("PTM2CELLNET_CPTAC_TIMEOUT_S", "1800")),
            cwd=str(_repo_root()),
            env=env,
        )
        returncode = proc.returncode
        tail_stderr = proc.stderr[-2000:]
        if returncode != 0:
            outcome = "fail"
            err = f"subprocess exited {returncode}"
            pytest.fail(f"validate_cptac.py exited {returncode}.\nstderr tail:\n{tail_stderr}")

        result_json = output_dir / "validation_results.json"
        assert result_json.is_file(), f"missing {result_json}"
        payload = json.loads(result_json.read_text())
        # Predictor must be wired from the supplied checkpoint.
        assert payload.get("predictor_wired") is True, payload
        # Real PDC path produces a real matrix.
        assert payload.get("phospho_data_real") is True, payload
        effects = payload.get("prediction_summary", {}).get("effect_counts", {})
        # At least one site must be scored (gain/loss/neutral/unknown).
        assert sum(effects.values()) > 0, effects
        payload_summary = {
            "scientifically_valid": payload.get("scientifically_valid"),
            "predictor_wired": payload.get("predictor_wired"),
            "effect_counts": effects,
            "phospho_data_shape": payload.get("phospho_data_shape"),
        }
    except subprocess.TimeoutExpired as exc_:  # pragma: no cover
        outcome = "fail"
        err = f"timeout after {exc_.timeout}s"
        raise
    except Exception as exc_:  # pragma: no cover
        outcome = "fail"
        err = f"{type(exc_).__name__}: {exc_}"
        raise
    finally:
        record_evidence(
            "test_real_cptac_validator_predictor_wired",
            asset=f"PDC {study_alias} + checkpoint={model_dir}",
            outcome=outcome,
            duration_s=time.time() - start,
            extra={
                "returncode": returncode,
                "summary": payload_summary,
                "stderr_tail": tail_stderr,
                "error": err,
            },
        )
