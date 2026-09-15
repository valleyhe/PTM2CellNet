"""Resolve the exact token dimensions used by the external PerturbGen code."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PerturbGenDimensionError(ValueError):
    """Raised when tokenized PerturbGen data cannot yield valid dimensions."""


@dataclass(frozen=True)
class PerturbGenDimensions:
    """Unbuffered dimensions expected by Perturb/val.py."""

    tgt_vocab_size: int
    max_seq_length: int


_DIMENSION_PROBE = r"""
import json
import os
import sys
from datasets import load_from_disk


def iter_input_ids(dataset):
    if "input_ids" not in dataset.column_names:
        raise ValueError("tokenized dataset has no input_ids column")
    for row in dataset["input_ids"]:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("tokenized input_ids rows must be non-empty lists")
        yield row


def inspect_dataset(path):
    dataset = load_from_disk(path)
    max_token = -1
    max_length = 0
    for row in iter_input_ids(dataset):
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in row):
            raise ValueError("tokenized input_ids must contain non-negative integers")
        max_token = max(max_token, max(row))
        max_length = max(max_length, len(row))
    if max_token < 0 or max_length <= 0:
        raise ValueError(f"tokenized dataset is empty: {path}")
    return max_token, max_length


_, src_max_length = inspect_dataset(sys.argv[1])
target_root = sys.argv[2]
target_entries = sorted(
    os.path.join(target_root, name)
    for name in os.listdir(target_root)
    if name.endswith(".dataset")
)
if not target_entries:
    raise ValueError(f"no target .dataset entries found in {target_root}")

max_target_token = -1
max_length = src_max_length
for target_path in target_entries:
    target_max_token, target_max_length = inspect_dataset(target_path)
    max_target_token = max(max_target_token, target_max_token)
    max_length = max(max_length, target_max_length)

if max_target_token < 0:
    raise ValueError("target datasets contain no token IDs")
print(json.dumps({"tgt_vocab_size": max_target_token + 1, "max_seq_length": max_length}))
"""


def parse_dimension_probe_output(stdout: str) -> PerturbGenDimensions:
    """Parse and validate the single JSON record emitted by the probe."""

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise PerturbGenDimensionError("PerturbGen dimension probe returned no output")
    try:
        payload: Any = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise PerturbGenDimensionError("PerturbGen dimension probe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise PerturbGenDimensionError("PerturbGen dimension probe output must be an object")

    values: dict[str, int] = {}
    for name in ("tgt_vocab_size", "max_seq_length"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PerturbGenDimensionError(f"PerturbGen dimension probe returned invalid {name}: {value!r}")
        values[name] = value
    return PerturbGenDimensions(**values)


def derive_perturbgen_dimensions(
    external_python: str | Path,
    *,
    src_dataset: str | Path,
    tgt_dataset_folder: str | Path,
    cwd: str | Path,
    timeout_seconds: int = 300,
) -> PerturbGenDimensions:
    """Compute dimensions with the same tokenized datasets used downstream.

    The project process does not import perturbgen. The probe runs in the
    locked external environment and uses scalar maxima, never Python's
    lexicographic max over ragged lists from the upstream fallback branch.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    command = [
        str(Path(external_python).expanduser().resolve(strict=True)),
        "-c",
        _DIMENSION_PROBE,
        str(Path(src_dataset).expanduser().resolve(strict=True)),
        str(Path(tgt_dataset_folder).expanduser().resolve(strict=True)),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(cwd).expanduser().resolve(strict=True)),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PerturbGenDimensionError(f"failed to execute PerturbGen dimension probe: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PerturbGenDimensionError(
            "PerturbGen dimension probe failed"
            + (f": {detail}" if detail else f" with return code {completed.returncode}")
        )
    return parse_dimension_probe_output(completed.stdout)
