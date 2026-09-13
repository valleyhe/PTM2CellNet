"""Explicit train / held-out donor split for DAVF training and E2E tokenise.

Frozen M6 already requires ≥2 training donors and ≥3 held-out donors that are
disjoint.  Training and tokenise previously did not receive those lists, so a
passing Gate-0 (≥3 shared donors) could not prove held-out cells stayed out of
DAVF/PerturbGen training.  This module is the shared validator and SHA binder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DONOR_SPLIT_SCHEMA_VERSION = "ptm2cellnet.donor_split/v1"
MIN_TRAIN_DONORS = 2
MIN_HELD_OUT_DONORS = 3


class DonorSplitError(ValueError):
    """Raised when a donor split is missing, leaked, or fails frozen binding."""


@dataclass(frozen=True)
class DonorSplit:
    """Canonical, hash-stable train / held-out donor lists."""

    train_donors: tuple[str, ...]
    held_out_donors: tuple[str, ...]
    sha256: str
    schema_version: str = DONOR_SPLIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "train_donors": list(self.train_donors),
            "held_out_donors": list(self.held_out_donors),
            "sha256": self.sha256,
        }


def parse_donor_list(value: str | Sequence[str] | None, *, name: str) -> tuple[str, ...]:
    """Parse a comma-separated string or sequence into unique non-empty labels."""

    if value is None:
        raise DonorSplitError(f"{name} is required")
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (bytes, bytearray)):
        raise DonorSplitError(f"{name} must be a string or sequence of strings")
    else:
        items = [str(item).strip() for item in value]
    if not items or any(not item for item in items):
        raise DonorSplitError(f"{name} must not contain empty donor labels")
    if len(items) != len(set(items)):
        raise DonorSplitError(f"{name} must be unique")
    return tuple(items)


def canonical_donor_split_bytes(train_donors: Sequence[str], held_out_donors: Sequence[str]) -> bytes:
    """Stable UTF-8 JSON used for SHA-256 identity of a split."""

    payload = {
        "schema_version": DONOR_SPLIT_SCHEMA_VERSION,
        "train_donors": list(train_donors),
        "held_out_donors": list(held_out_donors),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def donor_split_sha256(train_donors: Sequence[str], held_out_donors: Sequence[str]) -> str:
    return hashlib.sha256(canonical_donor_split_bytes(train_donors, held_out_donors)).hexdigest()


def build_donor_split(
    train_donors: str | Sequence[str],
    held_out_donors: str | Sequence[str],
) -> DonorSplit:
    """Validate M6-compatible disjoint lists and return a SHA-bound split."""

    train = parse_donor_list(train_donors, name="train_donors")
    held_out = parse_donor_list(held_out_donors, name="held_out_donors")
    if len(train) < MIN_TRAIN_DONORS:
        raise DonorSplitError(f"at least {MIN_TRAIN_DONORS} training donors are required")
    if len(held_out) < MIN_HELD_OUT_DONORS:
        raise DonorSplitError(f"at least {MIN_HELD_OUT_DONORS} held-out donors are required")
    overlap = sorted(set(train) & set(held_out))
    if overlap:
        raise DonorSplitError(f"donor leakage: donors appear in both train and held-out splits: {overlap}")
    train_sorted = tuple(sorted(train))
    held_sorted = tuple(sorted(held_out))
    return DonorSplit(
        train_donors=train_sorted,
        held_out_donors=held_sorted,
        sha256=donor_split_sha256(train_sorted, held_sorted),
    )


def load_donor_split(payload: Mapping[str, Any]) -> DonorSplit:
    """Load a previously written split and recompute the SHA."""

    if not isinstance(payload, Mapping):
        raise DonorSplitError("donor split payload must be a mapping")
    schema = payload.get("schema_version")
    if schema != DONOR_SPLIT_SCHEMA_VERSION:
        raise DonorSplitError(f"unsupported donor split schema_version {schema!r}")
    split = build_donor_split(payload.get("train_donors", ()), payload.get("held_out_donors", ()))
    recorded = payload.get("sha256")
    if recorded is not None and str(recorded) != split.sha256:
        raise DonorSplitError(
            f"donor split sha256 drifted: payload has {recorded!r}, canonical is {split.sha256}"
        )
    return split


def bind_frozen_donor_split(split: DonorSplit, frozen_manifest: Any) -> None:
    """Require exact list identity with a frozen cohort manifest."""

    train = tuple(frozen_manifest.train_donors)
    held_out = tuple(frozen_manifest.held_out_donors)
    frozen = build_donor_split(train, held_out)
    if split.train_donors != frozen.train_donors or split.held_out_donors != frozen.held_out_donors:
        raise DonorSplitError(
            "donor split does not match frozen cohort manifest train/held-out lists"
        )
    if split.sha256 != frozen.sha256:
        raise DonorSplitError(
            f"donor split sha256 {split.sha256} does not match frozen manifest sha256 {frozen.sha256}"
        )


def optional_donor_split_from_args(
    *,
    train_donors: str | Sequence[str] | None,
    held_out_donors: str | Sequence[str] | None,
    require: bool = False,
) -> DonorSplit | None:
    """Build a split when either list is provided, or require both."""

    present = train_donors is not None or held_out_donors is not None
    if require and not present:
        raise DonorSplitError("train_donors and held_out_donors are required")
    if not present:
        return None
    if train_donors is None or held_out_donors is None:
        raise DonorSplitError("train_donors and held_out_donors must be supplied together")
    return build_donor_split(train_donors, held_out_donors)


def write_donor_split(split: DonorSplit, path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(split.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved
