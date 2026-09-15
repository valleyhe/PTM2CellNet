"""Real-asset acceptance for the frozen between_donor M6 cohort contract.

Gated behind ``PTM2CELLNET_RUN_REAL_ASSET_TESTS=1``. When enabled, this test
loads the real frozen manifest for the GSE174367 AD case-control cohort
(``pairing=between_donor``, EX cell type), re-verifies the cohort h5ad
sha256 on load, and replays ``_validate_frozen_cohort_asset`` against the
real 844 MB cohort file so any drift in the cohort, the split, or the frozen
contract hard-fails before the six-stage GPU run consumes it.

The frozen assets live under ``outputs/perturbgen/frozen/`` which is
git-ignored, so the test skips with an explicit reason when the local
manifest is absent (fresh clone without the real data assets).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.integration.perturbgen.frozen_cohort import (
    _validate_frozen_cohort_asset,
    load_frozen_manifest,
)
from tests.real_assets import real_assets_enabled, record_evidence


FROZEN_ROOT = Path(
    os.environ.get(
        "PTM2CELLNET_FROZEN_MANIFEST",
        "outputs/perturbgen/frozen/20260914_gse174367_ex/manifest.json",
    )
)

pytestmark = [
    pytest.mark.real_assets,
    pytest.mark.skipif(
        not real_assets_enabled(),
        reason="set PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 to run the real frozen-cohort acceptance",
    ),
    pytest.mark.skipif(
        not FROZEN_ROOT.exists(),
        reason=f"local frozen manifest not found: {FROZEN_ROOT}",
    ),
]


def test_real_frozen_between_donor_manifest_validates_against_cohort() -> None:
    started = time.perf_counter()
    manifest = load_frozen_manifest(FROZEN_ROOT)
    assert manifest.pairing == "between_donor"
    assert manifest.cell_type == "EX"
    expected_donors = set(manifest.train_donors) | set(manifest.held_out_donors)
    assert len(manifest.train_donors) == 12
    assert len(manifest.held_out_donors) == 6
    assert {c.intervention_type for c in manifest.candidates} == {"KO"}
    assert all(c.matched_nulls >= 99 for c in manifest.candidates)

    ok, details, issues = _validate_frozen_cohort_asset(manifest)
    duration = time.perf_counter() - started
    assert ok, f"real frozen cohort validation failed: {issues}"
    assert issues == []
    assert details["n_cells"] > 0
    covered = set(details["target_donors"])
    assert covered == expected_donors
    normal_donors = set(details["state_donors"]["normal"])
    disease_donors = set(details["state_donors"]["disease"])
    assert not (normal_donors & disease_donors), "between_donor requires state-disjoint donor groups"
    assert len(normal_donors) >= 3 and len(disease_donors) >= 3

    record_evidence(
        "real_frozen_between_donor_cohort",
        asset=str(FROZEN_ROOT),
        outcome="pass",
        duration_s=duration,
        extra={
            "pairing": manifest.pairing,
            "cell_type": manifest.cell_type,
            "n_cells": details["n_cells"],
            "n_train_donors": len(manifest.train_donors),
            "n_held_out_donors": len(manifest.held_out_donors),
            "candidates": [c.gene_symbol for c in manifest.candidates],
        },
    )
