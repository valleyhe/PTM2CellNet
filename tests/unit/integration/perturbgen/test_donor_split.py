"""U-05: explicit train / held-out donor split binding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.integration.perturbgen.donor_split import (
    DonorSplitError,
    bind_frozen_donor_split,
    build_donor_split,
    load_donor_split,
    optional_donor_split_from_args,
)


def test_build_donor_split_sorts_and_hashes_disjoint_lists() -> None:
    split = build_donor_split("D2,D1", ["D5", "D4", "D3"])
    assert split.train_donors == ("D1", "D2")
    assert split.held_out_donors == ("D3", "D4", "D5")
    assert split.sha256 == build_donor_split(("D1", "D2"), ("D3", "D4", "D5")).sha256
    loaded = load_donor_split(split.to_payload())
    assert loaded.sha256 == split.sha256


def test_donor_leakage_and_cardinality_are_hard_errors() -> None:
    with pytest.raises(DonorSplitError, match="leakage"):
        build_donor_split(["D1", "D2"], ["D2", "D3", "D4"])
    with pytest.raises(DonorSplitError, match="at least 2 training"):
        build_donor_split(["D1"], ["D2", "D3", "D4"])
    with pytest.raises(DonorSplitError, match="at least 3 held-out"):
        build_donor_split(["D1", "D2"], ["D3", "D4"])


def test_bind_frozen_donor_split_requires_exact_lists() -> None:
    split = build_donor_split(["D1", "D2"], ["D3", "D4", "D5"])
    frozen = SimpleNamespace(train_donors=("D1", "D2"), held_out_donors=("D3", "D4", "D5"))
    bind_frozen_donor_split(split, frozen)
    with pytest.raises(DonorSplitError, match="does not match"):
        bind_frozen_donor_split(
            split,
            SimpleNamespace(train_donors=("D1", "D2", "D9"), held_out_donors=("D3", "D4", "D5")),
        )


def test_optional_split_is_absent_unless_required() -> None:
    assert optional_donor_split_from_args(train_donors=None, held_out_donors=None) is None
    with pytest.raises(DonorSplitError, match="required"):
        optional_donor_split_from_args(train_donors=None, held_out_donors=None, require=True)
    with pytest.raises(DonorSplitError, match="together"):
        optional_donor_split_from_args(train_donors="D1,D2", held_out_donors=None)
