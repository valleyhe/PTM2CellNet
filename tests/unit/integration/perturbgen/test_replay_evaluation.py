"""Replay evaluator contract regressions."""

import pytest

from src.integration.perturbgen.replay_evaluation import _evaluation_contract


def test_engineering_replay_keeps_legacy_omitted_lineage_defaults() -> None:
    assert _evaluation_contract({}) == ("engineering", "synthetic", "unspecified")


def test_formal_replay_requires_explicit_empirical_lineage() -> None:
    with pytest.raises(ValueError, match="evidence_class"):
        _evaluation_contract({"evaluation_mode": "formal", "pvalue_source": "empirical_path_results"})

    with pytest.raises(ValueError, match="pvalue_source"):
        _evaluation_contract({"evaluation_mode": "formal", "evidence_class": "empirical_null"})
