"""
Tests for src/project/scope.py — feature scope registry.

Covers DeferredFeature, CancelledFeature, and the three registry lists.
"""

import pytest
from src.project.scope import (
    DeferredFeature,
    CancelledFeature,
    DEFERRED_FEATURES,
    CANCELLED_FEATURES,
    COMPLETED_FEATURES,
    get_deferred_features,
    get_cancelled_features,
)


class TestDeferredFeature:
    def test_dataclass_fields(self):
        """DeferredFeature must have all required fields."""
        feature = DeferredFeature(
            requirement_id="V99-99",
            name="test_feature",
            summary="A test feature",
            rationale="For testing",
            recommended_path="path/to/impl",
        )
        assert feature.requirement_id == "V99-99"
        assert feature.name == "test_feature"
        assert feature.summary == "A test feature"
        assert feature.rationale == "For testing"
        assert feature.recommended_path == "path/to/impl"

    def test_is_frozen(self):
        """DeferredFeature must be a frozen dataclass (immutable)."""
        feature = DeferredFeature(
            requirement_id="V99-98",
            name="frozen_test",
            summary="Immutability check",
            rationale="Should not allow mutation",
            recommended_path="",
        )
        with pytest.raises(AttributeError):
            feature.name = "mutated"  # type: ignore[misc]

    def test_repr_includes_fields(self):
        """__repr__ should include key fields."""
        feature = DeferredFeature(
            requirement_id="V99-97",
            name="repr_test",
            summary="repr check",
            rationale="Check repr output",
            recommended_path="test/path",
        )
        r = repr(feature)
        assert "V99-97" in r
        assert "repr_test" in r


class TestCancelledFeature:
    def test_dataclass_fields(self):
        """CancelledFeature must have all required fields."""
        feature = CancelledFeature(
            requirement_id="V99-99",
            name="cancelled_test",
            rationale="No longer needed",
        )
        assert feature.requirement_id == "V99-99"
        assert feature.name == "cancelled_test"
        assert feature.rationale == "No longer needed"

    def test_is_frozen(self):
        """CancelledFeature must be immutable."""
        feature = CancelledFeature(
            requirement_id="V99-98",
            name="frozen_cancel",
            rationale="Immutability check",
        )
        with pytest.raises(AttributeError):
            feature.name = "mutated"  # type: ignore[misc]


class TestRegistryLists:
    """Verify the three feature registry lists."""

    def test_deferred_features_not_empty(self):
        """DEFERRED_FEATURES must contain registered deferred items."""
        assert len(DEFERRED_FEATURES) > 0
        # Should include ESM-3 integration (V2-01)
        ids = [f.requirement_id for f in DEFERRED_FEATURES]
        assert "V2-01" in ids

    def test_cancelled_features_not_empty(self):
        """CANCELLED_FEATURES must contain cancelled items."""
        assert len(CANCELLED_FEATURES) > 0
        ids = [f.requirement_id for f in CANCELLED_FEATURES]
        assert "V2-02" in ids or "V2-03" in ids

    def test_completed_features_not_empty(self):
        """COMPLETED_FEATURES must contain completed items."""
        assert len(COMPLETED_FEATURES) > 0
        assert "V1-01" in COMPLETED_FEATURES

    def test_get_deferred_functions(self):
        """get_deferred_features() must return a shallow copy of the list."""
        result = get_deferred_features()
        assert result == DEFERRED_FEATURES
        assert result is not DEFERRED_FEATURES

    def test_get_cancelled_functions(self):
        """get_cancelled_features() must return a shallow copy of the list."""
        result = get_cancelled_features()
        assert result == CANCELLED_FEATURES
        assert result is not CANCELLED_FEATURES
