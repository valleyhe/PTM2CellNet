"""Tests for per-PTM-type evaluation metrics (EVAL-01)."""
import numpy as np

from src.evaluation.metrics import calculate_per_ptm_type_metrics


class TestPerPTMTypeMetrics:
    """Test metrics calculated separately for each PTM type."""

    def test_group_by_ptm_type(self):
        """Correctly group predictions by PTM type."""
        # Need at least 5 samples per type
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
        y_score = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.9, 0.7, 0.3, 0.2, 0.8, 0.1, 0.9])
        ptm_types = (['Phosphorylation'] * 5 +
                     ['Ubiquitination'] * 5 +
                     ['Acetylation'] * 5)

        results = calculate_per_ptm_type_metrics(y_true, y_pred, y_score, ptm_types)

        # Should have results for all 3 PTM types
        assert 'Phosphorylation' in results
        assert 'Ubiquitination' in results
        assert 'Acetylation' in results

    def test_calculate_metrics_per_type(self):
        """Calculate AUC-ROC, AUPR, MCC, F1 per PTM type."""
        # Need at least 5 samples per type
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        y_score = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.9, 0.1])
        ptm_types = ['Phosphorylation'] * 5 + ['Ubiquitination'] * 5

        results = calculate_per_ptm_type_metrics(y_true, y_pred, y_score, ptm_types)

        # Check all expected metrics are present
        for ptm_type in ['Phosphorylation', 'Ubiquitination']:
            assert 'auc_roc' in results[ptm_type]
            assert 'auc_pr' in results[ptm_type]
            assert 'mcc' in results[ptm_type]
            assert 'f1' in results[ptm_type]
            assert 'precision' in results[ptm_type]
            assert 'recall' in results[ptm_type]
            assert 'support' in results[ptm_type]

    def test_skip_low_support_types(self):
        """Skip PTM types with too few samples."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 1, 0])
        y_score = np.array([0.9, 0.1, 0.8, 0.7, 0.3])
        # Only 2 samples for Phosphorylation (below threshold of 5)
        ptm_types = ['Phosphorylation', 'Phosphorylation',
                     'Ubiquitination', 'Ubiquitination', 'Ubiquitination']

        results = calculate_per_ptm_type_metrics(y_true, y_pred, y_score, ptm_types)

        # Phosphorylation should be skipped (only 2 samples)
        assert 'Phosphorylation' not in results
        # Ubiquitination should be present (3 samples, but wait - that's also below 5)
        # Actually with 3 samples it's also below threshold
        # Let me adjust the test

    def test_skip_low_support_types_properly(self):
        """Skip PTM types with fewer than 5 samples."""
        # Create data with enough samples for one type but not another
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 10 samples
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        y_score = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.9, 0.1])
        # 6 samples for Phosphorylation, 4 for Ubiquitination
        ptm_types = (['Phosphorylation'] * 6) + (['Ubiquitination'] * 4)

        results = calculate_per_ptm_type_metrics(y_true, y_pred, y_score, ptm_types)

        # Phosphorylation should be present (6 samples >= 5)
        assert 'Phosphorylation' in results
        # Ubiquitination should be skipped (4 samples < 5)
        assert 'Ubiquitination' not in results

    def test_per_ptm_metrics_structure(self):
        """Verify output structure contains all expected metrics."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_score = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15])
        ptm_types = ['Phosphorylation'] * 8

        results = calculate_per_ptm_type_metrics(y_true, y_pred, y_score, ptm_types)

        # Check structure
        assert 'Phosphorylation' in results
        metrics = results['Phosphorylation']

        # All metrics should be floats (or int for support)
        assert isinstance(metrics['auc_roc'], (int, float))
        assert isinstance(metrics['auc_pr'], (int, float))
        assert isinstance(metrics['mcc'], (int, float))
        assert isinstance(metrics['f1'], (int, float))
        assert isinstance(metrics['precision'], (int, float))
        assert isinstance(metrics['recall'], (int, float))
        assert isinstance(metrics['support'], int)

        # Support should be correct
        assert metrics['support'] == 8

    def test_per_ptm_metrics_values_in_valid_range(self):
        """Verify metric values are in valid ranges."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        y_score = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.9, 0.1])
        ptm_types = ['Phosphorylation'] * 10

        results = calculate_per_ptm_type_metrics(y_true, y_pred, y_score, ptm_types)

        metrics = results['Phosphorylation']

        # All metrics should be in valid ranges
        assert 0 <= metrics['auc_roc'] <= 1
        assert 0 <= metrics['auc_pr'] <= 1
        assert -1 <= metrics['mcc'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert metrics['support'] > 0

    def test_per_ptm_metrics_with_multiple_types(self):
        """Test with multiple PTM types having different performance."""
        np.random.seed(42)

        # Create data for 3 PTM types with different prediction quality
        # Need at least 5 samples per type
        y_true = np.array([1, 0, 1, 0, 1] * 3)  # 15 samples total
        # Perfect predictions for Phosphorylation
        y_score_phos = np.array([0.9, 0.1, 0.95, 0.05, 0.9])
        # Random predictions for Ubiquitination
        y_score_ubiq = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        # Good but not perfect for Acetylation
        y_score_acet = np.array([0.8, 0.3, 0.7, 0.2, 0.85])

        y_score = np.concatenate([y_score_phos, y_score_ubiq, y_score_acet])
        y_pred = (y_score > 0.5).astype(int)
        ptm_types = (['Phosphorylation'] * 5 +
                     ['Ubiquitination'] * 5 +
                     ['Acetylation'] * 5)

        results = calculate_per_ptm_type_metrics(y_true, y_pred, y_score, ptm_types)

        # Phosphorylation should have best metrics
        assert results['Phosphorylation']['auc_roc'] > results['Ubiquitination']['auc_roc']
        assert results['Phosphorylation']['f1'] > results['Ubiquitination']['f1']

        # All should have correct support
        assert results['Phosphorylation']['support'] == 5
        assert results['Ubiquitination']['support'] == 5
        assert results['Acetylation']['support'] == 5
