"""Tests for MCC and AUPR metrics (EVAL-01)."""
import pytest
import numpy as np
import torch
import math

from src.evaluation.metrics import calculate_mcc, calculate_aupr_torch


class TestMCC:
    """Test Matthews Correlation Coefficient calculation."""

    def test_mcc_perfect_prediction(self):
        """MCC should be 1.0 for perfect prediction."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        mcc = calculate_mcc(y_true, y_pred)

        assert mcc == 1.0

    def test_mcc_perfect_inverse(self):
        """MCC should be -1.0 for perfect inverse prediction."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1, 0])

        mcc = calculate_mcc(y_true, y_pred)

        assert mcc == -1.0

    def test_mcc_random_prediction(self):
        """MCC should be ~0 for random prediction."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=1000)
        y_pred = np.random.randint(0, 2, size=1000)

        mcc = calculate_mcc(y_true, y_pred)

        # Random predictions should have MCC close to 0
        assert abs(mcc) < 0.1

    def test_mcc_single_class_edge_case_true(self):
        """MCC should return NaN when y_true has only one class."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1, 1])

        mcc = calculate_mcc(y_true, y_pred)

        assert math.isnan(mcc)

    def test_mcc_single_class_edge_case_pred(self):
        """MCC should return NaN when y_pred has only one class."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1])

        mcc = calculate_mcc(y_true, y_pred)

        assert math.isnan(mcc)

    def test_mcc_with_numpy_arrays(self):
        """MCC should work with numpy arrays."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        mcc = calculate_mcc(y_true, y_pred)

        # MCC should be between -1 and 1
        assert -1.0 <= mcc <= 1.0

    def test_mcc_with_python_lists(self):
        """MCC should work with Python lists."""
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 1, 1, 0, 0]

        mcc = calculate_mcc(y_true, y_pred)

        assert -1.0 <= mcc <= 1.0


class TestAUPR:
    """Test Area Under Precision-Recall calculation."""

    def test_aupr_perfect_prediction(self):
        """AUPR should be 1.0 for perfect prediction."""
        y_true = torch.tensor([0, 0, 1, 1, 0, 1])
        y_score = torch.tensor([0.1, 0.2, 0.9, 0.95, 0.15, 0.85])

        aupr = calculate_aupr_torch(y_true, y_score)

        assert aupr > 0.99  # Allow small numerical error

    def test_aupr_random_prediction(self):
        """AUPR should be in valid range for random predictions."""
        np.random.seed(42)
        y_true = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        y_score = torch.tensor(np.random.random(8))

        aupr = calculate_aupr_torch(y_true, y_score)

        # AUPR should be in valid range [0, 1]
        # Note: With small sample sizes, random predictions can vary widely
        assert 0 <= aupr <= 1

    def test_aupr_imbalanced_data(self):
        """AUPR should be sensitive to class imbalance."""
        # Highly imbalanced: 1 positive out of 10
        y_true = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        # Good predictions: high score for positive
        y_score = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])

        aupr = calculate_aupr_torch(y_true, y_score)

        # Should be high for good predictions even with imbalance
        assert aupr > 0.8

    def test_aupr_with_numpy_arrays(self):
        """AUPR should work with numpy arrays (auto-converts to tensor)."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.95, 0.15, 0.85])

        aupr = calculate_aupr_torch(y_true, y_score)

        assert 0 <= aupr <= 1

    def test_aupr_with_python_lists(self):
        """AUPR should work with Python lists (auto-converts to tensor)."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_score = [0.1, 0.2, 0.9, 0.95, 0.15, 0.85]

        aupr = calculate_aupr_torch(y_true, y_score)

        assert 0 <= aupr <= 1

    def test_aupr_with_thresholds(self):
        """AUPR should work with specified number of thresholds."""
        y_true = torch.tensor([0, 0, 1, 1, 0, 1])
        y_score = torch.tensor([0.1, 0.2, 0.9, 0.95, 0.15, 0.85])

        aupr = calculate_aupr_torch(y_true, y_score, thresholds=100)

        assert 0 <= aupr <= 1

    def test_aupr_better_than_auroc_for_imbalance(self):
        """AUPR should drop more than AUROC for poor predictions on imbalanced data."""
        # Imbalanced data (10% positive)
        y_true = torch.tensor([0] * 90 + [1] * 10)
        # Poor predictions: random scores
        np.random.seed(42)
        y_score = torch.tensor(np.random.random(100))

        from src.evaluation.metrics import calculate_auc_roc

        aupr = calculate_aupr_torch(y_true, y_score)
        auroc = calculate_auc_roc(y_true.numpy(), y_score.numpy())

        # AUPR should be lower than AUROC for random predictions on imbalanced data
        assert aupr < auroc
