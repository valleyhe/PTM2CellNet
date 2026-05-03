"""
评估指标模块单元测试
测试metrics.py中所有分类、回归、MCC、AUPR和per-PTM-type指标
"""

import math

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_auc_pr,
    calculate_auc_roc,
    calculate_classification_report,
    calculate_confusion_matrix,
    calculate_f1_score,
    calculate_mae,
    calculate_mcc,
    calculate_mse,
    calculate_per_ptm_type_metrics,
    calculate_precision,
    calculate_r2,
    calculate_recall,
    calculate_rmse,
    calculate_aupr_torch,
)


class TestClassificationMetrics:
    """分类指标测试"""

    @pytest.fixture
    def binary_data(self):
        """二分类测试数据"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        y_score = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.95, 0.85])
        return y_true, y_pred, y_score

    @pytest.fixture
    def multiclass_data(self):
        """多分类测试数据"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 0, 0, 2, 1, 0])
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.1, 0.8],
            [0.2, 0.7, 0.1],
            [0.6, 0.3, 0.1],
        ])
        return y_true, y_pred, y_score

    def test_calculate_accuracy(self, binary_data):
        """准确率计算"""
        y_true, y_pred, _ = binary_data
        acc = calculate_accuracy(y_true, y_pred)
        expected = accuracy_score(y_true, y_pred)
        assert isinstance(acc, float)
        assert acc == pytest.approx(expected)
        assert 0.0 <= acc <= 1.0

    def test_calculate_precision_macro(self, multiclass_data):
        """macro精确率"""
        y_true, y_pred, _ = multiclass_data
        prec = calculate_precision(y_true, y_pred, average="macro")
        expected = precision_score(y_true, y_pred, average="macro")
        assert isinstance(prec, float)
        assert prec == pytest.approx(expected)

    def test_calculate_precision_micro(self, multiclass_data):
        """micro精确率"""
        y_true, y_pred, _ = multiclass_data
        prec = calculate_precision(y_true, y_pred, average="micro")
        expected = precision_score(y_true, y_pred, average="micro")
        assert prec == pytest.approx(expected)

    def test_calculate_precision_weighted(self, multiclass_data):
        """weighted精确率"""
        y_true, y_pred, _ = multiclass_data
        prec = calculate_precision(y_true, y_pred, average="weighted")
        expected = precision_score(y_true, y_pred, average="weighted")
        assert prec == pytest.approx(expected)

    def test_calculate_recall(self, binary_data):
        """召回率"""
        y_true, y_pred, _ = binary_data
        rec = calculate_recall(y_true, y_pred)
        expected = recall_score(y_true, y_pred, average="macro")
        assert isinstance(rec, float)
        assert rec == pytest.approx(expected)

    def test_calculate_f1_score(self, binary_data):
        """F1分数"""
        y_true, y_pred, _ = binary_data
        f1 = calculate_f1_score(y_true, y_pred)
        expected = f1_score(y_true, y_pred, average="macro")
        assert isinstance(f1, float)
        assert f1 == pytest.approx(expected)

    def test_calculate_auc_roc_binary(self, binary_data):
        """二分类AUC-ROC"""
        y_true, _, y_score = binary_data
        auc = calculate_auc_roc(y_true, y_score)
        expected = roc_auc_score(y_true, y_score)
        assert isinstance(auc, float)
        assert auc == pytest.approx(expected)
        assert 0.0 <= auc <= 1.0

    def test_calculate_auc_roc_multiclass(self, multiclass_data):
        """多分类AUC-ROC"""
        y_true, _, y_score = multiclass_data
        auc = calculate_auc_roc(y_true, y_score)
        expected = roc_auc_score(y_true, y_score, multi_class="ovr")
        assert isinstance(auc, float)
        assert auc == pytest.approx(expected)

    def test_calculate_auc_pr_binary(self, binary_data):
        """二分类AUC-PR"""
        y_true, _, y_score = binary_data
        auc_pr = calculate_auc_pr(y_true, y_score)
        expected = average_precision_score(y_true, y_score)
        assert isinstance(auc_pr, float)
        assert auc_pr == pytest.approx(expected)

    def test_calculate_auc_pr_multiclass(self, multiclass_data):
        """多分类AUC-PR"""
        y_true, _, y_score = multiclass_data
        auc_pr = calculate_auc_pr(y_true, y_score)
        expected = average_precision_score(
            np.eye(3)[y_true], y_score, average="macro"
        )
        assert isinstance(auc_pr, float)
        assert auc_pr == pytest.approx(expected, abs=1e-6)


class TestMCCMetric:
    """MCC专项测试"""

    def test_calculate_mcc_binary(self):
        """二分类MCC"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        mcc = calculate_mcc(y_true, y_pred)
        expected = matthews_corrcoef(y_true, y_pred)
        assert mcc == pytest.approx(expected)
        assert mcc == 1.0

    def test_calculate_mcc_multiclass(self):
        """多分类MCC"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 1])

        mcc = calculate_mcc(y_true, y_pred)
        expected = matthews_corrcoef(y_true, y_pred)
        assert mcc == pytest.approx(expected)
        assert -1.0 <= mcc <= 1.0

    def test_mcc_single_class_returns_nan(self):
        """单类别返回NaN (per EVAL-01)"""
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1, 1])

        mcc = calculate_mcc(y_true, y_pred)
        assert math.isnan(mcc)

    def test_mcc_sklearn_consistency(self):
        """与sklearn结果一致"""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, size=100)
        y_pred = np.random.randint(0, 3, size=100)

        mcc = calculate_mcc(y_true, y_pred)
        expected = matthews_corrcoef(y_true, y_pred)
        assert mcc == pytest.approx(expected)


class TestAUPRTorch:
    """AUPR torchmetrics测试"""

    def test_calculate_aupr_torch(self):
        """使用torchmetrics计算AUPR"""
        y_true = torch.tensor([0, 0, 1, 1, 0, 1])
        y_score = torch.tensor([0.1, 0.2, 0.9, 0.95, 0.15, 0.85])

        aupr = calculate_aupr_torch(y_true, y_score)
        assert isinstance(aupr, float)
        assert 0.0 <= aupr <= 1.0

    def test_aupr_with_logits(self):
        """处理logits输入"""
        y_true = torch.tensor([0, 0, 1, 1, 0, 1])
        y_score = torch.tensor([-1.0, -0.5, 2.0, 3.0, -0.8, 1.5])

        aupr = calculate_aupr_torch(y_true, y_score)
        assert isinstance(aupr, float)
        assert 0.0 <= aupr <= 1.0

    def test_aupr_torchmetrics_consistency(self):
        """与torchmetrics结果一致"""
        np.random.seed(42)
        y_true = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0])
        y_score = torch.tensor(np.random.random(8).astype(np.float32))

        aupr = calculate_aupr_torch(y_true, y_score)
        from torchmetrics.functional.classification import binary_average_precision

        expected = float(binary_average_precision(y_score, y_true).item())
        assert aupr == pytest.approx(expected)

    def test_aupr_with_numpy_arrays(self):
        """接受numpy数组输入"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.95, 0.15, 0.85])

        aupr = calculate_aupr_torch(y_true, y_score)
        assert isinstance(aupr, float)
        assert 0.0 <= aupr <= 1.0

    def test_aupr_perfect_prediction(self):
        """完美预测时AUPR接近1.0"""
        y_true = torch.tensor([0, 0, 0, 1, 1, 1])
        y_score = torch.tensor([0.1, 0.2, 0.15, 0.9, 0.95, 0.85])

        aupr = calculate_aupr_torch(y_true, y_score)
        assert aupr > 0.99


class TestPerPTMTypeMetrics:
    """按PTM类型分组的指标"""

    def test_calculate_per_ptm_type(self):
        """基本功能"""
        y_true = np.array([1, 0, 1, 0, 1] * 3)
        y_pred = np.array([1, 0, 1, 1, 0] * 3)
        y_score = np.array([0.9, 0.1, 0.8, 0.7, 0.3] * 3)
        ptm_types = ["Phosphorylation"] * 5 + ["Ubiquitination"] * 5 + ["Acetylation"] * 5

        results = calculate_per_ptm_type_metrics(y_true, y_pred, y_score, ptm_types)

        assert "Phosphorylation" in results
        assert "Ubiquitination" in results
        assert "Acetylation" in results

        for metrics in results.values():
            assert "auc_roc" in metrics
            assert "auc_pr" in metrics
            assert "mcc" in metrics
            assert "f1" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "support" in metrics

    def test_skip_small_groups(self):
        """样本数<5时跳过"""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        y_score = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.9, 0.1])
        ptm_types = ["Phosphorylation"] * 6 + ["Ubiquitination"] * 4

        results = calculate_per_ptm_type_metrics(y_true, y_pred, y_score, ptm_types)

        assert "Phosphorylation" in results
        assert "Ubiquitination" not in results

    def test_multiple_ptm_types(self):
        """多种PTM类型分别计算"""
        np.random.seed(42)
        y_true = np.array([1, 0, 1, 0, 1] * 3)
        y_score_phos = np.array([0.9, 0.1, 0.95, 0.05, 0.9])
        y_score_ubiq = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        y_score_acet = np.array([0.8, 0.3, 0.7, 0.2, 0.85])

        y_score = np.concatenate([y_score_phos, y_score_ubiq, y_score_acet])
        y_pred = (y_score > 0.5).astype(int)
        ptm_types = (
            ["Phosphorylation"] * 5 + ["Ubiquitination"] * 5 + ["Acetylation"] * 5
        )

        results = calculate_per_ptm_type_metrics(y_true, y_pred, y_score, ptm_types)

        assert results["Phosphorylation"]["auc_roc"] > results["Ubiquitination"]["auc_roc"]
        assert results["Phosphorylation"]["f1"] > results["Ubiquitination"]["f1"]
        assert results["Phosphorylation"]["support"] == 5
        assert results["Ubiquitination"]["support"] == 5
        assert results["Acetylation"]["support"] == 5


class TestRegressionMetrics:
    """回归指标测试"""

    @pytest.fixture
    def regression_data(self):
        """回归测试数据"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])
        return y_true, y_pred

    def test_calculate_mae(self, regression_data):
        """平均绝对误差"""
        y_true, y_pred = regression_data
        mae = calculate_mae(y_true, y_pred)
        expected = mean_absolute_error(y_true, y_pred)
        assert isinstance(mae, float)
        assert mae == pytest.approx(expected)
        assert mae >= 0.0

    def test_calculate_mse(self, regression_data):
        """均方误差"""
        y_true, y_pred = regression_data
        mse = calculate_mse(y_true, y_pred)
        expected = mean_squared_error(y_true, y_pred)
        assert isinstance(mse, float)
        assert mse == pytest.approx(expected)
        assert mse >= 0.0

    def test_calculate_rmse(self, regression_data):
        """均方根误差"""
        y_true, y_pred = regression_data
        rmse = calculate_rmse(y_true, y_pred)
        expected = np.sqrt(mean_squared_error(y_true, y_pred))
        assert isinstance(rmse, float)
        assert rmse == pytest.approx(expected)
        assert rmse >= 0.0

    def test_calculate_r2(self, regression_data):
        """R²分数"""
        y_true, y_pred = regression_data
        r2 = calculate_r2(y_true, y_pred)
        expected = r2_score(y_true, y_pred)
        assert isinstance(r2, float)
        assert r2 == pytest.approx(expected)


class TestConfusionMatrixAndReport:
    """混淆矩阵和报告"""

    def test_calculate_confusion_matrix(self):
        """混淆矩阵形状正确"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 1])

        cm = calculate_confusion_matrix(y_true, y_pred)
        expected = confusion_matrix(y_true, y_pred)

        assert cm.shape == (3, 3)
        assert cm.shape == expected.shape
        assert np.array_equal(cm, expected)

    def test_calculate_classification_report(self):
        """报告包含必要字段"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 1])

        report = calculate_classification_report(y_true, y_pred)
        expected = classification_report(y_true, y_pred, output_dict=True)

        assert isinstance(report, dict)
        assert "accuracy" in report
        assert "macro avg" in report
        assert "weighted avg" in report
        assert report["accuracy"] == pytest.approx(expected["accuracy"])
