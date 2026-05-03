"""
评估器模块单元测试
使用mock隔离模型和DataLoader依赖
"""

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluators import Evaluator


class MockModelDict(torch.nn.Module):
    """返回字典输出的模拟模型"""

    def __init__(self, task_type="classification", return_probs=True):
        super().__init__()
        self.task_type = task_type
        self.return_probs = return_probs
        self._counter = 0

    def forward(self, batch):
        batch_size = batch["sequence"].shape[0]
        if self.task_type == "classification":
            # 生成确定性输出，确保覆盖所有3个类别
            probabilities = torch.zeros(batch_size, 3)
            for i in range(batch_size):
                cls_idx = (self._counter + i) % 3
                probabilities[i, cls_idx] = 0.7
                probabilities[i, (cls_idx + 1) % 3] = 0.2
                probabilities[i, (cls_idx + 2) % 3] = 0.1
            self._counter += batch_size
            predictions = torch.argmax(probabilities, dim=-1)
            if not self.return_probs:
                return {"predictions": predictions}
            return {
                "predictions": predictions,
                "probabilities": probabilities,
            }
        else:
            predictions = torch.randn(batch_size, 1)
            return {"predictions": predictions}


class MockModelTensor(torch.nn.Module):
    """返回张量输出的模拟模型"""

    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self._counter = 0

    def forward(self, batch):
        batch_size = batch["sequence"].shape[0]
        logits = torch.zeros(batch_size, self.num_classes)
        for i in range(batch_size):
            cls_idx = (self._counter + i) % self.num_classes
            logits[i, cls_idx] = 2.0
            logits[i, (cls_idx + 1) % self.num_classes] = 0.5
        self._counter += batch_size
        return logits


class MockModelNoTensor(torch.nn.Module):
    """返回非张量输出的模拟模型"""

    def forward(self, batch):
        return "invalid_output"


class TestEvaluatorInit:
    """初始化测试"""

    def test_init_default_device(self):
        """默认设备选择(cuda/cpu)"""
        model = MockModelDict()
        evaluator = Evaluator(model)
        assert evaluator.device in ("cuda", "cpu")

    def test_init_explicit_device(self):
        """显式指定设备"""
        model = MockModelDict()
        evaluator = Evaluator(model, device="cpu")
        assert evaluator.device == "cpu"

    def test_model_to_device(self):
        """模型移动到指定设备"""
        model = MockModelDict()
        evaluator = Evaluator(model, device="cpu")
        # cpu模型无需验证device属性，验证eval模式即可
        assert evaluator.model.training is False

    def test_model_eval_mode(self):
        """模型设置为eval模式"""
        model = MockModelDict()
        model.train()
        evaluator = Evaluator(model)
        assert evaluator.model.training is False


class TestEvaluatorClassification:
    """分类评估测试"""

    def _make_dataloader(self, num_samples=8, task="classification"):
        """创建模拟DataLoader"""
        sequences = torch.randint(0, 20, (num_samples, 10))
        if task == "classification":
            labels = torch.tensor([i % 3 for i in range(num_samples)])
        else:
            labels = torch.randn(num_samples, 1)
        dataset = TensorDataset(sequences, labels)
        dataloader = DataLoader(
            [("not_a_dict",)], shuffle=False
        )  # 占位，不直接使用TensorDataset以符合batch为dict的要求

        # 实际返回dict batch的DataLoader
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return {
                    "sequence": sequences[idx],
                    "label": labels[idx],
                }

        return DataLoader(DictDataset(num_samples), batch_size=4, shuffle=False)

    def test_evaluate_classification(self):
        """分类任务评估"""
        model = MockModelDict(task_type="classification")
        evaluator = Evaluator(model, task_type="classification", device="cpu")
        dataloader = self._make_dataloader(8, "classification")
        result = evaluator.evaluate(dataloader)
        assert "metrics" in result

    def test_evaluate_returns_metrics(self):
        """返回metrics字典"""
        model = MockModelDict(task_type="classification")
        evaluator = Evaluator(model, task_type="classification", device="cpu")
        dataloader = self._make_dataloader(8, "classification")
        result = evaluator.evaluate(dataloader)
        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "confusion_matrix" in metrics

    def test_evaluate_with_probabilities(self):
        """包含概率输出时计算AUC"""
        model = MockModelDict(task_type="classification", return_probs=True)
        evaluator = Evaluator(model, task_type="classification", device="cpu")
        dataloader = self._make_dataloader(8, "classification")
        result = evaluator.evaluate(dataloader)
        metrics = result["metrics"]
        assert "auc_roc" in metrics
        assert "auc_pr" in metrics

    def test_batch_processing(self):
        """多batch数据处理"""
        model = MockModelDict(task_type="classification")
        evaluator = Evaluator(model, task_type="classification", device="cpu")
        dataloader = self._make_dataloader(16, "classification")
        result = evaluator.evaluate(dataloader, return_predictions=True)
        assert result["predictions"].shape[0] == 16
        assert result["targets"].shape[0] == 16
        assert "probabilities" in result

    def test_return_predictions(self):
        """return_predictions=True返回完整结果"""
        model = MockModelDict(task_type="classification")
        evaluator = Evaluator(model, task_type="classification", device="cpu")
        dataloader = self._make_dataloader(8, "classification")
        result = evaluator.evaluate(dataloader, return_predictions=True)
        assert "metrics" in result
        assert "predictions" in result
        assert "targets" in result
        assert "probabilities" in result


class TestEvaluatorRegression:
    """回归评估测试"""

    def _make_dataloader(self, num_samples=8):
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return {
                    "sequence": torch.randint(0, 20, (10,)),
                    "label": torch.randn(1),
                }

        return DataLoader(DictDataset(num_samples), batch_size=4, shuffle=False)

    def test_evaluate_regression(self):
        """回归任务评估"""
        model = MockModelDict(task_type="regression")
        evaluator = Evaluator(model, task_type="regression", device="cpu")
        dataloader = self._make_dataloader(8)
        result = evaluator.evaluate(dataloader)
        assert "metrics" in result

    def test_regression_metrics(self):
        """返回MAE, MSE, RMSE, R2"""
        model = MockModelDict(task_type="regression")
        evaluator = Evaluator(model, task_type="regression", device="cpu")
        dataloader = self._make_dataloader(8)
        result = evaluator.evaluate(dataloader)
        metrics = result["metrics"]
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics


class TestCalculateMetrics:
    """指标计算测试"""

    def test_metrics_with_probabilities(self):
        """带概率的指标"""
        model = MockModelDict(task_type="classification", return_probs=True)
        evaluator = Evaluator(model, task_type="classification", device="cpu")

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return {
                    "sequence": torch.randint(0, 20, (10,)),
                    "label": torch.tensor(idx % 3).long(),
                }

        dataloader = DataLoader(DictDataset(), batch_size=4, shuffle=False)
        result = evaluator.evaluate(dataloader)
        metrics = result["metrics"]
        assert "auc_roc" in metrics
        assert "auc_pr" in metrics

    def test_metrics_without_probabilities(self):
        """无概率时的指标"""
        model = MockModelDict(task_type="classification", return_probs=False)
        evaluator = Evaluator(model, task_type="classification", device="cpu")

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return {
                    "sequence": torch.randint(0, 20, (10,)),
                    "label": torch.tensor(idx % 3).long(),
                }

        dataloader = DataLoader(DictDataset(), batch_size=4, shuffle=False)
        result = evaluator.evaluate(dataloader)
        metrics = result["metrics"]
        assert "auc_roc" not in metrics
        assert "auc_pr" not in metrics
        assert "accuracy" in metrics

    def test_tensor_output_model(self):
        """模型返回原始tensor时的处理"""
        model = MockModelTensor(num_classes=3)
        evaluator = Evaluator(model, task_type="classification", device="cpu")

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return {
                    "sequence": torch.randint(0, 20, (10,)),
                    "label": torch.tensor(idx % 3).long(),
                }

        dataloader = DataLoader(DictDataset(), batch_size=4, shuffle=False)
        result = evaluator.evaluate(dataloader)
        assert "metrics" in result
        assert "accuracy" in result["metrics"]


class TestEvaluatorEdgeCases:
    """边界情况"""

    def test_empty_dataloader(self):
        """空DataLoader处理"""
        model = MockModelDict()
        evaluator = Evaluator(model, task_type="classification", device="cpu")

        class EmptyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError

        dataloader = DataLoader(EmptyDataset(), batch_size=4, shuffle=False)
        with pytest.raises((ValueError, IndexError)):
            evaluator.evaluate(dataloader)

    def test_single_batch(self):
        """单batch处理"""
        model = MockModelDict(task_type="classification")
        evaluator = Evaluator(model, task_type="classification", device="cpu")

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return {
                    "sequence": torch.randint(0, 20, (10,)),
                    "label": torch.tensor(idx % 3).long(),
                }

        dataloader = DataLoader(DictDataset(), batch_size=4, shuffle=False)
        result = evaluator.evaluate(dataloader, return_predictions=True)
        assert result["predictions"].shape[0] == 3

    def test_batch_dict_validation(self):
        """batch必须是dict"""
        model = MockModelDict()
        evaluator = Evaluator(model, device="cpu")

        dataset = [(torch.randint(0, 20, (10,)),)]
        dataloader = DataLoader(dataset, batch_size=1)
        with pytest.raises(TypeError, match="Batch must be a dict"):
            evaluator.evaluate(dataloader)

    def test_missing_label_key(self):
        """缺少label键时处理"""
        model = MockModelDict(task_type="classification")
        evaluator = Evaluator(model, task_type="classification", device="cpu")

        class BadDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {
                    "sequence": torch.randint(0, 20, (10,)),
                    # 缺少label
                }

        dataloader = DataLoader(BadDataset(), batch_size=2, shuffle=False)
        with pytest.raises(TypeError):
            evaluator.evaluate(dataloader)

    def test_invalid_model_output_type(self):
        """模型输出类型错误"""
        model = MockModelNoTensor()
        evaluator = Evaluator(model, device="cpu")

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {
                    "sequence": torch.randint(0, 20, (10,)),
                    "label": torch.randint(0, 3, ()).long(),
                }

        dataloader = DataLoader(DictDataset(), batch_size=2, shuffle=False)
        with pytest.raises(TypeError, match="Model output must be tensor or dict"):
            evaluator.evaluate(dataloader)


class TestEvaluatorMccAndPerPtmMetrics:
    """MCC 和 per-PTM-type 指标测试"""

    def test_mcc_in_metrics(self):
        """分类评估结果中始终包含 MCC"""
        model = MockModelDict(task_type="classification", return_probs=True)
        evaluator = Evaluator(model, task_type="classification", device="cpu")

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return {
                    "sequence": torch.randint(0, 20, (10,)),
                    "label": torch.tensor(idx % 3).long(),
                }

        dataloader = DataLoader(DictDataset(), batch_size=4, shuffle=False)
        result = evaluator.evaluate(dataloader)
        assert "mcc" in result["metrics"]

    def test_per_ptm_type_metrics_with_ptm_type(self):
        """batch 提供 ptm_type 时计算 per-PTM-type 指标"""

        class PtmTypeDataset(torch.utils.data.Dataset):
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return {
                    "sequence": torch.randint(0, 20, (10,)),
                    "label": torch.tensor(idx % 3).long(),
                    "ptm_type": "Phosphorylation" if idx % 2 == 0 else "Acetylation",
                }

        model = MockModelDict(task_type="classification", return_probs=True)
        evaluator = Evaluator(model, task_type="classification", device="cpu")
        dataloader = DataLoader(PtmTypeDataset(12), batch_size=4, shuffle=False)
        result = evaluator.evaluate(dataloader)
        assert "per_ptm_type_metrics" in result["metrics"]
        per_ptm = result["metrics"]["per_ptm_type_metrics"]
        assert isinstance(per_ptm, dict)
        assert len(per_ptm) >= 1
        assert "Phosphorylation" in per_ptm or "Acetylation" in per_ptm

    def test_mcc_single_class_returns_nan(self):
        """单类别时 MCC 返回 NaN"""
        model = MockModelDict(task_type="classification", return_probs=False)
        evaluator = Evaluator(model, task_type="classification", device="cpu")

        class SingleClassDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return {
                    "sequence": torch.randint(0, 20, (10,)),
                    "label": torch.tensor(0).long(),
                }

        dataloader = DataLoader(SingleClassDataset(), batch_size=4, shuffle=False)
        result = evaluator.evaluate(dataloader)
        assert "mcc" in result["metrics"]
        assert math.isnan(result["metrics"]["mcc"])
