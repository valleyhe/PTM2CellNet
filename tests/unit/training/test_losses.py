"""
损失函数模块单元测试
测试FocalLoss、DiceLoss和MultiTaskLoss的全面功能
"""

import torch
import torch.nn.functional as F

from src.training.losses import FocalLoss, DiceLoss, MultiTaskLoss


class TestFocalLossBasic:
    """Focal Loss基础功能测试"""

    def test_init_default(self):
        """验证默认参数初始化"""
        loss_fn = FocalLoss()
        assert loss_fn.alpha is None
        assert loss_fn.gamma == 2.0
        assert loss_fn.reduction == "mean"
        assert loss_fn.ignore_index == -100

    def test_init_with_alpha(self):
        """验证带alpha参数的初始化"""
        alpha = torch.tensor([1.0, 2.0])
        loss_fn = FocalLoss(alpha=alpha, gamma=1.5, reduction="sum")
        assert loss_fn.alpha is not None
        assert torch.allclose(loss_fn.alpha, alpha)
        assert loss_fn.gamma == 1.5
        assert loss_fn.reduction == "sum"

    def test_forward_basic(self):
        """测试基本前向传播"""
        loss_fn = FocalLoss()
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flow(self):
        """验证反向传播梯度正常"""
        loss_fn = FocalLoss(alpha=torch.tensor([1.0, 2.0]), gamma=2.0)
        logits = torch.randn(8, 2, requires_grad=True)
        targets = torch.randint(0, 2, (8,))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
        assert not torch.isnan(logits.grad).any()
        assert not torch.all(logits.grad == 0)


class TestFocalLossAlphaWeighting:
    """alpha加权测试"""

    def test_from_class_counts_binary(self):
        """二分类场景，[99, 1] -> alpha归一化验证"""
        loss_fn = FocalLoss.from_class_counts([99, 1])

        expected_alpha = torch.tensor([1.0, 99.0])
        expected_alpha = expected_alpha / expected_alpha.mean()

        assert loss_fn.alpha is not None
        assert torch.allclose(loss_fn.alpha, expected_alpha, rtol=0.1)

    def test_from_class_counts_multiclass(self):
        """多分类场景 [100, 50, 25]"""
        class_counts = [100, 50, 25]
        loss_fn = FocalLoss.from_class_counts(class_counts)

        total = sum(class_counts)
        num_classes = len(class_counts)
        expected_alpha = torch.tensor([
            total / (num_classes * c) for c in class_counts
        ])
        expected_alpha = expected_alpha / expected_alpha.mean()

        assert loss_fn.alpha is not None
        assert torch.allclose(loss_fn.alpha, expected_alpha, rtol=0.01)

    def test_alpha_weighting_effect(self):
        """验证alpha加权改变损失值"""
        logits = torch.tensor([
            [2.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ])
        targets = torch.tensor([0, 0, 0, 1])

        loss_no_alpha = FocalLoss(alpha=None)(logits, targets)
        alpha = torch.tensor([1.0, 9.0])
        loss_with_alpha = FocalLoss(alpha=alpha)(logits, targets)

        assert loss_with_alpha > loss_no_alpha

    def test_alpha_device_handling(self):
        """alpha张量自动移动到正确设备"""
        alpha = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = FocalLoss(alpha=alpha)

        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)
        assert loss.shape == ()


class TestFocalLossGammaEffect:
    """gamma参数测试"""

    def test_gamma_downweights_easy(self):
        """gamma=2.0比gamma=1.0损失更小"""
        logits = torch.tensor([
            [5.0, 0.0],
            [5.0, 0.0],
        ])
        targets = torch.tensor([0, 0])

        loss_gamma_1 = FocalLoss(gamma=1.0)(logits, targets)
        loss_gamma_2 = FocalLoss(gamma=2.0)(logits, targets)

        assert loss_gamma_2 < loss_gamma_1

    def test_gamma_zero_equals_ce(self):
        """gamma=0时接近交叉熵"""
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        focal_loss = FocalLoss(gamma=0.0, alpha=None)(logits, targets)
        ce_loss = F.cross_entropy(logits, targets)

        assert torch.allclose(focal_loss, ce_loss, rtol=0.1)


class TestFocalLossNumericalStability:
    """数值稳定性测试"""

    def test_extreme_logits(self):
        """输入[[100.0, 0.0]]不产生NaN"""
        loss_fn = FocalLoss()

        logits_pos = torch.tensor([[100.0, 0.0]])
        targets = torch.tensor([0])
        loss = loss_fn(logits_pos, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        logits_neg = torch.tensor([[-100.0, 0.0]])
        loss = loss_fn(logits_neg, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_extreme_imbalance(self):
        """类别比例10000:1不产生NaN梯度"""
        loss_fn = FocalLoss.from_class_counts([10000, 1])

        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 0, 0, 0])

        loss = loss_fn(logits, targets)
        loss.backward()

        assert not torch.isnan(loss)
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_gradient_no_nan(self):
        """梯度正常流动无NaN"""
        loss_fn = FocalLoss(alpha=torch.tensor([1.0, 2.0]), gamma=2.0)

        logits = torch.randn(8, 2, requires_grad=True)
        targets = torch.randint(0, 2, (8,))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestFocalLossReduction:
    """reduction模式测试"""

    def test_reduction_mean(self):
        """返回标量"""
        loss_fn = FocalLoss(reduction="mean")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_reduction_sum(self):
        """返回标量"""
        loss_fn = FocalLoss(reduction="sum")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_reduction_none(self):
        """返回(batch_size,)形状"""
        loss_fn = FocalLoss(reduction="none")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(logits, targets)
        assert loss.shape == (4,)


class TestFocalLossIgnoreIndex:
    """ignore_index功能测试"""

    def test_ignore_excludes(self):
        """忽略指定索引的样本"""
        loss_fn = FocalLoss(ignore_index=-100)

        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, -100, 2])

        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)
        assert loss.shape == ()

    def test_all_ignored_returns_zero(self):
        """全部忽略时返回0"""
        loss_fn = FocalLoss(ignore_index=-100)

        logits = torch.randn(4, 3)
        targets = torch.tensor([-100, -100, -100, -100])

        loss = loss_fn(logits, targets)
        assert loss.item() == 0.0


class TestDiceLoss:
    """DiceLoss测试"""

    def test_init(self):
        """验证初始化参数"""
        loss_fn = DiceLoss(smooth=1.0, reduction="mean")
        assert loss_fn.smooth == 1.0
        assert loss_fn.reduction == "mean"

    def test_forward(self):
        """测试前向传播"""
        loss_fn = DiceLoss()
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert loss.item() >= 0
        assert loss.item() <= 1

    def test_reduction_modes(self):
        """测试不同reduction模式"""
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss_mean = DiceLoss(reduction="mean")(logits, targets)
        loss_sum = DiceLoss(reduction="sum")(logits, targets)

        assert loss_mean.shape == ()
        assert loss_sum.shape == ()

        # For "none" reduction, test with segmentation-style inputs (spatial dims)
        logits_seg = torch.randn(4, 3, 8, 8)
        targets_seg = torch.randint(0, 3, (4, 8, 8))
        loss_none = DiceLoss(reduction="none")(logits_seg, targets_seg)
        # Returns per-class Dice loss per sample: (batch_size, num_classes)
        assert loss_none.shape == (4, 3)

    def test_perfect_prediction(self):
        """完美预测时Dice Loss接近0"""
        loss_fn = DiceLoss()
        logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
        targets = torch.tensor([0, 1])

        loss = loss_fn(logits, targets)
        assert loss.item() < 0.02


class TestMultiTaskLoss:
    """多任务损失测试"""

    def test_init(self):
        """验证ModuleDict正确创建"""
        loss_dict = {
            "task1": FocalLoss(),
            "task2": DiceLoss(),
        }
        mtl = MultiTaskLoss(loss_dict)

        assert "task1" in mtl.loss_dict
        assert "task2" in mtl.loss_dict
        assert len(mtl.weights) == 2
        assert mtl.weights["task1"] == 0.5
        assert mtl.weights["task2"] == 0.5

    def test_forward(self):
        """测试多任务前向传播"""
        loss_dict = {
            "task1": FocalLoss(),
            "task2": FocalLoss(),
        }
        mtl = MultiTaskLoss(loss_dict)

        predictions = {
            "task1": torch.randn(4, 3),
            "task2": torch.randn(4, 2),
        }
        targets = {
            "task1": torch.randint(0, 3, (4,)),
            "task2": torch.randint(0, 2, (4,)),
        }

        loss = mtl(predictions, targets)
        assert "task1" in loss
        assert "task2" in loss
        assert "total" in loss
        assert loss["total"].item() > 0

    def test_custom_weights(self):
        """测试自定义权重"""
        loss_dict = {
            "task1": FocalLoss(),
            "task2": FocalLoss(),
        }
        weights = {"task1": 0.8, "task2": 0.2}
        mtl = MultiTaskLoss(loss_dict, weights=weights)

        assert mtl.weights["task1"] == 0.8
        assert mtl.weights["task2"] == 0.2

        predictions = {
            "task1": torch.randn(4, 3),
            "task2": torch.randn(4, 2),
        }
        targets = {
            "task1": torch.randint(0, 3, (4,)),
            "task2": torch.randint(0, 2, (4,)),
        }

        loss = mtl(predictions, targets)
        expected_total = 0.8 * loss["task1"] + 0.2 * loss["task2"]
        assert torch.allclose(loss["total"], expected_total)

    def test_missing_task(self):
        """处理缺失任务的情况"""
        loss_dict = {
            "task1": FocalLoss(),
            "task2": FocalLoss(),
        }
        mtl = MultiTaskLoss(loss_dict)

        predictions = {
            "task1": torch.randn(4, 3),
        }
        targets = {
            "task1": torch.randint(0, 3, (4,)),
        }

        loss = mtl(predictions, targets)
        assert "task1" in loss
        assert "task2" not in loss
        assert "total" in loss
        # Default weight is 0.5 per task, so total should be 0.5 * task1
        assert torch.allclose(loss["total"], loss["task1"] * 0.5)

    def test_backward(self):
        """测试多任务损失反向传播"""
        loss_dict = {
            "task1": FocalLoss(),
        }
        mtl = MultiTaskLoss(loss_dict)

        pred = torch.randn(4, 3, requires_grad=True)
        target = torch.randint(0, 3, (4,))

        loss = mtl({"task1": pred}, {"task1": target})
        loss["total"].backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
