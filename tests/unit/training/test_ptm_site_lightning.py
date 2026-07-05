"""
PTMSiteLightning模块单元测试
"""

import pytest
import torch
from unittest.mock import patch

from src.training.ptm_site_lightning import PTMSiteLightning


class MockPTMSitePredictor(torch.nn.Module):
    """模拟PTMSitePredictor模型"""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, sequence_indices, **kwargs):
        B = sequence_indices.size(0)
        logits = self.fc(torch.randn(B, 64, device=sequence_indices.device))
        return {'logits': logits}


@pytest.fixture
def mock_model():
    return MockPTMSitePredictor(num_classes=2)


@pytest.fixture
def sample_batch():
    return {
        'sequence_indices': torch.randint(0, 21, (4, 31)),
        'label': torch.tensor([0, 1, 0, 1]),
    }


class TestPTMSiteLightningInit:
    """PTMSiteLightning初始化测试"""

    def test_init_with_model(self, mock_model):
        module = PTMSiteLightning(model=mock_model)
        assert module.model is mock_model
        assert isinstance(module.criterion, torch.nn.CrossEntropyLoss)

    def test_init_with_config(self, mock_model):
        config = {'pos_weight': 2.0}
        module = PTMSiteLightning(model=mock_model, config=config)
        assert module.config['pos_weight'] == 2.0

    @patch('src.training.ptm_site_lightning.create_model')
    def test_init_without_model(self, mock_create):
        mock_model = MockPTMSitePredictor()
        mock_create.return_value = mock_model
        module = PTMSiteLightning(config={'model': {'encoder_type': 'cnn'}})
        mock_create.assert_called_once_with({'encoder_type': 'cnn'})
        assert module.model is mock_model

    def test_init_saves_hyperparameters(self, mock_model):
        module = PTMSiteLightning(model=mock_model, config={'lr': 1e-4})
        assert 'config' in module.hparams

    def test_pos_weight_creates_weighted_loss(self, mock_model):
        config = {'pos_weight': 3.0}
        module = PTMSiteLightning(model=mock_model, config=config)
        weight = module.criterion.weight
        assert weight[1].item() == pytest.approx(3.0)
        assert weight[0].item() == pytest.approx(1.0)

    def test_torchmetrics_initialized(self, mock_model):
        module = PTMSiteLightning(model=mock_model)
        assert hasattr(module, 'train_acc')
        assert hasattr(module, 'val_acc')
        assert hasattr(module, 'val_auroc')
        assert hasattr(module, 'val_f1')
        assert hasattr(module, 'val_precision')
        assert hasattr(module, 'val_recall')
        assert hasattr(module, 'test_acc')
        assert hasattr(module, 'test_auroc')
        assert hasattr(module, 'test_f1')
        assert hasattr(module, 'test_precision')
        assert hasattr(module, 'test_recall')


class TestPTMSiteLightningForward:
    """前向传播测试"""

    def test_forward_returns_dict_with_logits(self, mock_model):
        module = PTMSiteLightning(model=mock_model)
        indices = torch.randint(0, 21, (2, 31))
        output = module(indices)
        assert 'logits' in output
        assert output['logits'].shape == (2, 2)


class TestPTMSiteLightningTrainingStep:
    """训练步骤测试"""

    def test_training_step_returns_loss(self, mock_model, sample_batch):
        module = PTMSiteLightning(model=mock_model)
        loss = module.training_step(sample_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_training_step_logs_metrics(self, mock_model, sample_batch):
        module = PTMSiteLightning(model=mock_model)
        with patch.object(module, 'log') as mock_log:
            module.training_step(sample_batch, 0)
            logged = {call[0][0] for call in mock_log.call_args_list}
            assert 'train_loss' in logged
            assert 'train_acc' in logged

    def test_training_step_updates_accuracy(self, mock_model, sample_batch):
        module = PTMSiteLightning(model=mock_model)
        with patch.object(module, 'log'):
            module.training_step(sample_batch, 0)
        assert module.train_acc.compute() is not None


class TestPTMSiteLightningValidationStep:
    """验证步骤测试"""

    def test_validation_step_returns_loss(self, mock_model, sample_batch):
        module = PTMSiteLightning(model=mock_model)
        loss = module.validation_step(sample_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_validation_step_logs_metrics(self, mock_model, sample_batch):
        module = PTMSiteLightning(model=mock_model)
        with patch.object(module, 'log') as mock_log:
            module.validation_step(sample_batch, 0)
            logged = {call[0][0] for call in mock_log.call_args_list}
            for key in ['val_loss', 'val_acc', 'val_auroc', 'val_f1',
                        'val_precision', 'val_recall']:
                assert key in logged

    def test_validation_step_updates_metrics(self, mock_model, sample_batch):
        module = PTMSiteLightning(model=mock_model)
        with patch.object(module, 'log'):
            module.validation_step(sample_batch, 0)
        assert module.val_acc.compute() is not None
        assert module.val_auroc.compute() is not None
        assert module.val_f1.compute() is not None

    def test_validation_probs_in_range(self, mock_model, sample_batch):
        module = PTMSiteLightning(model=mock_model)
        with patch.object(module, 'log'):
            outputs = module(sample_batch['sequence_indices'])
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1]
            assert (probs >= 0).all()
            assert (probs <= 1).all()


class TestPTMSiteLightningConfigureOptimizers:
    """优化器配置测试"""

    def test_default_adamw(self, mock_model):
        module = PTMSiteLightning(model=mock_model)
        optimizer = module.configure_optimizers()
        if isinstance(optimizer, dict):
            optimizer = optimizer['optimizer']
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_adam_optimizer(self, mock_model):
        config = {'optimizer': 'adam', 'learning_rate': 1e-3}
        module = PTMSiteLightning(model=mock_model, config=config)
        result = module.configure_optimizers()
        assert isinstance(result['optimizer'], torch.optim.Adam)

    def test_sgd_optimizer(self, mock_model):
        config = {'optimizer': 'sgd'}
        module = PTMSiteLightning(model=mock_model, config=config)
        result = module.configure_optimizers()
        assert isinstance(result['optimizer'], torch.optim.SGD)

    def test_cosine_scheduler(self, mock_model):
        config = {'scheduler': 'cosine', 'max_epochs': 50}
        module = PTMSiteLightning(model=mock_model, config=config)
        result = module.configure_optimizers()
        from torch.optim.lr_scheduler import CosineAnnealingLR
        assert isinstance(result['lr_scheduler']['scheduler'], CosineAnnealingLR)

    def test_plateau_scheduler(self, mock_model):
        config = {'scheduler': 'plateau'}
        module = PTMSiteLightning(model=mock_model, config=config)
        result = module.configure_optimizers()
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        assert isinstance(result['lr_scheduler']['scheduler'], ReduceLROnPlateau)

    def test_unknown_optimizer_raises(self, mock_model):
        config = {'optimizer': 'unknown_opt'}
        module = PTMSiteLightning(model=mock_model, config=config)
        with pytest.raises(ValueError, match='未知优化器'):
            module.configure_optimizers()

    def test_optimizer_learning_rate(self, mock_model):
        config = {'learning_rate': 5e-5}
        module = PTMSiteLightning(model=mock_model, config=config)
        result = module.configure_optimizers()
        opt = result if not isinstance(result, dict) else result['optimizer']
        for group in opt.param_groups:
            assert group['lr'] == pytest.approx(5e-5)
