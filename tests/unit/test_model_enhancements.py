"""
模型架构增强功能测试
测试内容: Attention Pooling、多任务预测器、回归任务、模型工具函数
"""

import pytest
import torch
from src.models.pooling import (
    AttentionPooling,
    MultiHeadAttentionPooling,
    WeightedMeanPooling,
    create_pooling_layer,
)
from src.models.multitask import MultiTaskPredictor, HierarchicalMultiTaskPredictor
from src.models.architectures import PTM2CellNet
from src.models.model_utils import (
    validate_model_config,
    count_parameters,
    get_model_memory_usage,
)


class TestAttentionPooling:
    """注意力池化测试"""

    @pytest.fixture
    def sample_features(self):
        """示例特征"""
        batch_size = 4
        seq_len = 50
        hidden_dim = 128
        return torch.randn(batch_size, seq_len, hidden_dim)

    @pytest.fixture
    def attention_mask(self):
        """注意力掩码"""
        # 模拟变长序列，后10个位置为padding
        mask = torch.ones(4, 50)
        mask[:, 40:] = 0
        return mask

    def test_attention_pooling(self, sample_features):
        """测试基础注意力池化"""
        pooling = AttentionPooling(hidden_dim=128, num_heads=1, dropout=0.1)
        output = pooling(sample_features)
        assert output.shape == (4, 128)

    def test_attention_pooling_with_mask(self, sample_features, attention_mask):
        """测试带mask的注意力池化"""
        pooling = AttentionPooling(hidden_dim=128, num_heads=1, dropout=0.1)
        output = pooling(sample_features, attention_mask)
        assert output.shape == (4, 128)

    def test_multihead_attention_pooling(self, sample_features):
        """测试多头注意力池化"""
        pooling = MultiHeadAttentionPooling(
            hidden_dim=128, num_heads=4, num_queries=2, dropout=0.1
        )
        output = pooling(sample_features)
        assert output.shape == (4, 128)

    def test_weighted_mean_pooling(self, sample_features):
        """测试加权均值池化"""
        pooling = WeightedMeanPooling(hidden_dim=128, temperature=1.0)
        output = pooling(sample_features)
        assert output.shape == (4, 128)

    def test_create_pooling_layer_mean(self):
        """测试创建mean池化层"""
        pooling = create_pooling_layer("mean", hidden_dim=128)
        x = torch.randn(2, 10, 128)
        output = pooling(x)
        assert output.shape == (2, 128)

    def test_create_pooling_layer_attention(self):
        """测试创建attention池化层"""
        pooling = create_pooling_layer("attention", hidden_dim=128, num_heads=2)
        x = torch.randn(2, 10, 128)
        output = pooling(x)
        assert output.shape == (2, 128)


class TestMultiTaskPredictor:
    """多任务预测器测试"""

    @pytest.fixture
    def sample_features(self):
        """示例特征"""
        return torch.randn(4, 256)

    @pytest.fixture
    def task_configs(self):
        """任务配置"""
        return [
            {"name": "cell_type", "type": "classification", "num_classes": 5},
            {"name": "cell_state", "type": "classification", "num_classes": 3},
            {"name": "expression_level", "type": "regression", "output_dim": 1},
        ]

    def test_multitask_predictor_init(self, task_configs):
        """测试多任务预测器初始化"""
        predictor = MultiTaskPredictor(
            input_dim=256,
            task_configs=task_configs,
            hidden_dims=[512, 256],
            dropout=0.1,
        )
        assert predictor.num_tasks == 3
        assert set(predictor.get_task_names()) == {"cell_type", "cell_state", "expression_level"}
        assert predictor.get_classification_tasks() == ["cell_type", "cell_state"]
        assert predictor.get_regression_tasks() == ["expression_level"]

    def test_multitask_predictor_forward(self, sample_features, task_configs):
        """测试多任务预测器前向传播"""
        predictor = MultiTaskPredictor(
            input_dim=256,
            task_configs=task_configs,
            hidden_dims=[512, 256],
            dropout=0.1,
        )
        outputs = predictor(sample_features)

        # 检查输出结构
        assert "cell_type" in outputs
        assert "cell_state" in outputs
        assert "expression_level" in outputs

        # 检查分类任务输出
        assert "logits" in outputs["cell_type"]
        assert "probabilities" in outputs["cell_type"]
        assert "predictions" in outputs["cell_type"]
        assert outputs["cell_type"]["logits"].shape == (4, 5)
        assert outputs["cell_state"]["logits"].shape == (4, 3)

        # 检查回归任务输出
        assert "predictions" in outputs["expression_level"]
        assert outputs["expression_level"]["predictions"].shape == (4, 1)

    def test_hierarchical_multitask_predictor(self, sample_features):
        """测试分层多任务预测器"""
        primary_task = {"name": "cell_type", "type": "classification", "num_classes": 5}
        sub_tasks = [
            {"name": "cell_state", "type": "classification", "num_classes": 3},
        ]

        predictor = HierarchicalMultiTaskPredictor(
            input_dim=256,
            primary_task=primary_task,
            sub_tasks=sub_tasks,
            hidden_dims=[512, 256],
            dropout=0.1,
        )

        outputs = predictor(sample_features)
        assert "cell_type" in outputs
        assert "cell_state" in outputs


class TestPTM2CellNetRegression:
    """PTM2CellNet回归任务测试"""

    @pytest.fixture
    def sample_batch(self):
        """示例批次数据"""
        batch_size = 4
        seq_len = 50
        vocab_size = 20
        return {
            "sequence": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "ptm_mask": torch.randint(0, 2, (batch_size, seq_len)).float(),
            "ptm_types": torch.randint(0, 10, (batch_size, seq_len)),
        }

    def test_ptm2cellnet_regression(self, sample_batch):
        """测试回归任务"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            embed_dim=128,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=1,  # 回归任务输出维度
            task_type="regression",
        )
        output = model(sample_batch)

        # 回归任务返回predictions，并提供logits别名兼容旧调用方
        assert "predictions" in output
        assert output["predictions"].shape == (4, 1)
        assert "logits" in output
        assert torch.equal(output["predictions"], output["logits"])
        assert "probabilities" not in output

    def test_ptm2cellnet_attention_pooling(self, sample_batch):
        """测试attention池化"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            embed_dim=128,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=4,
            pool_type="attention",
        )
        output = model(sample_batch)
        assert "logits" in output
        assert output["logits"].shape == (4, 4)

    def test_ptm2cellnet_model_info(self, sample_batch):
        """测试获取模型信息"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            embed_dim=128,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=4,
        )
        info = model.get_model_info()

        assert "encoder_type" in info
        assert "embed_dim" in info
        assert "num_classes" in info
        assert "total_params" in info
        assert "trainable_params" in info
        assert info["encoder_type"] == "cnn"


class TestModelUtils:
    """模型工具函数测试"""

    def test_validate_model_config_valid(self):
        """测试配置验证（有效配置）"""
        config = {
            "model": {
                "encoder_type": "transformer",
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
                "num_classes": 4,
            }
        }
        is_valid, errors = validate_model_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_validate_model_config_invalid(self):
        """测试配置验证（无效配置）"""
        config = {
            "model": {
                "encoder_type": "invalid_encoder",
                "hidden_dim": -1,
                "dropout": 1.5,
            }
        }
        is_valid, errors = validate_model_config(config)
        assert not is_valid
        assert len(errors) > 0

    def test_count_parameters(self):
        """测试参数量统计"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            embed_dim=64,
            max_seq_len=100,
            num_ptm_types=5,
            num_classes=4,
        )
        stats = count_parameters(model)

        assert "total_params" in stats
        assert "trainable_params" in stats
        assert stats["total_params"] > 0

    def test_get_model_memory_usage(self):
        """测试内存使用估算"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            embed_dim=64,
            max_seq_len=100,
            num_ptm_types=5,
            num_classes=4,
        )
        memory = get_model_memory_usage(model, batch_size=8, seq_len=100)

        assert "params_memory_mb" in memory
        assert "activation_memory_mb" in memory
        assert "recommended_batch_size" in memory

    def test_ptm2cellnet_from_config_validation(self):
        """测试从配置创建模型时的验证"""
        invalid_config = {
            "model": {
                "encoder_type": "unknown_encoder",
            }
        }
        with pytest.raises(ValueError):
            PTM2CellNet.from_config(invalid_config)
