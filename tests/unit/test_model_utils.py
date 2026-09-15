"""单元测试: src/models/model_utils.py 工具函数。

覆盖:
  - _infer_activation_depth: 检查从不同模型结构中推断层数的逻辑
  - estimate_max_batch_size: 验证批次大小估算在配置覆盖/默认模式下工作
  - get_model_memory_usage: 验证内存估算结构

对应缺口分析 §3.1 — Gap 5:
estimate_max_batch_size 硬编码层数已修复，但缺乏回归测试。
"""

from torch import nn

from src.models.model_utils import (
    _infer_activation_depth,
    estimate_max_batch_size,
    get_model_memory_usage,
)


# =============================================================================
# 辅助模型: 不同层数暴露方式
# =============================================================================


class ModelWithNumLayers(nn.Module):
    """通过 num_layers 属性暴露层数。"""

    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.linear = nn.Linear(hidden_dim, 10)


class ModelWithEncoderLayers(nn.Module):
    """通过 encoder.layers (ModuleList) 暴露层数。"""

    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])


class ModelWithNestedModuleList(nn.Module):
    """仅通过 nn.ModuleList 暴露层数（启发式路径）。"""

    def __init__(self, hidden_dim=128, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])


class ModelWithoutLayerInfo(nn.Module):
    """不暴露层数的模型 — 应使用保守回退值。"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 10)


# =============================================================================
# TestInferActivationDepth
# =============================================================================


class TestInferActivationDepth:
    """_infer_activation_depth 多路径探测逻辑。"""

    def test_uses_explicit_num_layers_attr(self):
        """模型属性 num_layers 优先使用。"""
        model = ModelWithNumLayers(hidden_dim=128, num_layers=6)
        dim, layers = _infer_activation_depth(model)
        assert dim == 128
        assert layers == 6

    def test_uses_n_layers_attr(self):
        """n_layers 属性在无 num_layers 时被识别。"""
        model = ModelWithoutLayerInfo(hidden_dim=64)
        model.n_layers = 8
        dim, layers = _infer_activation_depth(model)
        assert dim == 64
        assert layers == 8

    def test_uses_encoder_layers_modulelist(self):
        """encoder.layers (ModuleList) 被计数。"""
        model = ModelWithEncoderLayers(hidden_dim=128, num_layers=4)
        dim, layers = _infer_activation_depth(model)
        assert layers == 4

    def test_uses_modulelist_heuristic(self):
        """ModuleList 启发式搜索作为后备。"""
        model = ModelWithNestedModuleList(hidden_dim=128, num_layers=3)
        dim, layers = _infer_activation_depth(model)
        assert layers == 3

    def test_fallback_to_10(self):
        """无层数信息时使用保守回退值 10。"""
        model = ModelWithoutLayerInfo(hidden_dim=256)
        dim, layers = _infer_activation_depth(model)
        assert dim == 256
        assert layers == 10  # 保守回退

    def test_config_override_takes_highest_priority(self):
        """显式 config 覆盖优先于所有模型内省。"""
        model = ModelWithoutLayerInfo(hidden_dim=256)
        dim, layers = _infer_activation_depth(model, config={"num_layers": 2})
        assert layers == 2, "config 中的 num_layers 应覆盖默认回退"

    def test_config_override_n_layers(self):
        """config 中的 n_layers 同样生效。"""
        model = ModelWithNumLayers(hidden_dim=128, num_layers=6)
        dim, layers = _infer_activation_depth(model, config={"n_layers": 3})
        assert layers == 3, "config 中的 n_layers 应覆盖模型属性"

    def test_config_override_num_encoder_layers(self):
        """config 中的 num_encoder_layers 同样生效。"""
        model = ModelWithNumLayers(hidden_dim=128, num_layers=6)
        dim, layers = _infer_activation_depth(model, config={"num_encoder_layers": 4})
        assert layers == 4

    def test_invalid_config_key_ignored(self):
        """config 中存在无效 key 时回退到模型内省。"""
        model = ModelWithNumLayers(hidden_dim=128, num_layers=6)
        dim, layers = _infer_activation_depth(model, config={"invalid_key": 99})
        assert layers == 6, "无效 key 不应覆盖模型属性"

    def test_empty_config_ignored(self):
        """空 config 被忽略，使用模型内省。"""
        model = ModelWithNumLayers(hidden_dim=128, num_layers=6)
        dim, layers = _infer_activation_depth(model, config={})
        assert layers == 6


# =============================================================================
# TestEstimateMaxBatchSize
# =============================================================================


class TestEstimateMaxBatchSize:
    """estimate_max_batch_size 功能验证。"""

    def test_returns_at_least_1(self):
        """对于极小内存返回至少 1。"""
        model = ModelWithNumLayers(hidden_dim=128, num_layers=2)
        batch_size = estimate_max_batch_size(
            model,
            seq_len=1000,
            available_memory_gb=0.001,  # 极小内存
        )
        assert batch_size >= 1

    def test_returns_reasonable_value(self):
        """正常配置返回合理的批次大小。"""
        model = ModelWithNumLayers(hidden_dim=128, num_layers=2)
        batch_size = estimate_max_batch_size(
            model,
            seq_len=512,
            available_memory_gb=8.0,
        )
        # 8 GB * 0.5 reserve - param_mem / (512 * 128 * 2 * 4 / 1MB)
        # 应远大于 1 且不超过上限 512
        assert 1 <= batch_size <= 512
        assert batch_size > 1, "8GB 内存下批次大小应 > 1"

    def test_respects_max_cap(self):
        """返回大小不超过 MAX_BATCH_SIZE_CAP (512)。"""
        model = ModelWithNumLayers(hidden_dim=16, num_layers=1)
        batch_size = estimate_max_batch_size(
            model,
            seq_len=10,
            available_memory_gb=999,  # 极大内存
        )
        assert batch_size <= 512

    def test_uses_config_for_layer_count(self):
        """config 中的层数影响批次估算。"""
        model = ModelWithNumLayers(hidden_dim=256, num_layers=12)
        # 使用 config 覆盖为更少层数
        bs_default = estimate_max_batch_size(
            model,
            seq_len=1000,
            available_memory_gb=4.0,
        )
        bs_config = estimate_max_batch_size(
            model,
            seq_len=1000,
            available_memory_gb=4.0,
            config={"num_layers": 2},
        )
        assert bs_config > bs_default, "更少层数应产生更大的估算批次"


# =============================================================================
# TestGetModelMemoryUsage
# =============================================================================


class TestGetModelMemoryUsage:
    """get_model_memory_usage 功能验证。"""

    def test_returns_expected_keys(self):
        """返回结构包含所有必填字段。"""
        model = ModelWithNumLayers(hidden_dim=64, num_layers=2)
        usage = get_model_memory_usage(
            model,
            batch_size=8,
            seq_len=512,
        )
        expected_keys = {
            "params_memory_mb",
            "activation_memory_mb",
            "total_memory_mb",
            "recommended_batch_size",
        }
        assert expected_keys.issubset(usage.keys())

    def test_params_memory_positive(self):
        """参数内存为正数。"""
        model = ModelWithNumLayers(hidden_dim=128, num_layers=3)
        usage = get_model_memory_usage(model, batch_size=1, seq_len=512)
        assert usage["params_memory_mb"] > 0

    def test_total_memory_is_sum(self):
        """total_memory_mb 是各分项之和。"""
        model = ModelWithNumLayers(hidden_dim=128, num_layers=2)
        usage = get_model_memory_usage(model, batch_size=4, seq_len=256)
        assert abs(usage["total_memory_mb"] - (usage["params_memory_mb"] + usage["activation_memory_mb"])) < 1e-6

    def test_recommended_batch_size_reasonable(self):
        """推荐的批次大小在合理范围内。"""
        model = ModelWithNumLayers(hidden_dim=64, num_layers=2)
        usage = get_model_memory_usage(model, batch_size=1, seq_len=256)
        assert 1 <= usage["recommended_batch_size"] <= 512
