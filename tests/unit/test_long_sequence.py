"""
长序列处理模块测试
测试滑动窗口和位置映射功能
"""

import pytest
import torch
import torch.nn as nn

from src.models.long_sequence import LongSequenceHandler, SlidingWindowESM2
from src.models.pretrained_encoders import ESM2Encoder
from src.models.architectures import PTM2CellNet


class MockESM2Encoder(nn.Module):
    """用于测试的虚拟ESM-2编码器"""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 简单的嵌入层模拟
        self.embed = nn.Embedding(30, hidden_dim)

    def tokenize(self, sequences, max_length: int = 1024):
        """模拟tokenize"""
        import torch

        batch_size = len(sequences)
        # 简单模拟：序列长度 + 2（特殊token）
        seq_lengths = [min(len(seq) + 2, max_length) for seq in sequences]
        max_len = max(seq_lengths)

        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, length in enumerate(seq_lengths):
            # 填充随机token id
            input_ids[i, :length] = torch.randint(1, 29, (length,))
            input_ids[i, 0] = 0  # <cls>
            input_ids[i, length-1] = 2  # <eos>
            attention_mask[i, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(self, input_ids, attention_mask=None):
        """模拟前向传播"""
        embeddings = self.embed(input_ids)
        return embeddings


class TestLongSequenceHandler:
    """LongSequenceHandler测试类"""

    def test_short_sequence_single_window(self):
        """测试短序列（<=1022）使用单个窗口"""
        handler = LongSequenceHandler(
            sequence_length=500,
            window_size=1022,
            overlap=100,
        )

        assert handler.get_num_windows() == 1
        assert handler.window_boundaries == [(0, 500)]

    def test_long_sequence_multiple_windows(self):
        """测试长序列（>1022）使用多个窗口"""
        handler = LongSequenceHandler(
            sequence_length=1500,
            window_size=1022,
            overlap=100,
        )

        # 1500 = 1022 + 478, stride = 922
        # 窗口1: [0, 1022]
        # 窗口2: [922, 1500] (但确保包含末尾)
        assert handler.get_num_windows() >= 2

    def test_position_mapping_basic(self):
        """测试基本位置映射"""
        handler = LongSequenceHandler(
            sequence_length=1500,
            window_size=1022,
            overlap=100,
        )

        # 位置0应该在第一个窗口
        window_idx, window_pos = handler.map_position_to_window(0)
        assert window_idx == 0
        assert window_pos == 0

    def test_position_mapping_c_terminal(self):
        """测试C端位置映射"""
        handler = LongSequenceHandler(
            sequence_length=1500,
            window_size=1022,
            overlap=100,
        )

        # C端位置应该在最后一个窗口
        window_idx, window_pos = handler.map_position_to_window(1499)
        last_window = handler.get_num_windows() - 1
        assert window_idx == last_window

    def test_position_mapping_out_of_range(self):
        """测试超出范围的位置"""
        handler = LongSequenceHandler(
            sequence_length=1000,
            window_size=1022,
            overlap=100,
        )

        with pytest.raises(ValueError):
            handler.map_position_to_window(1000)

        with pytest.raises(ValueError):
            handler.map_position_to_window(-1)

    def test_c_terminal_preservation(self):
        """测试C端残基不被截断"""
        # 创建一个刚好超过窗口大小的序列
        handler = LongSequenceHandler(
            sequence_length=1100,
            window_size=1022,
            overlap=100,
        )

        # C端位置（最后几个残基）应该能被映射
        c_terminal_pos = 1099
        window_idx, window_pos = handler.map_position_to_window(c_terminal_pos)

        # 确保C端在窗口内
        start, end = handler.window_boundaries[window_idx]
        assert start <= c_terminal_pos < end


class TestSlidingWindowESM2:
    """SlidingWindowESM2测试类"""

    @pytest.fixture
    def mock_encoder(self):
        """创建模拟ESM-2编码器"""
        return MockESM2Encoder(hidden_dim=128)

    def test_short_sequence_passthrough(self, mock_encoder):
        """测试短序列直接通过（不分割）"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
            pool_type="mean",
        )

        # 短序列
        sequences = ["ACDEFGHIKLMNPQRSTVWY" * 10]  # 200 residues

        output = sliding_encoder(sequences)

        # 输出应该是 [batch_size, hidden_dim]
        assert output.shape == (1, 128)

    def test_long_sequence_split(self, mock_encoder):
        """测试长序列分割为多个窗口"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
            pool_type="mean",
        )

        # 长序列（1500 residues）
        sequences = ["ACDEFGHIKLMNPQRSTVWY" * 75]  # 1500 residues

        output = sliding_encoder(sequences)

        # 输出应该是 [batch_size, hidden_dim]
        assert output.shape == (1, 128)

    def test_pooling_strategies(self, mock_encoder):
        """测试不同池化策略"""
        sequences = ["ACDEFGHIKLMNPQRSTVWY" * 75]  # 1500 residues

        for pool_type in ["mean", "max", "attention"]:
            sliding_encoder = SlidingWindowESM2(
                esm2_encoder=mock_encoder,
                window_size=1022,
                overlap=100,
                pool_type=pool_type,
                hidden_dim=128,
            )

            output = sliding_encoder(sequences)
            assert output.shape == (1, 128), f"Pool type {pool_type} failed"

    def test_window_creation_with_overlap(self, mock_encoder):
        """测试窗口创建和重叠"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=100,
            overlap=20,
            pool_type="mean",
        )

        # 250 residues -> 窗口大小100，重叠20，步长80
        # 窗口1: [0, 100]
        # 窗口2: [80, 180]
        # 窗口3: [160, 260] -> 截断到 [150, 250]
        sequence = "A" * 250
        windows = sliding_encoder._create_windows(sequence)

        assert len(windows) >= 2
        # 验证重叠
        if len(windows) >= 2:
            # 第一个窗口的结尾和第二个窗口的开头应该有重叠
            overlap_len = 100 - 80  # window_size - stride
            assert len(windows[0]) == 100
            assert windows[0][-overlap_len:] == windows[1][:overlap_len]

    def test_empty_sequence_raises(self, mock_encoder):
        """测试空序列抛出异常"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
        )

        with pytest.raises(ValueError):
            sliding_encoder([""])

    def test_batch_processing(self, mock_encoder):
        """测试批处理"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
            pool_type="mean",
        )

        # 不同长度的序列
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY" * 10,   # 200 residues
            "ACDEFGHIKLMNPQRSTVWY" * 75,   # 1500 residues
            "ACDEFGHIKLMNPQRSTVWY" * 5,    # 100 residues
        ]

        output = sliding_encoder(sequences)

        # 输出应该是 [batch_size, hidden_dim]
        assert output.shape == (3, 128)

    def test_return_all_windows(self, mock_encoder):
        """测试返回所有窗口嵌入"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
            pool_type="mean",
        )

        sequences = ["ACDEFGHIKLMNPQRSTVWY" * 75]  # 1500 residues

        output = sliding_encoder(sequences, return_all_windows=True)

        # 输出应该是 [batch_size, num_windows, hidden_dim]
        assert output.ndim == 3
        assert output.shape[0] == 1
        assert output.shape[2] == 128
        assert output.shape[1] >= 2  # 至少2个窗口

    def test_attention_pooling_learnable(self, mock_encoder):
        """测试注意力池化有可学习参数"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
            pool_type="attention",
            hidden_dim=128,
        )

        # 检查注意力池化层存在且有参数
        assert hasattr(sliding_encoder, "attention_pool")
        assert len(list(sliding_encoder.attention_pool.parameters())) > 0

    def test_very_long_sequence(self, mock_encoder):
        """测试非常长的序列（>5000 residues）"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
            pool_type="mean",
        )

        # 5000 residues
        sequences = ["ACDEFGHIKLMNPQRSTVWY" * 250]

        output = sliding_encoder(sequences)

        # 应该能正常处理，输出 [batch_size, hidden_dim]
        assert output.shape == (1, 128)

    def test_get_position_handler(self, mock_encoder):
        """测试获取位置处理器"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
        )

        handler = sliding_encoder.get_position_handler(sequence_length=1500)

        assert isinstance(handler, LongSequenceHandler)
        assert handler.sequence_length == 1500
        assert handler.window_size == 1022
        assert handler.overlap == 100


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def mock_encoder(self):
        return MockESM2Encoder(hidden_dim=128)

    def test_exact_window_boundary(self, mock_encoder):
        """测试刚好在窗口边界的长度"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
        )

        # 刚好1022 residues
        sequences = ["ACDEFGHIKLMNPQRSTVWY" * 51 + "ACDEFGHIKLM"]  # 1022

        output = sliding_encoder(sequences)
        assert output.shape == (1, 128)

    def test_just_over_window_boundary(self, mock_encoder):
        """测试刚好超过窗口边界"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
        )

        # 1023 residues（刚好超过）
        sequences = ["ACDEFGHIKLMNPQRSTVWY" * 51 + "ACDEFGHIKLMN"]  # 1023

        output = sliding_encoder(sequences)
        assert output.shape == (1, 128)

        # 应该创建多个窗口
        windows = sliding_encoder._create_windows(sequences[0])
        assert len(windows) >= 2

    def test_single_residue(self, mock_encoder):
        """测试单残基序列"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=1022,
            overlap=100,
        )

        sequences = ["A"]

        output = sliding_encoder(sequences)
        assert output.shape == (1, 128)

    def test_position_mapping_consistency(self, mock_encoder):
        """测试位置映射的一致性"""
        sliding_encoder = SlidingWindowESM2(
            esm2_encoder=mock_encoder,
            window_size=100,
            overlap=20,
        )

        handler = sliding_encoder.get_position_handler(sequence_length=250)

        # 所有位置都应该能映射
        for pos in range(250):
            window_idx, window_pos = handler.map_position_to_window(pos)
            assert 0 <= window_idx < handler.get_num_windows()
            assert 0 <= window_pos < 100

            # 验证映射正确性
            start, end = handler.window_boundaries[window_idx]
            assert start <= pos < end or window_idx == handler.get_num_windows() - 1


class TestESM2EncoderLongSequenceWiring:
    """ESM2Encoder encode_sequences 长序列集成测试"""

    @pytest.fixture
    def mock_esm2_encoder(self, monkeypatch):
        """创建一个模拟的ESM2Encoder，避免下载真实模型"""
        def mock_init(self, *args, **kwargs):
            nn.Module.__init__(self)
            self.hidden_dim = 128
            self.tokenizer = MockTokenizer()
            # 添加一个虚拟参数以确保device/dtype检测正常工作
            self.dummy = nn.Parameter(torch.zeros(1))
            self._mock_embed = nn.Embedding(30, 128)

        def mock_forward(self, input_ids, attention_mask=None):
            return self._mock_embed(input_ids)

        monkeypatch.setattr(ESM2Encoder, "__init__", mock_init)
        monkeypatch.setattr(ESM2Encoder, "forward", mock_forward)
        encoder = ESM2Encoder(model_size="8M")
        return encoder

    def test_encode_sequences_short(self, mock_esm2_encoder):
        """测试短序列编码"""
        output = mock_esm2_encoder.encode_sequences(["ACDEFG" * 10])
        assert output.shape == (1, 60, 128)

    def test_encode_sequences_long(self, mock_esm2_encoder):
        """测试长序列通过LongSequenceHandler处理"""
        output = mock_esm2_encoder.encode_sequences(["A" * 1500])
        assert output.shape == (1, 1500, 128)
        # 长度与输入一致，说明没有截断
        assert output.size(1) == 1500


class MockTokenizer:
    """模拟ESM-2 tokenizer"""

    def __call__(self, sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024):
        import torch
        batch_size = len(sequences)
        seq_lengths = [min(len(seq) + 2, max_length) for seq in sequences]
        max_len = max(seq_lengths)

        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, length in enumerate(seq_lengths):
            input_ids[i, :length] = torch.randint(1, 29, (length,))
            input_ids[i, 0] = 0  # <cls>
            input_ids[i, length - 1] = 2  # <eos>
            attention_mask[i, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class TestPTM2CellNetLongSequenceRouting:
    """PTM2CellNet 长序列路由测试"""

    def test_ptm2cellnet_routes_string_sequences(self, monkeypatch):
        """测试PTM2CellNet.forward将字符串列表路由到encode_sequences"""
        call_count = 0
        called_sequences = []

        class MockEncoder(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.hidden_dim = 128
                self.dummy = nn.Parameter(torch.zeros(1))

            def encode_sequences(self, sequences):
                nonlocal call_count, called_sequences
                call_count += 1
                called_sequences = sequences
                batch_size = len(sequences)
                max_len = max(len(s) for s in sequences)
                return torch.zeros(batch_size, max_len, 128)

            def forward(self, *args, **kwargs):
                raise RuntimeError("不应调用forward")

        # 替换pretrained_encoders模块中的ESM2Encoder
        import src.models.pretrained_encoders as pretrained_module

        monkeypatch.setattr(pretrained_module, "ESM2Encoder", MockEncoder)

        model = PTM2CellNet(
            encoder_type="esm2_8M",
            num_ptm_types=2,
            num_classes=2,
            embed_dim=128,
        )

        batch = {
            "sequence": ["A" * 1500],
            "ptm_types": torch.zeros(1, 1500).long(),
            "ptm_mask": torch.ones(1, 1500),
        }

        output = model(batch)

        assert call_count == 1
        assert called_sequences == ["A" * 1500]
        assert "logits" in output
