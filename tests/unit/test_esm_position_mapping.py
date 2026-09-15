"""
ESM-2 Token Position Mapping验证测试
验证PTM位置在tokenization过程中的正确偏移
"""

import os
import json
import pytest
import torch
import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


@pytest.fixture(scope="module")
def esm2_encoder():
    """提供ESM2Encoder实例。"""
    from src.models.pretrained_encoders import ESM2Encoder

    return ESM2Encoder(model_size="8M", freeze=True)


class TestESMTokenPositionMapping:
    """ESM-2 token position mapping测试类"""

    def test_cls_token_offset_basic(self, esm2_encoder):
        """
        验证+1 offset的基本原理：
        - 序列"ACD"：A在位置1，C在位置2，D在位置3（1-based）
        - Tokenized后：[<cls>, A, C, D, <eos>]（位置0,1,2,3,4）
        - D的PTM应该在tokenized位置3
        """
        from src.data.datasets import ESMTokenizedDataset

        sequence = "ACD"
        df = pd.DataFrame(
            [
                {
                    "sequence": sequence,
                    "ptm_sites": json.dumps([{"position": 3, "type": "phosphorylation"}]),  # D的PTM
                    "cell_state": "Test",
                }
            ]
        )

        dataset = ESMTokenizedDataset(df=df, tokenizer=esm2_encoder.tokenizer)
        sample = dataset[0]

        # Tokenized序列应该是 [<cls>, A, C, D, <eos>]，长度=5
        assert sample["input_ids"].shape[0] == 5

        # PTM在原始位置3（D），应该在tokenized位置3（0-based，跳过<cls>）
        # ptm_mask应该在索引3处为1
        assert sample["ptm_mask"][3].item() == 1.0
        assert sample["ptm_mask"][0].item() == 0.0  # <cls>
        assert sample["ptm_mask"][4].item() == 0.0  # <eos>
        assert sample["ptm_mask"].sum().item() == 1.0

    def test_position_mapping_first_aa(self, esm2_encoder):
        """验证第一个氨基酸(位置1)的PTM映射到tokenized位置1"""
        from src.data.datasets import ESMTokenizedDataset

        df = pd.DataFrame(
            [
                {
                    "sequence": "ACDEF",
                    "ptm_sites": json.dumps([{"position": 1, "type": "phosphorylation"}]),  # A的PTM
                    "cell_state": "Test",
                }
            ]
        )

        dataset = ESMTokenizedDataset(df=df, tokenizer=esm2_encoder.tokenizer)
        sample = dataset[0]

        # 第一个氨基酸A应该在tokenized位置1（跳过<cls>在位置0）
        assert sample["ptm_mask"][1].item() == 1.0
        assert sample["ptm_mask"][0].item() == 0.0  # <cls>
        assert sample["ptm_mask"].sum().item() == 1.0

    def test_position_mapping_last_aa(self, esm2_encoder):
        """验证最后一个氨基酸的PTM不映射到<eos>"""
        from src.data.datasets import ESMTokenizedDataset

        sequence = "ACDEF"
        df = pd.DataFrame(
            [
                {
                    "sequence": sequence,
                    "ptm_sites": json.dumps([{"position": len(sequence), "type": "phosphorylation"}]),  # F的PTM
                    "cell_state": "Test",
                }
            ]
        )

        dataset = ESMTokenizedDataset(df=df, tokenizer=esm2_encoder.tokenizer)
        sample = dataset[0]

        # 最后一个氨基酸F（位置5）应该在tokenized位置5
        # Tokenized: [<cls>, A, C, D, E, F, <eos>]，长度=7
        # F在索引5（0-based），不是<eos>（索引6）
        assert sample["input_ids"].shape[0] == 7
        assert sample["ptm_mask"][5].item() == 1.0  # F的位置
        assert sample["ptm_mask"][6].item() == 0.0  # <eos>
        assert sample["ptm_mask"].sum().item() == 1.0

    def test_multiple_ptm_positions_mapping(self, esm2_encoder):
        """验证多个PTM位置的正确映射"""
        from src.data.datasets import ESMTokenizedDataset

        sequence = "ACDEFGHIKLMNPQRSTVWY"
        df = pd.DataFrame(
            [
                {
                    "sequence": sequence,
                    "ptm_sites": json.dumps(
                        [
                            {"position": 1, "type": "phosphorylation"},  # A
                            {"position": 5, "type": "acetylation"},  # E
                            {"position": 10, "type": "methylation"},  # K
                            {"position": 20, "type": "ubiquitination"},  # Y
                        ]
                    ),
                    "cell_state": "Test",
                }
            ]
        )

        dataset = ESMTokenizedDataset(df=df, tokenizer=esm2_encoder.tokenizer)
        sample = dataset[0]

        # Tokenized: [<cls>, A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y, <eos>]
        # 位置:      [0,    1, 2, 3, 4, 5, 6, 7, 8, 9, 10...]
        # A(1) -> 1, E(5) -> 5, K(10) -> 10, Y(20) -> 20
        assert sample["ptm_mask"][1].item() == 1.0  # A
        assert sample["ptm_mask"][5].item() == 1.0  # E
        assert sample["ptm_mask"][10].item() == 1.0  # K
        assert sample["ptm_mask"][20].item() == 1.0  # Y
        assert sample["ptm_mask"].sum().item() == 4.0

    def test_ptm_position_embedding_alignment(self, esm2_encoder):
        """
        验证PTM位置嵌入与序列嵌入对齐：
        PTM position embedding应该对应tokenized序列中的正确位置
        """
        from src.data.datasets import ESMTokenizedDataset
        from src.models.architectures import PTM2CellNet

        # 创建一个已知序列，在特定位置有PTM
        sequence = "ACDEFGHIKLMN"
        ptm_position = 5  # E的PTM

        df = pd.DataFrame(
            [
                {
                    "sequence": sequence,
                    "ptm_sites": json.dumps([{"position": ptm_position, "type": "phosphorylation"}]),
                    "cell_state": "Test",
                }
            ]
        )

        dataset = ESMTokenizedDataset(df=df, tokenizer=esm2_encoder.tokenizer)
        sample = dataset[0]

        # 验证ptm_mask在正确的位置
        # position=5（1-based）-> 0-based index 4 -> tokenized position 5 (skip <cls>)
        expected_tokenized_pos = ptm_position  # +1 for <cls>，所以等于原始1-based位置
        assert sample["ptm_mask"][expected_tokenized_pos].item() == 1.0

        # 创建模型并验证forward不报错
        model = PTM2CellNet(
            encoder_type="esm2_8M",
            num_classes=2,
            freeze_encoder=True,
        )
        model.eval()

        batch = {
            "input_ids": sample["input_ids"].unsqueeze(0),
            "attention_mask": sample["attention_mask"].unsqueeze(0),
            "ptm_mask": sample["ptm_mask"].unsqueeze(0),
            "ptm_types": sample["ptm_types"].unsqueeze(0),
        }

        with torch.no_grad():
            output = model(batch)

        assert "logits" in output

    def test_position_consistency_with_sequence_length(self, esm2_encoder):
        """验证PTM位置与序列长度的一致性"""
        from src.data.datasets import ESMTokenizedDataset

        sequences = ["A", "AC", "ACD", "ACDEF", "ACDEFGHIKLMNPQRSTVWY"]

        for seq in sequences:
            for pos in range(1, len(seq) + 1):
                df = pd.DataFrame(
                    [
                        {
                            "sequence": seq,
                            "ptm_sites": json.dumps([{"position": pos, "type": "phosphorylation"}]),
                            "cell_state": "Test",
                        }
                    ]
                )

                dataset = ESMTokenizedDataset(df=df, tokenizer=esm2_encoder.tokenizer)
                sample = dataset[0]

                # 序列tokenized后长度 = len(seq) + 2 (<cls> and <eos>)
                assert sample["input_ids"].shape[0] == len(seq) + 2

                # PTM在tokenized位置应该是原始位置+1（跳过<cls>）
                expected_pos = pos  # 原始1-based位置正好等于+1后的位置
                assert sample["ptm_mask"][expected_pos].item() == 1.0
                assert sample["ptm_mask"].sum().item() == 1.0

    def test_no_position_confusion_with_cls_eos(self, esm2_encoder):
        """验证PTM位置不会错误映射到<cls>或<eos>"""
        from src.data.datasets import ESMTokenizedDataset

        # 在边界位置测试
        df = pd.DataFrame(
            [
                {
                    "sequence": "ACDEF",
                    "ptm_sites": json.dumps(
                        [
                            {"position": 1, "type": "phosphorylation"},  # 第一个氨基酸
                            {"position": 5, "type": "acetylation"},  # 最后一个氨基酸
                        ]
                    ),
                    "cell_state": "Test",
                }
            ]
        )

        dataset = ESMTokenizedDataset(df=df, tokenizer=esm2_encoder.tokenizer)
        sample = dataset[0]

        # <cls>和<eos>不应该有PTM
        assert sample["ptm_mask"][0].item() == 0.0  # <cls>
        assert sample["ptm_mask"][6].item() == 0.0  # <eos>

        # PTM应该在正确的氨基酸位置
        assert sample["ptm_mask"][1].item() == 1.0  # A (第一个氨基酸)
        assert sample["ptm_mask"][5].item() == 1.0  # F (最后一个氨基酸)
