"""
ESM-2预训练编码器模块
功能: 使用Facebook ESM-2蛋白质语言模型进行序列编码
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ESM2Encoder(nn.Module):
    """
    ESM-2预训练编码器

    使用Facebook的ESM-2模型进行蛋白质序列编码
    """

    # ESM-2模型配置
    ESM2_CONFIGS = {
        "esm2_t6_8M_UR50D": {
            "num_layers": 6,
            "embed_dim": 320,
            "num_heads": 20,
            "params": "8M",
        },
        "esm2_t12_35M_UR50D": {
            "num_layers": 12,
            "embed_dim": 480,
            "num_heads": 20,
            "params": "35M",
        },
        "esm2_t30_150M_UR50D": {
            "num_layers": 30,
            "embed_dim": 640,
            "num_heads": 20,
            "params": "150M",
        },
        "esm2_t33_650M_UR50D": {
            "num_layers": 33,
            "embed_dim": 1280,
            "num_heads": 20,
            "params": "650M",
        },
    }

    # 氨基酸到ESM词表的映射
    AA_TO_ESM = {
        "A": 5,
        "C": 23,
        "D": 13,
        "E": 9,
        "F": 18,
        "G": 6,
        "H": 21,
        "I": 12,
        "K": 15,
        "L": 17,
        "M": 1,
        "N": 22,
        "P": 16,
        "Q": 10,
        "R": 14,
        "S": 20,
        "T": 11,
        "V": 19,
        "W": 8,
        "Y": 7,
        "-": 32,  # padding使用ESM的mask token
    }

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        freeze: bool = True,
        use_pooler: bool = True,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        初始化ESM-2编码器

        参数:
            model_name: ESM-2模型名称
            freeze: 是否冻结预训练参数
            use_pooler: 是否使用池化层获取序列表示
            output_dim: 输出维度（None则使用ESM原始维度）
            dropout: Dropout率
        """
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.use_pooler = use_pooler
        self.output_dim = output_dim

        # 延迟加载ESM（避免导入错误）
        self.esm_model = None
        self.batch_converter = None
        self._loaded = False

        # 配置信息
        if model_name in self.ESM2_CONFIGS:
            self.esm_dim = self.ESM2_CONFIGS[model_name]["embed_dim"]
        else:
            self.esm_dim = 480  # 默认

        # 输出投影层
        if output_dim and output_dim != self.esm_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(self.esm_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout),
            )
        else:
            self.output_proj = None
            self.output_dim = self.esm_dim

        logger.info(f"ESM-2编码器初始化: {model_name}, embed_dim={self.esm_dim}")

    def _load_esm(self):
        """延迟加载ESM模型"""
        if self._loaded:
            return

        try:
            import esm

            logger.info(f"加载ESM-2模型: {self.model_name}")

            # 加载预训练模型
            self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
            self.batch_converter = self.alphabet.get_batch_converter()

            # 冻结参数
            if self.freeze:
                for param in self.esm_model.parameters():
                    param.requires_grad = False
                self.esm_model.eval()
                logger.info("ESM-2参数已冻结")

            self._loaded = True

        except ImportError:
            raise ImportError("请安装fair-esm: pip install fair-esm") from None

    def encode_sequence(
        self,
        sequences: List[str],
    ) -> torch.Tensor:
        """
        编码蛋白质序列

        参数:
            sequences: 氨基酸序列列表

        返回:
            (batch, embed_dim) 编码向量
        """
        self._load_esm()

        # 准备数据
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]

        # 使用ESM的batch converter
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        # 移动到正确设备
        batch_tokens = batch_tokens.to(next(self.parameters()).device if list(self.parameters()) else "cpu")

        # 编码
        with torch.no_grad() if self.freeze else torch.enable_grad():
            results = self.esm_model(
                batch_tokens,
                repr_layers=[self.esm_model.num_layers],
                return_contacts=False,
            )

        # 获取表示
        token_representations = results["representations"][self.esm_model.num_layers]

        # 池化: 使用CLS token或平均池化
        if self.use_pooler:
            # ESM的CLS token是第一个位置
            sequence_representations = token_representations[:, 0]
        else:
            # 平均池化（排除CLS和EOS）
            sequence_representations = token_representations[:, 1:-1].mean(dim=1)

        return sequence_representations

    def forward(
        self,
        sequence_indices: torch.Tensor,
        sequences: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            sequence_indices: (batch, seq_len) 序列索引（兼容原有接口）
            sequences: 可选的实际序列字符串列表

        返回:
            (batch, output_dim) 编码向量
        """
        # 如果提供了实际序列，使用ESM编码
        if sequences is not None:
            embeddings = self.encode_sequence(sequences)
        else:
            # 从索引重建序列
            sequences = self._indices_to_sequences(sequence_indices)
            embeddings = self.encode_sequence(sequences)

        # 投影
        if self.output_proj is not None:
            embeddings = self.output_proj(embeddings)

        return embeddings

    def _indices_to_sequences(self, indices: torch.Tensor) -> List[str]:
        """将索引转换为氨基酸序列"""
        # 反向映射
        idx_to_aa = {v: k for k, v in self.AA_TO_ESM.items()}
        idx_to_aa[20] = "-"  # padding

        sequences = []
        for seq_indices in indices.cpu().numpy():
            seq = "".join(idx_to_aa.get(idx, "-") for idx in seq_indices)
            # 移除padding
            seq = seq.replace("-", "")
            if not seq:
                seq = "A"  # 空序列用单个氨基酸代替
            sequences.append(seq)

        return sequences

    def get_sequence_embedding(
        self,
        sequence: str,
        position: int,
        window_size: int = 15,
    ) -> torch.Tensor:
        """
        获取特定位置周围的序列嵌入

        参数:
            sequence: 完整蛋白质序列
            position: 目标位置
            window_size: 窗口大小

        返回:
            (embed_dim,) 位置嵌入向量
        """
        self._load_esm()

        # 编码完整序列
        with torch.no_grad() if self.freeze else torch.enable_grad():
            results = self.esm_model(
                torch.tensor([[self.alphabet.get_idx(aa) for aa in sequence]]),
                repr_layers=[self.esm_model.num_layers],
                return_contacts=False,
            )

        # 获取特定位置的表示
        token_repr = results["representations"][self.esm_model.num_layers][0]

        # ESM添加了CLS token，所以位置需要+1
        pos_repr = token_repr[position + 1]

        return pos_repr


class ESM2PTMPredictor(nn.Module):
    """
    基于ESM-2的PTM预测模型

    使用ESM-2作为序列编码器，加上任务特定的分类头
    """

    def __init__(
        self,
        esm_model: str = "esm2_t12_35M_UR50D",
        hidden_dim: int = 256,
        num_classes: int = 2,
        ptm_types: Optional[List[str]] = None,
        freeze_esm: bool = True,
        dropout: float = 0.1,
    ):
        """
        初始化ESM-2 PTM预测器

        参数:
            esm_model: ESM-2模型名称
            hidden_dim: 隐藏层维度
            num_classes: 分类数
            ptm_types: PTM类型列表（多任务）
            freeze_esm: 是否冻结ESM参数
            dropout: Dropout率
        """
        super().__init__()

        self.ptm_types = ptm_types or ["Phosphorylation"]

        # ESM-2编码器
        self.encoder = ESM2Encoder(
            model_name=esm_model,
            freeze=freeze_esm,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # 分类头
        if len(self.ptm_types) == 1:
            # 单任务
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
        else:
            # 多任务
            self.classifiers = nn.ModuleDict(
                {
                    ptm: nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, num_classes),
                    )
                    for ptm in self.ptm_types
                }
            )

    def forward(
        self,
        sequences: List[str],
        ptm_type: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            sequences: 氨基酸序列列表
            ptm_type: PTM类型（多任务时需要）

        返回:
            logits和probs
        """
        # 编码
        features = self.encoder.encode_sequence(sequences)

        # 分类
        if len(self.ptm_types) == 1:
            logits = self.classifier(features)
        else:
            if ptm_type is None:
                ptm_type = self.ptm_types[0]
            logits = self.classifiers[ptm_type](features)

        probs = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "features": features,
        }


def create_esm2_model(
    config: Dict,
) -> ESM2PTMPredictor:
    """根据配置创建ESM-2模型"""
    return ESM2PTMPredictor(
        esm_model=config.get("esm_model", "esm2_t12_35M_UR50D"),
        hidden_dim=config.get("hidden_dim", 256),
        num_classes=config.get("num_classes", 2),
        ptm_types=config.get("ptm_types"),
        freeze_esm=config.get("freeze_esm", True),
        dropout=config.get("dropout", 0.1),
    )
