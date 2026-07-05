"""
预训练模型编码器模块
功能概述: 封装ESM-2、ESM-3、ProtBERT等预训练模型，提供统一接口
设计思路: 使用HuggingFace Transformers库，支持冻结/微调策略

For extended ESM-2 models (ESM2PTMPredictor, ESM2FineTunedModel) used in
specialized PTM prediction and fine-tuning workflows, see
``scripts/tools/esm2_encoder.py``.
"""

from typing import List, Optional, cast

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class PretrainedEncoder(nn.Module):
    """
    预训练模型编码器基类
    封装HuggingFace Transformers模型，提供统一接口
    """

    def __init__(
        self,
        model_name: str,
        freeze: bool = False,
        use_attention_output: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        初始化预训练编码器

        参数:
            model_name: HuggingFace模型名称
            freeze: 是否冻结预训练权重
            use_attention_output: 是否输出注意力权重
            cache_dir: 模型缓存目录
        """
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.use_attention_output = use_attention_output

        # 加载模型配置和权重
        logger.info(f"加载预训练模型: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        # 启用注意力输出需要 eager 注意力实现（其他实现不支持 output_attentions=True）。
        # 否则 transformers 会对每次前向调用发出 UserWarning。
        if self.use_attention_output and hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")

        # 获取隐藏层维度
        self.hidden_dim = self.config.hidden_size

        # 冻结参数
        if freeze:
            self._freeze_parameters()
            logger.info(f"已冻结预训练模型参数: {model_name}")

    def _freeze_parameters(self) -> None:
        """冻结模型参数"""
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            input_ids: [batch_size, seq_len] 输入序列索引
            attention_mask: [batch_size, seq_len] 注意力掩码（可选）

        返回:
            embeddings: [batch_size, seq_len, hidden_dim] 序列表示
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.use_attention_output,
        )

        # 返回最后一层隐藏状态
        return cast(torch.Tensor, outputs.last_hidden_state)

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        获取注意力权重（用于可解释性分析）

        参数:
            input_ids: [batch_size, seq_len] 输入序列索引
            attention_mask: [batch_size, seq_len] 注意力掩码（可选）

        返回:
            attention_weights: 注意力权重元组
        """
        if not self.use_attention_output:
            logger.warning("模型未配置输出注意力权重")
            return None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        return outputs.attentions

    def unfreeze_layers(self, num_layers: int) -> None:
        """
        解冻最后N层参数（用于渐进式微调）

        参数:
            num_layers: 要解冻的层数
        """
        layers_attr = None
        if hasattr(self.model, "layers"):
            layers_attr = self.model.layers
        elif hasattr(self.model, "layer"):
            layers_attr = self.model.layer
        elif hasattr(self.model, "encoder"):
            # ESM-2等模型: model.encoder.layer
            if hasattr(self.model.encoder, "layers"):
                layers_attr = self.model.encoder.layers
            elif hasattr(self.model.encoder, "layer"):
                layers_attr = self.model.encoder.layer

        if layers_attr is not None:
            total_layers = len(layers_attr)
            for i in range(max(0, total_layers - num_layers), total_layers):
                for param in layers_attr[i].parameters():
                    param.requires_grad = True
            logger.info(f"已解冻最后 {num_layers} 层参数")
        else:
            logger.warning("模型结构不支持逐层解冻")

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        获取模型参数数量

        参数:
            trainable_only: 是否只计算可训练参数

        返回:
            参数数量
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class ESM2Encoder(PretrainedEncoder):
    """
    ESM-2编码器
    支持不同大小的ESM-2模型（150M, 650M, 2B等）
    支持LoRA参数高效微调
    """

    # ESM-2模型名称映射
    MODEL_NAMES = {
        "150M": "facebook/esm2_t30_150M_UR50D",
        "650M": "facebook/esm2_t33_650M_UR50D",
        "2B": "facebook/esm2_t36_3B_UR50D",
        "8M": "facebook/esm2_t6_8M_UR50D",
        "35M": "facebook/esm2_t12_35M_UR50D",
        "70M": "facebook/esm2_t18_70M_UR50D",
        "3B": "facebook/esm2_t36_3B_UR50D",
        "15B": "facebook/esm2_t48_15B_UR50D",
    }

    @staticmethod
    def _normalize_model_size(model_size: str) -> str:
        """
        标准化模型尺寸字符串（大小写无关）

        参数:
            model_size: 原始模型尺寸字符串（如"150m", "150M", "150MB"等）

        返回:
            标准化的模型尺寸（如"150M"）

        异常:
            ValueError: 输入无效时抛出
        """
        # 输入验证
        if not isinstance(model_size, str):
            raise ValueError(f"模型尺寸必须是字符串，得到: {type(model_size).__name__}")

        if not model_size or not model_size.strip():
            raise ValueError("模型尺寸不能为空字符串")

        # 转换为大写并移除可能的后缀
        normalized = model_size.upper().rstrip("B")

        # 检查处理后是否为空（例如输入只有"B"的情况）
        if not normalized:
            raise ValueError(f"无效的模型尺寸: '{model_size}'")

        # 验证是否只包含数字和有效后缀（M/B）
        if not normalized[:-1].isdigit() or normalized[-1] not in ('M', 'B'):
            raise ValueError(
                f"无效的模型尺寸格式: '{model_size}'。"
                f"期望格式如: '150M', '650m', '2B'"
            )

        return normalized

    def __init__(
        self,
        model_size: str = "150M",
        freeze: bool = False,
        cache_dir: Optional[str] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        """
        初始化ESM-2编码器

        参数:
            model_size: 模型大小（"150M", "650M", "2B"等，大小写不敏感）
            freeze: 是否冻结预训练权重
            cache_dir: 模型缓存目录
            use_lora: 是否使用LoRA进行微调
            lora_r: LoRA秩
            lora_alpha: LoRA alpha参数
            lora_dropout: LoRA dropout率
        """
        # 标准化模型尺寸（大小写无关）
        model_size_normalized = self._normalize_model_size(model_size)

        # 获取模型名称
        model_name = self.MODEL_NAMES.get(model_size_normalized)
        if model_name is None:
            available_sizes = ", ".join(self.MODEL_NAMES.keys())
            raise ValueError(
                f"不支持的ESM-2模型大小: {model_size}。"
                f"可用的模型大小: {available_sizes}"
            )

        super().__init__(model_name, freeze=freeze, cache_dir=cache_dir)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        self.model_size = model_size_normalized
        self.use_lora = use_lora

        # 应用LoRA（如果需要）
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout)

        logger.info(
            f"ESM-2 {model_size} 编码器初始化完成，"
            f"隐藏维度: {self.hidden_dim}，"
            f"参数量: {self.get_num_parameters() / 1e6:.1f}M，"
            f"可训练参数: {self.get_num_parameters(trainable_only=True) / 1e6:.2f}M，"
            f"LoRA: {use_lora}"
        )

    def _apply_lora(self, r: int, alpha: int, dropout: float) -> None:
        """
        应用LoRA到模型

        参数:
            r: LoRA秩
            alpha: LoRA alpha
            dropout: LoRA dropout
        """
        try:
            from ..training.peft_config import apply_lora_to_encoder, get_lora_config

            lora_config = get_lora_config(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
            )
            apply_lora_to_encoder(self, lora_config)
            logger.info(f"LoRA应用成功: r={r}, alpha={alpha}, dropout={dropout}")

        except ImportError as e:
            logger.error(f"应用LoRA失败，peft库可能未安装: {e}")
            raise

    def save_lora_adapters(self, save_path: str) -> None:
        """
        保存LoRA适配器权重

        参数:
            save_path: 保存路径
        """
        if not self.use_lora:
            logger.warning("模型未使用LoRA，无需保存适配器")
            return

        try:
            from ..training.peft_config import save_lora_adapters
            save_lora_adapters(self.model, save_path)
        except ImportError as e:
            logger.error(f"保存LoRA适配器失败: {e}")
            raise

    def load_lora_adapters(self, load_path: str) -> None:
        """
        加载LoRA适配器权重

        参数:
            load_path: 适配器加载路径
        """
        if not self.use_lora:
            logger.warning("模型未使用LoRA，无法加载适配器")
            return

        try:
            from ..training.peft_config import load_lora_adapters
            self.model = load_lora_adapters(self.model, load_path)
            logger.info(f"LoRA适配器已从 {load_path} 加载")
        except ImportError as e:
            logger.error(f"加载LoRA适配器失败: {e}")
            raise

    def tokenize(self, sequences, max_length: int = 1024) -> dict:
        """对蛋白质序列进行tokenize，返回input_ids和attention_mask"""
        return cast(dict, self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ))

    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        编码蛋白质序列列表，支持长序列(>1022)的滑动窗口处理。

        参数:
            sequences: 蛋白质序列字符串列表

        返回:
            embeddings: [batch_size, max_seq_len, hidden_dim] 每个残基的嵌入
        """
        from .long_sequence import SlidingWindowHandler

        handler = SlidingWindowHandler(window_size=1022, overlap=100)
        return handler.encode_sequences(self, sequences)


class ESM3Encoder(PretrainedEncoder):
    """
    ESM-3编码器
    支持ESM-3多模态蛋白质语言模型（序列、结构、功能）
    支持LoRA参数高效微调
    """

    # ESM-3模型名称映射
    MODEL_NAMES = {
        "small": "facebook/esm3_sm_2B",
        "medium": "facebook/esm3_md_6B",
        "large": "facebook/esm3_lg_12B",
    }

    @staticmethod
    def _normalize_model_size(model_size: str) -> str:
        """
        标准化模型尺寸字符串

        参数:
            model_size: 原始模型尺寸字符串（如"small", "SMALL", "Small"等）

        返回:
            标准化的模型尺寸（如"small"）

        异常:
            ValueError: 输入无效时抛出
        """
        if not isinstance(model_size, str):
            raise ValueError(f"模型尺寸必须是字符串，得到: {type(model_size).__name__}")

        if not model_size or not model_size.strip():
            raise ValueError("模型尺寸不能为空字符串")

        normalized = model_size.strip().lower()

        if normalized not in ESM3Encoder.MODEL_NAMES:
            available = ", ".join(ESM3Encoder.MODEL_NAMES.keys())
            raise ValueError(
                f"无效的ESM-3模型尺寸: '{model_size}'。"
                f"可用的模型尺寸: {available}"
            )

        return normalized

    def __init__(
        self,
        model_size: str = "small",
        freeze: bool = False,
        cache_dir: Optional[str] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_steps: int = 1,
    ):
        """
        初始化ESM-3编码器

        参数:
            model_size: 模型大小（"small", "medium", "large"，大小写不敏感）
            freeze: 是否冻结预训练权重
            cache_dir: 模型缓存目录
            use_lora: 是否使用LoRA进行微调
            lora_r: LoRA秩
            lora_alpha: LoRA alpha参数
            lora_dropout: LoRA dropout率
            num_steps: ESM-3生成步数（用于多步推理模式）
        """
        # 标准化模型尺寸
        model_size_normalized = self._normalize_model_size(model_size)

        # 获取模型名称
        model_name = self.MODEL_NAMES[model_size_normalized]

        super().__init__(model_name, freeze=freeze, cache_dir=cache_dir)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        self.model_size = model_size_normalized
        self.use_lora = use_lora
        self.num_steps = num_steps

        # 应用LoRA（如果需要）
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout)

        logger.info(
            f"ESM-3 {model_size} 编码器初始化完成，"
            f"隐藏维度: {self.hidden_dim}，"
            f"参数量: {self.get_num_parameters() / 1e6:.1f}M，"
            f"可训练参数: {self.get_num_parameters(trainable_only=True) / 1e6:.2f}M，"
            f"LoRA: {use_lora}，"
            f"num_steps: {num_steps}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        structure_tokens: Optional[torch.Tensor] = None,
        function_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播，支持ESM-3多模态输入

        参数:
            input_ids: [batch_size, seq_len] 序列token索引
            attention_mask: [batch_size, seq_len] 注意力掩码（可选）
            structure_tokens: [batch_size, seq_len] 结构token（可选）
            function_tokens: [batch_size, seq_len] 功能token（可选）

        返回:
            embeddings: [batch_size, seq_len, hidden_dim] 序列表示
        """
        forward_kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": self.use_attention_output,
        }

        # ESM-3多模态: 传入结构和功能token（如果提供）
        if structure_tokens is not None:
            forward_kwargs["structure_tokens"] = structure_tokens
        if function_tokens is not None:
            forward_kwargs["function_tokens"] = function_tokens

        outputs = self.model(**forward_kwargs)

        # 返回最后一层隐藏状态
        return cast(torch.Tensor, outputs.last_hidden_state)

    def _apply_lora(self, r: int, alpha: int, dropout: float) -> None:
        """
        应用LoRA到模型

        参数:
            r: LoRA秩
            alpha: LoRA alpha
            dropout: LoRA dropout
        """
        try:
            from ..training.peft_config import apply_lora_to_encoder, get_lora_config

            lora_config = get_lora_config(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
            )
            apply_lora_to_encoder(self, lora_config)
            logger.info(f"LoRA应用成功: r={r}, alpha={alpha}, dropout={dropout}")

        except ImportError as e:
            logger.error(f"应用LoRA失败，peft库可能未安装: {e}")
            raise

    def save_lora_adapters(self, save_path: str) -> None:
        """
        保存LoRA适配器权重

        参数:
            save_path: 保存路径
        """
        if not self.use_lora:
            logger.warning("模型未使用LoRA，无需保存适配器")
            return

        try:
            from ..training.peft_config import save_lora_adapters
            save_lora_adapters(self.model, save_path)
        except ImportError as e:
            logger.error(f"保存LoRA适配器失败: {e}")
            raise

    def load_lora_adapters(self, load_path: str) -> None:
        """
        加载LoRA适配器权重

        参数:
            load_path: 适配器加载路径
        """
        if not self.use_lora:
            logger.warning("模型未使用LoRA，无法加载适配器")
            return

        try:
            from ..training.peft_config import load_lora_adapters
            self.model = load_lora_adapters(self.model, load_path)
            logger.info(f"LoRA适配器已从 {load_path} 加载")
        except ImportError as e:
            logger.error(f"加载LoRA适配器失败: {e}")
            raise

    def tokenize(self, sequences, max_length: int = 1024) -> dict:
        """对蛋白质序列进行tokenize，返回input_ids和attention_mask"""
        return cast(dict, self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ))

    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        编码蛋白质序列列表，支持长序列(>1022)的滑动窗口处理。

        参数:
            sequences: 蛋白质序列字符串列表

        返回:
            embeddings: [batch_size, max_seq_len, hidden_dim] 每个残基的嵌入
        """
        from .long_sequence import SlidingWindowHandler

        handler = SlidingWindowHandler(window_size=1022, overlap=100)
        return handler.encode_sequences(self, sequences)


class ProtBERTEncoder(PretrainedEncoder):
    """
    ProtBERT编码器
    使用ProtBERT预训练模型进行蛋白质序列编码
    """

    def __init__(
        self,
        freeze: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        初始化ProtBERT编码器

        参数:
            freeze: 是否冻结预训练权重
            cache_dir: 模型缓存目录
        """
        # ProtBERT使用BERT架构，需要使用特定的BertConfig/BertModel
        # 而不是AutoConfig/AutoModel，因为其配置文件缺少model_type字段
        from transformers import BertConfig, BertModel

        # 必须先调用父类__init__，否则无法设置属性
        nn.Module.__init__(self)

        model_name = "Rostlab/prot_bert"
        self.model_name = model_name
        self.freeze = freeze
        self.use_attention_output = True

        logger.info(f"加载预训练模型: {model_name}")
        try:
            self.config = BertConfig.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=True
            )
            self.model = BertModel.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=True
            )
        except (OSError, ValueError):
            # 如果本地缓存不可用，尝试在线加载（会下载模型）
            logger.warning("本地缓存不可用，尝试在线加载模型")
            self.config = BertConfig.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)

        # 获取隐藏层维度
        self.hidden_dim = self.config.hidden_size

        # 冻结参数
        if freeze:
            self._freeze_parameters()
            logger.info(f"已冻结预训练模型参数: {model_name}")

        logger.info(
            f"ProtBERT编码器初始化完成，"
            f"隐藏维度: {self.hidden_dim}，"
            f"参数量: {self.get_num_parameters() / 1e6:.1f}M"
        )


class ProtT5Encoder(PretrainedEncoder):
    """
    ProtT5编码器
    使用ProtT5预训练模型进行蛋白质序列编码
    """

    def __init__(
        self,
        freeze: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        初始化ProtT5编码器

        参数:
            freeze: 是否冻结预训练权重
            cache_dir: 模型缓存目录
        """
        # ProtT5-XL-U50是推荐版本
        super().__init__(
            "Rostlab/prot_t5_xl_uniref50",
            freeze=freeze,
            cache_dir=cache_dir,
        )

        logger.info(
            f"ProtT5编码器初始化完成，"
            f"隐藏维度: {self.hidden_dim}，"
            f"参数量: {self.get_num_parameters() / 1e6:.1f}M"
        )
