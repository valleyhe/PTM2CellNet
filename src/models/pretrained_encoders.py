"""
预训练模型编码器模块
功能概述: 封装ESM-2、ESM-3、ProtBERT等预训练模型，提供统一接口
设计思路: 使用HuggingFace Transformers库，支持冻结/微调策略

For extended ESM-2 models (ESM2PTMPredictor, ESM2FineTunedModel) used in
specialized PTM prediction and fine-tuning workflows, see
``scripts/tools/esm2_encoder.py``.
"""

import os
from typing import Any, Dict, List, Optional, Union, cast

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


class ESM3Encoder(nn.Module):
    """
    ESM-3编码器（使用EvolutionaryScale esm包）
    封装ESM-3多模态蛋白质语言模型（序列、结构、功能）
    
    与ESM2Encoder不同，此类不继承PretrainedEncoder，因为ESM-3模型
    使用完全不同的架构（几何注意力，多模态输入），需要esm包支持。
    
    支持从本地检查点加载或从HuggingFace自动下载。
    """

    # ESM-3模型名称映射（仅开放权重small版本可用）
    MODEL_NAMES = {
        "small": "esm3_sm_open_v1",
    }

    # ESM-3 small开放权重的架构参数
    _D_MODEL = 1536
    _N_HEADS = 24
    _V_HEADS = 256
    _N_LAYERS = 48

    def __init__(
        self,
        model_size: str = "small",
        freeze: bool = False,
        cache_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        初始化ESM-3编码器

        参数:
            model_size: 模型大小（仅支持"small"，大小写不敏感）
            freeze: 是否冻结预训练权重
            cache_dir: ESM-3 model cache directory (sets ESM_DATA_DIR env var)
            checkpoint_path: 本地检查点路径。若提供且文件存在则从本地加载；
                           否则从HuggingFace自动下载
            device: 运行设备（默认自动检测）
        """
        super().__init__()

        # 标准化模型尺寸
        model_size_normalized = self._normalize_model_size(model_size)
        self.model_size = model_size_normalized

        # 确定设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # 应用缓存目录（Task 1a: 设置 ESM_DATA_DIR 环境变量）
        self._cache_dir = cache_dir
        if cache_dir is not None:
            os.environ["ESM_DATA_DIR"] = str(cache_dir)
            logger.info("ESM-3 cache directory set to: %s", cache_dir)

        # 加载模型
        self.model = self._load_model(checkpoint_path, device)
        self.hidden_dim = self._D_MODEL

        # 创建tokenizer适配器（兼容HF tokenizer接口）
        self.tokenizer = ESM3TokenizerAdapter(self.model.tokenizers)

        # 冻结参数
        self.freeze = freeze
        if freeze:
            self._freeze_parameters()

        logger.info(
            f"ESM-3 {model_size} 编码器初始化完成，"
            f"隐藏维度: {self.hidden_dim}，"
            f"参数量: {self.get_num_parameters() / 1e6:.1f}M，"
            f"可训练参数: {self.get_num_parameters(trainable_only=True) / 1e6:.2f}M，"
            f"设备: {device}，"
            f"加载方式: {'本地检查点' if checkpoint_path else 'HuggingFace'}"
        )

    def _load_model(
        self,
        checkpoint_path: Optional[str],
        device: Union[str, torch.device],
    ) -> nn.Module:
        """加载ESM-3模型，优先使用本地检查点"""
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            return self._load_from_local(checkpoint_path, device)
        return self._load_from_huggingface(device)

    def _load_from_local(
        self, path: str, device: Union[str, torch.device]
    ) -> nn.Module:
        """从本地检查点文件加载ESM-3模型"""
        from esm.pretrained import (
            ESM3,
            ESM3_structure_encoder_v0,
            ESM3_structure_decoder_v0,
            ESM3_function_decoder_v0,
            get_esm3_model_tokenizers,
            ESM3_OPEN_SMALL,
        )

        logger.info(f"从本地检查点加载ESM-3模型: {path}")
        with torch.device(device):
            model = ESM3(
                d_model=self._D_MODEL,
                n_heads=self._N_HEADS,
                v_heads=self._V_HEADS,
                n_layers=self._N_LAYERS,
                structure_encoder_fn=ESM3_structure_encoder_v0,
                structure_decoder_fn=ESM3_structure_decoder_v0,
                function_decoder_fn=ESM3_function_decoder_v0,
                tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
            ).eval()

        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        self._model_source = "local_checkpoint"
        return cast(nn.Module, model)

    def _load_from_huggingface(
        self, device: Union[str, torch.device]
    ) -> nn.Module:
        """从HuggingFace加载ESM-3模型"""
        from esm.pretrained import ESM3_sm_open_v0

        logger.info("从HuggingFace加载ESM-3模型 (esm3_sm_open_v0)")
        # ESM3_sm_open_v0 内部调用 data_root()/data/weights/esm3_sm_open_v1.pth
        # 当本地已存在（包括我们的symlink）时不会重复下载
        model = ESM3_sm_open_v0(device=device)
        self._model_source = "huggingface"
        return cast(nn.Module, model)

    @property
    def model_source(self) -> str:
        """Return the source of the loaded model: 'local_checkpoint', 'huggingface', or 'unknown'."""
        return getattr(self, "_model_source", "unknown")

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
            input_ids: [batch_size, seq_len] 序列token索引（ESM-3 sequence_tokens）
            attention_mask: [batch_size, seq_len] 注意力掩码（当前未用于ESM-3，保留兼容）
            structure_tokens: [batch_size, seq_len] 结构token（可选）
            function_tokens: [batch_size, seq_len] 功能token（可选）

        返回:
            embeddings: [batch_size, seq_len, d_model] 序列表示
        """
        # ESM-3 forward使用纯关键字参数
        forward_kwargs: Dict[str, Any] = {
            "sequence_tokens": input_ids,
        }

        # 可选多模态输入
        if structure_tokens is not None:
            forward_kwargs["structure_tokens"] = structure_tokens
        if function_tokens is not None:
            forward_kwargs["function_tokens"] = function_tokens

        # ESM-3前向传播，返回ESMOutput对象
        output = self.model.forward(**forward_kwargs)

        # 返回嵌入（embeddings字段存储最后一层隐藏状态）
        embeddings = output.embeddings
        return cast(torch.Tensor, embeddings)

    def _freeze_parameters(self) -> None:
        """冻结所有模型参数"""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info(f"已冻结ESM-3模型参数")

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """获取模型参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def tokenize(self, sequences: Any, max_length: int = 1024) -> dict:
        """对蛋白质序列进行tokenize，返回input_ids和attention_mask"""
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """编码蛋白质序列列表，支持长序列的滑动窗口处理。

        参数:
            sequences: 蛋白质序列字符串列表

        返回:
            embeddings: [batch_size, max_seq_len, hidden_dim] 每个残基的嵌入
        """
        from .long_sequence import SlidingWindowHandler

        handler = SlidingWindowHandler(window_size=1022, overlap=100)
        return handler.encode_sequences(self, sequences)

    @staticmethod
    def _normalize_model_size(model_size: str) -> str:
        """标准化模型尺寸字符串（大小写不敏感）"""
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


class ESM3TokenizerAdapter:
    """
    ESM-3 tokenizer适配器
    
    将ESM-3的TokenizerCollection接口适配为HuggingFace tokenizer兼容接口，
    使得ESMTokenizedDataset等下游组件可以无缝使用ESM-3 tokenizer。
    
    ESM-3的序列tokenizer直接将每个氨基酸映射到一个token ID，不添加
    <cls>/<eos>等特殊token。本适配器为保持与ESM-2数据集兼容，会在序列
    首尾添加占位token（pad_token_id），使position mapping逻辑保持一致。
    """

    def __init__(self, tokenizers: Any):
        """
        初始化适配器

        参数:
            tokenizers: ESM-3模型中的TokenizerCollection实例
        """
        self._tokenizers = tokenizers
        self._seq_tokenizer = tokenizers.sequence

        # ESM-3 tokenizer属性
        self.mask_token_id = int(self._seq_tokenizer.mask_token_id)
        self.pad_token_id = int(self._seq_tokenizer.pad_token_id)
        self.vocab_size = int(self._seq_tokenizer.vocab_size)

        # 用于与ESM-2位置对齐的占位token ID
        self.bos_token_id = self.pad_token_id
        self.eos_token_id = self.pad_token_id
        self.unk_token_id = self.mask_token_id

        logger.debug(
            f"ESM3TokenizerAdapter初始化: "
            f"vocab_size={self.vocab_size}, "
            f"mask_token_id={self.mask_token_id}, "
            f"pad_token_id={self.pad_token_id}"
        )

    def __call__(
        self,
        sequences: Any,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 1024,
    ) -> dict:
        """
        对序列进行tokenize，返回兼容HF格式的字典

        参数:
            sequences: 字符串或字符串列表
            return_tensors: 返回格式（仅支持"pt"）
            padding: 是否padding到等长
            truncation: 是否截断超长序列
            max_length: 最大长度（含占位token）

        返回:
            {"input_ids": Tensor, "attention_mask": Tensor}
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        all_input_ids: List[List[int]] = []
        all_attention_masks: List[List[int]] = []

        for seq in sequences:
            # 使用ESM-3 tokenizer编码序列
            token_ids = self._seq_tokenizer.encode(seq)

            # 截断（预留2个位置给BOS/EOS占位token）
            if truncation and len(token_ids) > max_length - 2:
                token_ids = token_ids[:max_length - 2]

            # 添加BOS和EOS占位token（与ESM-2的<cls>/<eos>对齐）
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

            all_input_ids.append(token_ids)
            all_attention_masks.append([1] * len(token_ids))

        # Padding到等长
        if padding:
            max_len = max(len(ids) for ids in all_input_ids)
            if max_len > max_length:
                max_len = max_length

            padded_ids: List[List[int]] = []
            padded_masks: List[List[int]] = []
            for ids, mask in zip(all_input_ids, all_attention_masks):
                if len(ids) > max_len:
                    ids = ids[:max_len]
                    mask = mask[:max_len]
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [self.pad_token_id] * pad_len)
                padded_masks.append(mask + [0] * pad_len)
            all_input_ids = padded_ids
            all_attention_masks = padded_masks

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
            }

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
        }

    def tokenize(self, text: str) -> List[str]:
        """兼容HF tokenizer的tokenize方法（用于测试）"""
        return [str(t) for t in self._seq_tokenizer.encode(text)]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def esm3_encoder(*args: Any, strict: bool = False, **kwargs: Any) -> nn.Module:
    """Return an ESM-3 encoder, falling back to ESM-2 if unavailable.

    Args:
        *args: Positional arguments forwarded to the encoder constructor.
        strict: If True, raise RuntimeError when ESM-3 is unavailable instead
            of falling back to ESM-2. Use in production to prevent silent
            model substitution.  May also be enabled globally via the
            ``PTM2CELLNET_STRICT_MODEL_ASSETS`` environment variable.
        **kwargs: Keyword arguments forwarded to the encoder constructor.

    Note: ESM-3 model weights may not be publicly available. When unavailable,
    this function transparently falls back to ESM2Encoder with the same arguments
    (unless ``strict=True`` or the environment variable is set).
    """
    # Also check global strict mode env var
    _global_strict = os.environ.get("PTM2CELLNET_STRICT_MODEL_ASSETS", "").lower() in ("1", "true", "yes")
    if _global_strict:
        strict = True

    # Dynamic lookup to support monkeypatch in tests and import-time availability
    esm3_cls = globals().get("ESM3Encoder", None)
    if esm3_cls is not None:
        try:
            return cast(nn.Module, esm3_cls(*args, **kwargs))
        except (OSError, ImportError, ValueError, RuntimeError) as exc:
            logger.warning(
                "ESM3Encoder instantiation failed (%s: %s). %s",
                type(exc).__name__, exc,
                "Strict mode forbids fallback." if strict else "Falling back to ESM2Encoder.",
            )
            if strict:
                raise RuntimeError(
                    "ESM3Encoder instantiation failed and strict=True forbids "
                    f"fallback to ESM-2. Original error: {type(exc).__name__}: {exc}"
                ) from exc

    if strict:
        raise RuntimeError(
            "ESM3Encoder is not available and strict=True forbids fallback. "
            "Ensure ESM-3 model weights and the required dependencies are installed, "
            "or set strict=False to allow ESM-2 fallback."
        )

    # Fall back to ESM-2
    logger.info("ESM3Encoder unavailable; using ESM2Encoder as fallback.")
    return ESM2Encoder(*args, **kwargs)
