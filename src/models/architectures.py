# mypy: disable-error-code="arg-type,assignment,dict-item,operator,return-value"
"""
完整模型架构
功能概述: 组合编码器、PTM模块与预测器

设计要点 (2026-07-03 重构)：
    ``PTM2CellNet`` 与 ``PTM2CellNetLarge`` 之前各持 ~400 行几乎完全一致的
    构造/forward/from_config 代码。现统一抽取到 ``PTM2CellNetBase``，子类
    仅声明其专属默认值与 ``from_config`` 的 config 覆盖优先级。这样：
      * 新增编码器 / 任务类型只需改一处；
      * 两个对外类名 (``PTM2CellNet`` / ``PTM2CellNetLarge``) 与签名保持兼容，
        现有 import、checkpoint、config 全部不受影响；
      * ``PTM2CellNetBase`` 也对外暴露，可用于自定义变体 (例如研究脚本)。
"""

from pathlib import Path
from typing import Any, Dict, Optional, cast

import torch
import yaml
from torch import nn

from .encoders import CNNEncoder, TransformerEncoder, LSTMEncoder, GRUEncoder
from .mamba_encoder import MambaEncoder
from .ptm_modules import PTMModule
from .predictors import ClassificationPredictor, RegressionPredictor, CellStatePredictor
from .pooling import create_pooling_layer
from .multitask import MultiTaskPredictor
from ..utils.logging import setup_logger

# DAVF Integration imports (Phase 11, 12)
from .davf_inference import DAVFInferenceModule, DAVFInferenceConfig
from .ptm_direction_mapper import PTMDirectionMapper

logger = setup_logger(__name__)


class PTM2CellNetBase(nn.Module):
    """共享架构基类 — 子类化以固定默认规模。

    该基类持有 encoder / ptm_module / pooling / predictor / 可选 DAVF 分支
    的构造、forward、``get_model_info``、DAVF config 加载等所有共享逻辑。
    子类只需通过类属性 ``VARIANT`` 与 ``_DEFAULT_*`` 描述差异，并按需覆写
    ``from_config`` 的 config 解析顺序。

    Args:
        encoder_type: 编码器类型 (cnn/transformer/lstm/gru/mamba/esm2_*/protbert/prott5)。
        vocab_size: 词表大小（含 padding）。
        embed_dim: 嵌入维度。
        max_seq_len: 最大序列长度。
        num_ptm_types: PTM 类型数。
        num_classes: 类别数（或回归输出维度）。
        num_layers: 编码器层数。
        num_heads: 注意力头数。
        dropout: dropout 概率。
        freeze_encoder: 是否冻结预训练编码器。
        pretrained_cache_dir: 预训练模型缓存目录。
        ptm_fusion_type: PTM 融合类型 (attention/gated/...)。
        task_type: 任务类型 (classification/regression/multitask)。
        pool_type: 池化类型 (mean/max/cls/attention)。
        multitask_configs: 多任务配置（仅 task_type=multitask 时使用）。
        use_davf: 是否启用 DAVF 分支。
        davf_config: DAVF 配置字典。
        recurrent_hidden_multiplier: LSTM/GRU 编码器 ``hidden_dim`` 相对
            ``embed_dim`` 的倍数。Base 默认 2，Large 默认 4；保留为参数以
            显式表达差异，避免重复构造分支。
    """

    # 子类覆盖：标识变体名 ("base" / "large" / 自定义)。
    VARIANT: str = "base"

    # 子类覆盖：构造时的默认规模。Base 用较小值；Large 用较大值。
    _DEFAULT_EMBED_DIM: int = 128
    _DEFAULT_NUM_LAYERS: int = 2
    _DEFAULT_NUM_HEADS: int = 4
    _DEFAULT_RECURRENT_HIDDEN_MULTIPLIER: int = 2

    # DAVF 集成配置文件路径（相对项目根）。两个子类共用同一份配置。
    DAVF_CONFIG_PATH = str(Path(__file__).parent.parent.parent / "configs" / "davf_integration.yaml")
    _DAVF_INFERENCE_KEYS = {
        "state_space",
        "checkpoint_path",
        "feature_dim",
        "hidden_dim",
        "freeze",
        "latent_dim",
        "num_genes",
        "num_steps",
        "device",
    }

    def __init__(
        self,
        encoder_type: str = "transformer",
        vocab_size: int = 21,
        embed_dim: Optional[int] = None,
        max_seq_len: int = 1000,
        num_ptm_types: int = 10,
        num_classes: int = 4,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        pretrained_cache_dir: Optional[str] = None,
        ptm_fusion_type: str = "attention",
        task_type: str = "classification",
        pool_type: str = "mean",
        multitask_configs: Optional[list] = None,
        use_davf: bool = False,
        davf_config: Optional[Dict[str, Any]] = None,
        recurrent_hidden_multiplier: Optional[int] = None,
    ):
        super().__init__()
        # 应用子类默认值（仅在调用方未显式传值时生效）。
        if embed_dim is None:
            embed_dim = self._DEFAULT_EMBED_DIM
        if num_layers is None:
            num_layers = self._DEFAULT_NUM_LAYERS
        if num_heads is None:
            num_heads = self._DEFAULT_NUM_HEADS
        if recurrent_hidden_multiplier is None:
            recurrent_hidden_multiplier = self._DEFAULT_RECURRENT_HIDDEN_MULTIPLIER

        encoder_type = encoder_type.lower()
        self.encoder_type = encoder_type
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.task_type = task_type
        self.pool_type = pool_type

        built = self._build_encoder(
            encoder_type=encoder_type,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
            pretrained_cache_dir=pretrained_cache_dir,
            recurrent_hidden_multiplier=recurrent_hidden_multiplier,
        )
        # _build_encoder returns (encoder, effective_embed_dim). For pretrained
        # backbones (ESM-2 / ProtBERT / ProtT5) the effective embed dim comes
        # from the loaded model and overrides the user-supplied value, matching
        # the historical behaviour where ``self.embed_dim`` was reassigned
        # after constructing the encoder.
        encoder, effective_embed_dim = built
        if effective_embed_dim is not None and effective_embed_dim != embed_dim:
            self.embed_dim = effective_embed_dim
        self.encoder = encoder

        # PTM 模块
        self.ptm_module = PTMModule(
            num_ptm_types=num_ptm_types,
            embed_dim=self.embed_dim,
            max_position=max_seq_len,
            num_attention_heads=max(1, num_heads),
            num_layers=max(1, num_layers),
            dropout=dropout,
            fusion_type=ptm_fusion_type,
        )

        # 池化层
        self.pooling = create_pooling_layer(
            pool_type=pool_type,
            hidden_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 预测器
        self.predictor = self._build_predictor(
            task_type=task_type,
            input_dim=self.embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            multitask_configs=multitask_configs,
        )

        # DAVF 分支初始化（D-08, D-09, D-10）
        self.use_davf = use_davf
        self.davf_config = dict(davf_config or {})
        self.davf_module: Optional[DAVFInferenceModule] = None
        self.ptm_mapper: Optional[PTMDirectionMapper] = None
        self.davf_feature_dim: int = 0

        if use_davf:
            if task_type == "multitask":
                raise ValueError("DAVF integration not supported with multitask mode")

            davf_config_local = self.davf_config
            davf_inference_config = DAVFInferenceConfig(
                state_space=davf_config_local.get("state_space", "scvi_latent"),
                checkpoint_path=davf_config_local.get(
                    "checkpoint_path",
                    "checkpoints/latent_davf_ibd_norman/best_model.pt",
                ),
                feature_dim=davf_config_local.get("feature_dim", 128),
                hidden_dim=davf_config_local.get("hidden_dim", 256),
                freeze=davf_config_local.get("freeze", True),
                latent_dim=davf_config_local.get("latent_dim", 10),
                num_genes=davf_config_local.get("num_genes", 5000),
                num_steps=davf_config_local.get("num_steps", 50),
                device=davf_config_local.get("device", None),
            )

            self.davf_module = DAVFInferenceModule(davf_inference_config)
            self.ptm_mapper = PTMDirectionMapper()
            self.davf_feature_dim = davf_inference_config.feature_dim

            # 用扩展后的输入维度重建 predictor（D-04）
            combined_input_dim = self.embed_dim + self.davf_feature_dim
            self.predictor = self._build_predictor(
                task_type=task_type,
                input_dim=combined_input_dim,
                num_classes=num_classes,
                dropout=dropout,
                multitask_configs=None,
                override_hidden_base=combined_input_dim,
            )

            logger.info(
                f"DAVF branch enabled: feature_dim={self.davf_feature_dim}, "
                f"combined_input_dim={combined_input_dim}"
            )

    # ------------------------------------------------------------------
    # Builder helpers — kept as methods so subclasses can override individual
    # pieces (e.g. custom encoder registry) without forking __init__.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_encoder(
        encoder_type: str,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        freeze_encoder: bool,
        pretrained_cache_dir: Optional[str],
        recurrent_hidden_multiplier: int,
    ) -> tuple[nn.Module, Optional[int]]:
        """Instantiate the sequence encoder based on ``encoder_type``.

        Centralised so adding a new encoder touches exactly one place rather
        than being copy-pasted across PTM2CellNet and PTM2CellNetLarge.

        Returns ``(encoder, effective_embed_dim)``. For non-pretrained encoders
        the second element is ``None`` (the caller's ``embed_dim`` is already
        correct). For pretrained backbones (ESM-2 / ProtBERT / ProtT5) it is
        the backbone's own hidden dim, which the caller must use to override
        ``self.embed_dim`` so subsequent PTM module / pooling / predictor are
        built with the right dimension.
        """
        if encoder_type == "cnn":
            return CNNEncoder(vocab_size, embed_dim, max_len=max_seq_len, dropout=dropout), None
        if encoder_type == "transformer":
            return TransformerEncoder(
                vocab_size,
                embed_dim,
                max_len=max_seq_len,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            ), None
        if encoder_type == "lstm":
            return LSTMEncoder(
                vocab_size,
                embed_dim,
                max_len=max_seq_len,
                hidden_dim=embed_dim * recurrent_hidden_multiplier,
                num_layers=num_layers,
                dropout=dropout,
            ), None
        if encoder_type == "gru":
            return GRUEncoder(
                vocab_size,
                embed_dim,
                max_len=max_seq_len,
                hidden_dim=embed_dim * recurrent_hidden_multiplier,
                num_layers=num_layers,
                dropout=dropout,
            ), None
        if encoder_type == "mamba":
            logger.info(
                f"使用Mamba编码器，隐藏维度: {embed_dim}, 层数: {num_layers}"
            )
            return MambaEncoder(
                vocab_size=vocab_size,
                hidden_dim=embed_dim,
                num_layers=num_layers,
                state_dim=16,
                dropout=dropout,
                max_len=max_seq_len,
            ), None
        if encoder_type.startswith("esm2"):
            from .pretrained_encoders import ESM2Encoder

            model_size = encoder_type.split("_")[1] if "_" in encoder_type else "150M"
            enc = ESM2Encoder(
                model_size=model_size,
                freeze=freeze_encoder,
                cache_dir=pretrained_cache_dir,
            )
            logger.info(f"使用ESM-2 {model_size} 预训练编码器，隐藏维度: {enc.hidden_dim}")
            return enc, enc.hidden_dim
        if encoder_type == "protbert":
            from .pretrained_encoders import ProtBERTEncoder

            enc = ProtBERTEncoder(
                freeze=freeze_encoder,
                cache_dir=pretrained_cache_dir,
            )
            logger.info(f"使用ProtBERT预训练编码器，隐藏维度: {enc.hidden_dim}")
            return enc, enc.hidden_dim
        if encoder_type == "prott5":
            from .pretrained_encoders import ProtT5Encoder

            enc = ProtT5Encoder(
                freeze=freeze_encoder,
                cache_dir=pretrained_cache_dir,
            )
            logger.info(f"使用ProtT5预训练编码器，隐藏维度: {enc.hidden_dim}")
            return enc, enc.hidden_dim
        raise ValueError(f"不支持的编码器类型: {encoder_type}")

    @staticmethod
    def _build_predictor(
        task_type: str,
        input_dim: int,
        num_classes: int,
        dropout: float,
        multitask_configs: Optional[list],
        override_hidden_base: Optional[int] = None,
    ) -> CellStatePredictor:
        """Construct the predictor head.

        ``override_hidden_base`` lets the DAVF branch rebuild the predictor
        with the expanded input dim without duplicating the hidden-dim math.
        """
        hidden_base = override_hidden_base if override_hidden_base is not None else input_dim
        if task_type == "classification":
            return ClassificationPredictor(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dims=[hidden_base * 2, hidden_base],
                dropout=dropout,
            )
        if task_type == "regression":
            return RegressionPredictor(
                input_dim=input_dim,
                output_dim=num_classes,
                hidden_dims=[hidden_base * 2, hidden_base],
                dropout=dropout,
            )
        if task_type == "multitask":
            if multitask_configs is None:
                raise ValueError("多任务模式需要提供multitask_configs")
            return MultiTaskPredictor(
                input_dim=input_dim,
                task_configs=multitask_configs,
                hidden_dims=[hidden_base * 2, hidden_base],
                dropout=dropout,
            )
        raise ValueError(f"不支持的任务类型: {task_type}")

    # ------------------------------------------------------------------
    # Forward / introspection
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 支持两种输入：ESM tokenized（input_ids）或自定义索引（sequence）
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            seq_emb = self.encoder(input_ids, attention_mask)
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
        elif (
            isinstance(batch.get("sequence"), list)
            and batch["sequence"]
            and isinstance(batch["sequence"][0], str)
            and hasattr(self.encoder, "encode_sequences")
        ):
            seq_emb = self.encoder.encode_sequences(batch["sequence"])
            batch_size = seq_emb.size(0)
            seq_len = seq_emb.size(1)
            device = seq_emb.device
            attention_mask = None
        else:
            sequences = batch["sequence"]
            seq_emb = self.encoder(sequences)
            batch_size, seq_len = sequences.shape
            device = sequences.device
            attention_mask = None

        ptm_types = batch.get(
            "ptm_types", torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        )
        ptm_mask = batch.get("ptm_mask")
        if ptm_mask is None:
            ptm_mask = (ptm_types > 0).float()

        ptm_positions = batch.get(
            "ptm_positions",
            torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1),
        )

        fused = self.ptm_module(seq_emb, ptm_types, ptm_positions, ptm_mask)

        # 池化
        if isinstance(self.pooling, nn.Module):
            pooled = self.pooling(fused, attention_mask)
        else:
            pooled = fused.mean(dim=1)

        # DAVF feature extraction and concatenation (D-11 through D-14)
        if self.use_davf:
            B = pooled.shape[0]
            device = pooled.device

            if "davf_sites" in batch and "davf_gene_names" in batch:
                mapper_output = self.ptm_mapper.map_batch(
                    batch["davf_sites"],
                    batch["davf_gene_names"],
                )
                davf_output = self.davf_module(mapper_output)
                davf_features = davf_output.davf_features.to(device)
            else:
                # No PTM sites provided - use zero features (D-18)
                davf_features = torch.zeros(
                    B,
                    self.davf_feature_dim,
                    device=device,
                    dtype=pooled.dtype,
                )

            pooled = torch.cat([pooled, davf_features], dim=-1)

        output = self.predictor(pooled)
        return cast(Dict[str, torch.Tensor], output)

    def get_model_info(self) -> Dict[str, Any]:
        from .model_utils import count_parameters

        param_stats = count_parameters(self)
        info: Dict[str, Any] = {
            "variant": self.VARIANT,
            "encoder_type": self.encoder_type,
            "embed_dim": self.embed_dim,
            "num_classes": self.num_classes,
            "task_type": self.task_type,
            "pool_type": self.pool_type,
            "freeze_encoder": self.freeze_encoder,
            "use_davf": self.use_davf,
            "davf_feature_dim": self.davf_feature_dim if self.use_davf else 0,
            "davf_config": dict(self.davf_config) if self.use_davf else None,
            "total_params": param_stats["total_params"],
            "trainable_params": param_stats["trainable_params"],
        }
        return info

    # ------------------------------------------------------------------
    # DAVF config loading & from_config
    # ------------------------------------------------------------------

    @classmethod
    def _load_davf_config_defaults(cls) -> Dict[str, Any]:
        config_path = Path(cls.DAVF_CONFIG_PATH)
        try:
            yaml_cfg = yaml.safe_load(config_path.read_text())
        except (FileNotFoundError, OSError, yaml.YAMLError) as exc:
            raise ValueError(
                "DAVF config file configs/davf_integration.yaml is missing or invalid"
            ) from exc

        if not isinstance(yaml_cfg, dict) or not isinstance(yaml_cfg.get("davf"), dict):
            raise ValueError("DAVF config file configs/davf_integration.yaml is missing or invalid")

        davf_cfg = dict(yaml_cfg["davf"])
        davf_cfg["scvi_model_path"] = davf_cfg.get("scvi_model_path")
        davf_cfg["geneformer_path"] = davf_cfg.get("geneformer_path")
        return davf_cfg

    @classmethod
    def _resolve_model_args_from_config(
        cls,
        config: Dict[str, Any],
        use_large_overrides: bool = False,
    ) -> Dict[str, Any]:
        """Translate a config dict into ``__init__`` kwargs.

        Shared by both subclasses. ``use_large_overrides=True`` enables the
        ``model.large`` sub-config that PTM2CellNetLarge honours, so the
        difference between the two ``from_config`` methods shrinks to one
        boolean.
        """
        from .model_utils import validate_model_config

        is_valid, errors = validate_model_config(config)
        if not is_valid:
            raise ValueError(f"配置验证失败: {errors}")

        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        # Large variant reads an optional ``model.large`` override block on top
        # of base ``model`` keys; base variant ignores it.
        large_cfg = model_cfg.get("large", {}) if use_large_overrides else {}

        def _pick(key: str, default: Any) -> Any:
            return large_cfg.get(key, model_cfg.get(key, default))

        encoder_type = _pick("encoder_type", "transformer")
        embed_dim = _pick("hidden_dim", cls._DEFAULT_EMBED_DIM)
        num_layers = _pick("num_layers", cls._DEFAULT_NUM_LAYERS)
        num_heads = _pick("num_heads", cls._DEFAULT_NUM_HEADS)
        dropout = _pick("dropout", 0.1)
        freeze_encoder = _pick("freeze_encoder", False)
        pretrained_cache_dir = _pick("pretrained_cache_dir", None)
        ptm_fusion_type = _pick("ptm_fusion_type", "attention")
        task_type = _pick("task_type", "classification")
        pool_type = _pick("pool_type", "mean")
        multitask_configs = _pick("multitask_configs", None)
        recurrent_hidden_multiplier = _pick("recurrent_hidden_multiplier", None)

        vocab_size = _pick(
            "vocab_size",
            len(data_cfg.get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY")) + 1,
        )
        max_seq_len = data_cfg.get("max_sequence_length", 1000)
        num_ptm_types = len(data_cfg.get("ptm_types", [])) or model_cfg.get("num_ptm_types", 10)
        num_classes = _pick("num_classes", 4)

        # DAVF configuration
        use_davf = _pick("use_davf", False)
        davf_config = _pick("davf_config", None)
        merged_davf_config: Optional[Dict[str, Any]] = None

        if use_davf:
            merged_davf_config = cls._load_davf_config_defaults()
            if davf_config is not None:
                merged_davf_config.update(davf_config)
            merged_davf_config["scvi_model_path"] = merged_davf_config.get("scvi_model_path")
            merged_davf_config["geneformer_path"] = merged_davf_config.get("geneformer_path")

        return {
            "encoder_type": encoder_type,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "max_seq_len": max_seq_len,
            "num_ptm_types": num_ptm_types,
            "num_classes": num_classes,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "freeze_encoder": freeze_encoder,
            "pretrained_cache_dir": pretrained_cache_dir,
            "ptm_fusion_type": ptm_fusion_type,
            "task_type": task_type,
            "pool_type": pool_type,
            "multitask_configs": multitask_configs,
            "recurrent_hidden_multiplier": recurrent_hidden_multiplier,
            "use_davf": use_davf,
            "davf_config": merged_davf_config if use_davf else davf_config,
        }


class PTM2CellNetLarge(PTM2CellNetBase):
    """PTM2CellNetLarge — scaled-up variant of PTM2CellNet.

    Same architecture as PTM2CellNet but with larger defaults:
      embed_dim=256, num_layers=4, num_heads=8, hidden_dim=embed_dim*4
    (for the recurrent encoders). All construction/forward/from_config logic
    lives on :class:`PTM2CellNetBase`; this subclass only fixes the defaults
    and reads the ``model.large`` config override block.
    """

    VARIANT = "large"
    _DEFAULT_EMBED_DIM = 256
    _DEFAULT_NUM_LAYERS = 4
    _DEFAULT_NUM_HEADS = 8
    _DEFAULT_RECURRENT_HIDDEN_MULTIPLIER = 4

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PTM2CellNetLarge":
        return cls(**cls._resolve_model_args_from_config(config, use_large_overrides=True))


class PTM2CellNet(PTM2CellNetBase):
    """PTM2CellNet端到端模型 (base 规模)。

    所有共享逻辑继承自 :class:`PTM2CellNetBase`；此类仅固化 base 规模的
    默认超参并忽略 ``model.large`` 覆盖块。API、checkpoint、测试和文档中
    引用的 ``PTM2CellNet`` 类名保持不变。
    """

    VARIANT = "base"
    _DEFAULT_EMBED_DIM = 128
    _DEFAULT_NUM_LAYERS = 2
    _DEFAULT_NUM_HEADS = 4
    _DEFAULT_RECURRENT_HIDDEN_MULTIPLIER = 2

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PTM2CellNet":
        return cls(**cls._resolve_model_args_from_config(config, use_large_overrides=False))
