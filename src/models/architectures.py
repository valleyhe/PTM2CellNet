# mypy: ignore-errors
"""
完整模型架构
功能概述: 组合编码器、PTM模块与预测器
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


class PTM2CellNetLarge(nn.Module):
    """PTM2CellNetLarge — scaled-up variant of PTM2CellNet.

    Same architecture as PTM2CellNet but with larger defaults:
      embed_dim=256, num_layers=4, num_heads=8, hidden_dim=embed_dim*4.
    """

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
        embed_dim: int = 256,
        max_seq_len: int = 1000,
        num_ptm_types: int = 10,
        num_classes: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        pretrained_cache_dir: Optional[str] = None,
        ptm_fusion_type: str = "attention",
        task_type: str = "classification",
        pool_type: str = "mean",
        multitask_configs: Optional[list] = None,
        use_davf: bool = False,
        davf_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        encoder_type = encoder_type.lower()
        self.encoder_type = encoder_type
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.task_type = task_type
        self.pool_type = pool_type

        encoder: nn.Module
        if encoder_type == "cnn":
            encoder = CNNEncoder(vocab_size, embed_dim, max_len=max_seq_len, dropout=dropout)
        elif encoder_type == "transformer":
            encoder = TransformerEncoder(
                vocab_size, embed_dim, max_len=max_seq_len, num_layers=num_layers, num_heads=num_heads, dropout=dropout
            )
        elif encoder_type == "lstm":
            encoder = LSTMEncoder(
                vocab_size,
                embed_dim,
                max_len=max_seq_len,
                hidden_dim=embed_dim * 4,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif encoder_type == "gru":
            encoder = GRUEncoder(
                vocab_size,
                embed_dim,
                max_len=max_seq_len,
                hidden_dim=embed_dim * 4,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif encoder_type == "mamba":
            encoder = MambaEncoder(
                vocab_size=vocab_size,
                hidden_dim=embed_dim,
                num_layers=num_layers,
                state_dim=16,
                dropout=dropout,
                max_len=max_seq_len,
            )
            logger.info(f"使用Mamba编码器，隐藏维度: {embed_dim}, 层数: {num_layers}")
        elif encoder_type.startswith("esm2"):
            from .pretrained_encoders import ESM2Encoder
            model_size = encoder_type.split("_")[1] if "_" in encoder_type else "150M"
            encoder = ESM2Encoder(
                model_size=model_size,
                freeze=freeze_encoder,
                cache_dir=pretrained_cache_dir,
            )
            self.embed_dim = encoder.hidden_dim
            logger.info(f"使用ESM-2 {model_size} 预训练编码器，隐藏维度: {self.embed_dim}")
        elif encoder_type == "protbert":
            from .pretrained_encoders import ProtBERTEncoder
            encoder = ProtBERTEncoder(
                freeze=freeze_encoder,
                cache_dir=pretrained_cache_dir,
            )
            self.embed_dim = encoder.hidden_dim
            logger.info(f"使用ProtBERT预训练编码器，隐藏维度: {self.embed_dim}")
        elif encoder_type == "prott5":
            from .pretrained_encoders import ProtT5Encoder
            encoder = ProtT5Encoder(
                freeze=freeze_encoder,
                cache_dir=pretrained_cache_dir,
            )
            self.embed_dim = encoder.hidden_dim
            logger.info(f"使用ProtT5预训练编码器，隐藏维度: {self.embed_dim}")
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        self.encoder = encoder

        # PTM模块
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
        if task_type == "classification":
            self.predictor: CellStatePredictor = ClassificationPredictor(
                input_dim=self.embed_dim,
                num_classes=num_classes,
                hidden_dims=[self.embed_dim * 2, self.embed_dim],
                dropout=dropout,
            )
        elif task_type == "regression":
            self.predictor = RegressionPredictor(
                input_dim=self.embed_dim,
                output_dim=num_classes,
                hidden_dims=[self.embed_dim * 2, self.embed_dim],
                dropout=dropout,
            )
        elif task_type == "multitask":
            if multitask_configs is None:
                raise ValueError("多任务模式需要提供multitask_configs")
            self.predictor = MultiTaskPredictor(
                input_dim=self.embed_dim,
                task_configs=multitask_configs,
                hidden_dims=[self.embed_dim * 2, self.embed_dim],
                dropout=dropout,
            )
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

        # DAVF branch initialization
        self.use_davf = use_davf
        self.davf_config = dict(davf_config or {})
        self.davf_module: Optional[DAVFInferenceModule] = None
        self.ptm_mapper: Optional[PTMDirectionMapper] = None
        self.davf_feature_dim: int = 0

        if use_davf:
            if task_type == "multitask":
                raise ValueError("DAVF integration not supported with multitask mode")

            davf_config = self.davf_config
            davf_inference_config = DAVFInferenceConfig(
                state_space=davf_config.get("state_space", "scvi_latent"),
                checkpoint_path=davf_config.get("checkpoint_path", "checkpoints/latent_davf_ibd_norman/best_model.pt"),
                feature_dim=davf_config.get("feature_dim", 128),
                hidden_dim=davf_config.get("hidden_dim", 256),
                freeze=davf_config.get("freeze", True),
                latent_dim=davf_config.get("latent_dim", 10),
                num_genes=davf_config.get("num_genes", 5000),
                num_steps=davf_config.get("num_steps", 50),
                device=davf_config.get("device", None),
            )

            self.davf_module = DAVFInferenceModule(davf_inference_config)
            self.ptm_mapper = PTMDirectionMapper()
            self.davf_feature_dim = davf_inference_config.feature_dim

            combined_input_dim = self.embed_dim + self.davf_feature_dim
            if task_type == "classification":
                self.predictor = ClassificationPredictor(
                    input_dim=combined_input_dim,
                    num_classes=num_classes,
                    hidden_dims=[combined_input_dim * 2, combined_input_dim],
                    dropout=dropout,
                )
            elif task_type == "regression":
                self.predictor = RegressionPredictor(
                    input_dim=combined_input_dim,
                    output_dim=num_classes,
                    hidden_dims=[combined_input_dim * 2, combined_input_dim],
                    dropout=dropout,
                )

            logger.info(f"DAVF branch enabled: feature_dim={self.davf_feature_dim}, combined_input_dim={combined_input_dim}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

        ptm_types = batch.get("ptm_types", torch.zeros(batch_size, seq_len, dtype=torch.long, device=device))
        ptm_mask = batch.get("ptm_mask")
        if ptm_mask is None:
            ptm_mask = (ptm_types > 0).float()

        ptm_positions = batch.get(
            "ptm_positions",
            torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1),
        )

        fused = self.ptm_module(seq_emb, ptm_types, ptm_positions, ptm_mask)

        if isinstance(self.pooling, nn.Module):
            pooled = self.pooling(fused, attention_mask)
        else:
            pooled = fused.mean(dim=1)

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
                davf_features = torch.zeros(
                    B, self.davf_feature_dim,
                    device=device,
                    dtype=pooled.dtype,
                )

            pooled = torch.cat([pooled, davf_features], dim=-1)

        output = self.predictor(pooled)

        if self.task_type == "multitask":
            return cast(Dict[str, torch.Tensor], output)

        return cast(Dict[str, torch.Tensor], output)

    def get_model_info(self) -> Dict[str, Any]:
        from .model_utils import count_parameters

        param_stats = count_parameters(self)

        return {
            "variant": "large",
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
    def from_config(cls, config: Dict[str, Any]) -> "PTM2CellNetLarge":
        from .model_utils import validate_model_config

        is_valid, errors = validate_model_config(config)
        if not is_valid:
            raise ValueError(f"配置验证失败: {errors}")

        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        # "large" config overrides — higher defaults than base PTM2CellNet
        large_cfg = model_cfg.get("large", {})
        encoder_type = large_cfg.get("encoder_type", model_cfg.get("encoder_type", "transformer"))
        embed_dim = large_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 256))
        num_layers = large_cfg.get("num_layers", model_cfg.get("num_layers", 4))
        num_heads = large_cfg.get("num_heads", model_cfg.get("num_heads", 8))
        dropout = large_cfg.get("dropout", model_cfg.get("dropout", 0.1))
        freeze_encoder = large_cfg.get("freeze_encoder", model_cfg.get("freeze_encoder", False))
        pretrained_cache_dir = large_cfg.get("pretrained_cache_dir", model_cfg.get("pretrained_cache_dir", None))
        ptm_fusion_type = large_cfg.get("ptm_fusion_type", model_cfg.get("ptm_fusion_type", "attention"))
        task_type = large_cfg.get("task_type", model_cfg.get("task_type", "classification"))
        pool_type = large_cfg.get("pool_type", model_cfg.get("pool_type", "mean"))
        multitask_configs = large_cfg.get("multitask_configs", model_cfg.get("multitask_configs", None))

        vocab_size = large_cfg.get("vocab_size", model_cfg.get("vocab_size", len(data_cfg.get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY")) + 1))
        max_seq_len = data_cfg.get("max_sequence_length", 1000)
        num_ptm_types = len(data_cfg.get("ptm_types", [])) or model_cfg.get("num_ptm_types", 10)
        num_classes = large_cfg.get("num_classes", model_cfg.get("num_classes", 4))

        # DAVF configuration
        use_davf = large_cfg.get("use_davf", model_cfg.get("use_davf", False))
        davf_config = large_cfg.get("davf_config", model_cfg.get("davf_config", None))
        merged_davf_config = None

        if use_davf:
            merged_davf_config = cls._load_davf_config_defaults()
            if davf_config is not None:
                merged_davf_config.update(davf_config)

            merged_davf_config["scvi_model_path"] = merged_davf_config.get("scvi_model_path")
            merged_davf_config["geneformer_path"] = merged_davf_config.get("geneformer_path")
            davf_config = merged_davf_config

        return cls(
            encoder_type=encoder_type,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            num_ptm_types=num_ptm_types,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
            pretrained_cache_dir=pretrained_cache_dir,
            ptm_fusion_type=ptm_fusion_type,
            task_type=task_type,
            pool_type=pool_type,
            multitask_configs=multitask_configs,
            use_davf=use_davf,
            davf_config=merged_davf_config if use_davf else davf_config,
        )


class PTM2CellNet(nn.Module):
    """PTM2CellNet端到端模型"""

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
        embed_dim: int = 128,
        max_seq_len: int = 1000,
        num_ptm_types: int = 10,
        num_classes: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        pretrained_cache_dir: Optional[str] = None,
        ptm_fusion_type: str = "attention",
        task_type: str = "classification",
        pool_type: str = "mean",
        multitask_configs: Optional[list] = None,
        use_davf: bool = False,
        davf_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        encoder_type = encoder_type.lower()
        self.encoder_type = encoder_type
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.task_type = task_type
        self.pool_type = pool_type

        encoder: nn.Module
        if encoder_type == "cnn":
            encoder = CNNEncoder(vocab_size, embed_dim, max_len=max_seq_len, dropout=dropout)
        elif encoder_type == "transformer":
            encoder = TransformerEncoder(
                vocab_size, embed_dim, max_len=max_seq_len, num_layers=num_layers, num_heads=num_heads, dropout=dropout
            )
        elif encoder_type == "lstm":
            encoder = LSTMEncoder(
                vocab_size,
                embed_dim,
                max_len=max_seq_len,
                hidden_dim=embed_dim * 2,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif encoder_type == "gru":
            encoder = GRUEncoder(
                vocab_size,
                embed_dim,
                max_len=max_seq_len,
                hidden_dim=embed_dim * 2,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif encoder_type == "mamba":
            encoder = MambaEncoder(
                vocab_size=vocab_size,
                hidden_dim=embed_dim,
                num_layers=num_layers,
                state_dim=16,
                dropout=dropout,
                max_len=max_seq_len,
            )
            logger.info(f"使用Mamba编码器，隐藏维度: {embed_dim}, 层数: {num_layers}")
        # 预训练编码器（延迟导入，避免强制依赖transformers）
        elif encoder_type.startswith("esm2"):
            from .pretrained_encoders import ESM2Encoder
            model_size = encoder_type.split("_")[1] if "_" in encoder_type else "150M"
            encoder = ESM2Encoder(
                model_size=model_size,
                freeze=freeze_encoder,
                cache_dir=pretrained_cache_dir,
            )
            self.embed_dim = encoder.hidden_dim  # 更新embed_dim
            logger.info(f"使用ESM-2 {model_size} 预训练编码器，隐藏维度: {self.embed_dim}")
        elif encoder_type == "protbert":
            from .pretrained_encoders import ProtBERTEncoder
            encoder = ProtBERTEncoder(
                freeze=freeze_encoder,
                cache_dir=pretrained_cache_dir,
            )
            self.embed_dim = encoder.hidden_dim
            logger.info(f"使用ProtBERT预训练编码器，隐藏维度: {self.embed_dim}")
        elif encoder_type == "prott5":
            from .pretrained_encoders import ProtT5Encoder
            encoder = ProtT5Encoder(
                freeze=freeze_encoder,
                cache_dir=pretrained_cache_dir,
            )
            self.embed_dim = encoder.hidden_dim
            logger.info(f"使用ProtT5预训练编码器，隐藏维度: {self.embed_dim}")
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        self.encoder = encoder

        # PTM模块
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
        if task_type == "classification":
            self.predictor: CellStatePredictor = ClassificationPredictor(
                input_dim=self.embed_dim,
                num_classes=num_classes,
                hidden_dims=[self.embed_dim * 2, self.embed_dim],
                dropout=dropout,
            )
        elif task_type == "regression":
            self.predictor = RegressionPredictor(
                input_dim=self.embed_dim,
                output_dim=num_classes,
                hidden_dims=[self.embed_dim * 2, self.embed_dim],
                dropout=dropout,
            )
        elif task_type == "multitask":
            if multitask_configs is None:
                raise ValueError("多任务模式需要提供multitask_configs")
            self.predictor = MultiTaskPredictor(
                input_dim=self.embed_dim,
                task_configs=multitask_configs,
                hidden_dims=[self.embed_dim * 2, self.embed_dim],
                dropout=dropout,
            )
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

        # DAVF branch initialization (D-08, D-09, D-10)
        self.use_davf = use_davf
        self.davf_config = dict(davf_config or {})
        self.davf_module: Optional[DAVFInferenceModule] = None
        self.ptm_mapper: Optional[PTMDirectionMapper] = None
        self.davf_feature_dim: int = 0

        if use_davf:
            if task_type == "multitask":
                raise ValueError("DAVF integration not supported with multitask mode")

            davf_config = self.davf_config
            davf_inference_config = DAVFInferenceConfig(
                state_space=davf_config.get("state_space", "scvi_latent"),
                checkpoint_path=davf_config.get("checkpoint_path", "checkpoints/latent_davf_ibd_norman/best_model.pt"),
                feature_dim=davf_config.get("feature_dim", 128),
                hidden_dim=davf_config.get("hidden_dim", 256),
                freeze=davf_config.get("freeze", True),
                latent_dim=davf_config.get("latent_dim", 10),
                num_genes=davf_config.get("num_genes", 5000),
                num_steps=davf_config.get("num_steps", 50),
                device=davf_config.get("device", None),
            )

            self.davf_module = DAVFInferenceModule(davf_inference_config)
            self.ptm_mapper = PTMDirectionMapper()
            self.davf_feature_dim = davf_inference_config.feature_dim

            # Re-create predictor with expanded input dimension (D-04)
            combined_input_dim = self.embed_dim + self.davf_feature_dim
            if task_type == "classification":
                self.predictor = ClassificationPredictor(
                    input_dim=combined_input_dim,
                    num_classes=num_classes,
                    hidden_dims=[combined_input_dim * 2, combined_input_dim],
                    dropout=dropout,
                )
            elif task_type == "regression":
                self.predictor = RegressionPredictor(
                    input_dim=combined_input_dim,
                    output_dim=num_classes,
                    hidden_dims=[combined_input_dim * 2, combined_input_dim],
                    dropout=dropout,
                )

            logger.info(f"DAVF branch enabled: feature_dim={self.davf_feature_dim}, combined_input_dim={combined_input_dim}")

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

        ptm_types = batch.get("ptm_types", torch.zeros(batch_size, seq_len, dtype=torch.long, device=device))
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
            # Get batch size and device from pooled
            B = pooled.shape[0]
            device = pooled.device

            # Extract DAVF features if PTM sites provided
            if "davf_sites" in batch and "davf_gene_names" in batch:
                # Map PTM sites to DAVF inputs
                mapper_output = self.ptm_mapper.map_batch(
                    batch["davf_sites"],
                    batch["davf_gene_names"],
                )

                # Get DAVF features
                davf_output = self.davf_module(mapper_output)
                davf_features = davf_output.davf_features.to(device)  # [B, 128]
            else:
                # No PTM sites provided - use zero features (D-18)
                davf_features = torch.zeros(
                    B, self.davf_feature_dim,
                    device=device,
                    dtype=pooled.dtype,
                )

            # Concatenate ESM-2 embedding with DAVF features (D-13)
            pooled = torch.cat([pooled, davf_features], dim=-1)  # [B, embed_dim + 128]

        output = self.predictor(pooled)

        if self.task_type == "multitask":
            return cast(Dict[str, torch.Tensor], output)

        return cast(Dict[str, torch.Tensor], output)

    def get_model_info(self) -> Dict[str, Any]:
        from .model_utils import count_parameters

        param_stats = count_parameters(self)

        return {
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
    def from_config(cls, config: Dict[str, Any]) -> "PTM2CellNet":
        from .model_utils import validate_model_config

        is_valid, errors = validate_model_config(config)
        if not is_valid:
            raise ValueError(f"配置验证失败: {errors}")

        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        encoder_type = model_cfg.get("encoder_type", "transformer")
        embed_dim = model_cfg.get("hidden_dim", 128)
        num_layers = model_cfg.get("num_layers", 2)
        num_heads = model_cfg.get("num_heads", 4)
        dropout = model_cfg.get("dropout", 0.1)
        freeze_encoder = model_cfg.get("freeze_encoder", False)
        pretrained_cache_dir = model_cfg.get("pretrained_cache_dir", None)
        ptm_fusion_type = model_cfg.get("ptm_fusion_type", "attention")
        task_type = model_cfg.get("task_type", "classification")
        pool_type = model_cfg.get("pool_type", "mean")
        multitask_configs = model_cfg.get("multitask_configs", None)

        vocab_size = model_cfg.get("vocab_size", len(data_cfg.get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY")) + 1)
        max_seq_len = data_cfg.get("max_sequence_length", 1000)
        num_ptm_types = len(data_cfg.get("ptm_types", [])) or model_cfg.get("num_ptm_types", 10)
        num_classes = model_cfg.get("num_classes", 4)

        # DAVF configuration (Phase 13)
        use_davf = model_cfg.get("use_davf", False)
        davf_config = model_cfg.get("davf_config", None)
        merged_davf_config = None

        if use_davf:
            merged_davf_config = cls._load_davf_config_defaults()
            if davf_config is not None:
                merged_davf_config.update(davf_config)

            merged_davf_config["scvi_model_path"] = merged_davf_config.get("scvi_model_path")
            merged_davf_config["geneformer_path"] = merged_davf_config.get("geneformer_path")
            davf_config = merged_davf_config

        return cls(
            encoder_type=encoder_type,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            num_ptm_types=num_ptm_types,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
            pretrained_cache_dir=pretrained_cache_dir,
            ptm_fusion_type=ptm_fusion_type,
            task_type=task_type,
            pool_type=pool_type,
            multitask_configs=multitask_configs,
            use_davf=use_davf,
            davf_config=merged_davf_config if use_davf else davf_config,
        )
