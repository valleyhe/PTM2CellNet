"""模型模块 - 编码器、PTM模块、预测器和整体架构"""

from .encoders import (
    SequenceEncoder,
    CNNEncoder,
    TransformerEncoder,
    LSTMEncoder,
    PositionalEncoding,
)
from .ptm_modules import PTMEmbedding, PTMAttention, PTMModule, GatedPTMFusion
from .predictors import (
    CellStatePredictor,
    ClassificationPredictor,
    RegressionPredictor,
)
from .pooling import (
    AttentionPooling,
    MultiHeadAttentionPooling,
    WeightedMeanPooling,
    create_pooling_layer,
)
from .multitask import (
    MultiTaskPredictor,
    HierarchicalMultiTaskPredictor,
)
from .architectures import PTM2CellNet
from .model_utils import (
    validate_model_config,
    count_parameters,
    get_model_memory_usage,
    estimate_max_batch_size,
    print_model_summary,
    compare_models,
)

# DAVF Integration exports (Phase 11, 12)
from .ptm_direction_mapper import (
    PTMDirectionMapper,
    PTMDirectionMapperOutput,
    PTM_DIRECTION_MAP,
    DEFAULT_DIRECTION,
    MAX_TARGETS,
)
from .davf_inference import (
    DAVFInferenceModule,
    DAVFInferenceConfig,
    DAVFInferenceOutput,
    DeltaProjection,
)

__all__ = [
    # Encoders
    "SequenceEncoder",
    "CNNEncoder",
    "TransformerEncoder",
    "LSTMEncoder",
    "PositionalEncoding",
    # PTM Modules
    "PTMEmbedding",
    "PTMAttention",
    "PTMModule",
    "GatedPTMFusion",
    # Predictors
    "CellStatePredictor",
    "ClassificationPredictor",
    "RegressionPredictor",
    # Pooling
    "AttentionPooling",
    "MultiHeadAttentionPooling",
    "WeightedMeanPooling",
    "create_pooling_layer",
    # Multitask
    "MultiTaskPredictor",
    "HierarchicalMultiTaskPredictor",
    # Architecture
    "PTM2CellNet",
    # Utils
    "validate_model_config",
    "count_parameters",
    "get_model_memory_usage",
    "estimate_max_batch_size",
    "print_model_summary",
    "compare_models",
    # DAVF Integration
    "PTMDirectionMapper",
    "PTMDirectionMapperOutput",
    "PTM_DIRECTION_MAP",
    "DEFAULT_DIRECTION",
    "MAX_TARGETS",
    "DAVFInferenceModule",
    "DAVFInferenceConfig",
    "DAVFInferenceOutput",
    "DeltaProjection",
]
