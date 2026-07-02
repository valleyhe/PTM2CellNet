"""模型模块 - 编码器、PTM模块、预测器和整体架构"""

from .encoders import (
    SequenceEncoder,
    CNNEncoder,
    TransformerEncoder,
    LSTMEncoder,
    GRUEncoder,
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
from .architectures import PTM2CellNet, PTM2CellNetLarge
from .ensemble import PTM2CellNetEnsemble
from .geneformer_embedding import GeneformerEmbeddingLoader
from .davf_checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    resume_training_state,
)
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
    register_ptm_direction,
    register_pathway_context_override,
)
from .davf_inference import (
    DAVFInferenceModule,
    DAVFInferenceConfig,
    DAVFInferenceOutput,
    DeltaProjection,
)
from .delta_predictor import DeltaPredictor

from .external_tools import (
    AlphaFoldClient,
    BLASTClient,
    ClustalWClient,
    PSIPREDClient,
)

# scVI adapter (V22-02) - gene<->latent mapping; scvi-tools is an optional dep.
# The module imports cleanly even when scvi-tools is absent (SCVI_AVAILABLE flag).
from .scvi_adapter import ScVIAdapter, ScVIAdapterConfig, SCVI_AVAILABLE

# Deferred v1.0 roadmap features (V2-01..V2-05): documented stubs.
from .roadmap import (
    get_deferred_features,
    DeferredFeature,
    DEFERRED_FEATURES,
)

__all__ = [
    # Encoders
    "SequenceEncoder",
    "CNNEncoder",
    "TransformerEncoder",
    "LSTMEncoder",
    "GRUEncoder",
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
    "PTM2CellNetLarge",
    # Ensemble
    "PTM2CellNetEnsemble",
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
    "register_ptm_direction",
    "register_pathway_context_override",
    "DAVFInferenceModule",
    "DAVFInferenceConfig",
    "DAVFInferenceOutput",
    "DeltaProjection",
    "DeltaPredictor",
    # External tool stubs
    "AlphaFoldClient",
    "BLASTClient",
    "ClustalWClient",
    "PSIPREDClient",
    "GeneformerEmbeddingLoader",
    "save_checkpoint",
    "load_checkpoint",
    "resume_training_state",
    # scVI adapter (V22-02)
    "ScVIAdapter",
    "ScVIAdapterConfig",
    "SCVI_AVAILABLE",
]
