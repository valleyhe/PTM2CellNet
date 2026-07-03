"""
数据模型模块
功能概述: 定义API请求和响应的Pydantic数据模型
设计思路: 使用Pydantic进行数据验证和序列化
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PTMSite(BaseModel):
    """
    PTM位点模型
    表示单个蛋白质翻译后修饰位点

    ``gene_symbol`` is optional but required when the request enables DAVF
    inference (DAVF plan Task 5): without a gene identifier the DAVF branch
    cannot map the PTM site to a gene latent. It is accepted (and ignored)
    by the non-DAVF path for backwards compatibility.
    """
    position: int = Field(..., description="PTM位点在序列中的位置（从1开始）", ge=1)
    type: str = Field(..., description="PTM类型，如phosphorylation、acetylation等")
    amino_acid: Optional[str] = Field(None, description="该位点的氨基酸")
    gene_symbol: Optional[str] = Field(
        None,
        description="基因符号（如BRAF、TP53）。启用 DAVF 推理（use_davf=true）时必填，"
        "用于将 PTM 位点映射到 DAVF 基因潜在空间。",
    )


class PredictionRequest(BaseModel):
    """
    单样本预测请求模型
    """
    sequence: str = Field(..., description="蛋白质氨基酸序列")
    ptm_sites: Optional[List[PTMSite]] = Field(default_factory=list, description="PTM位点列表")
    use_davf: bool = Field(
        False,
        description="是否启用 DAVF（PTM→细胞状态方向感知向量）特征。"
        "启用后，模型会联合序列嵌入与 DAVF 特征进行预测；"
        "ptm_sites 中的 gene_symbol 字段将被传递给 DAVF 模块。",
    )


class BatchPredictionRequest(BaseModel):
    """
    批量预测请求模型
    """
    samples: List[PredictionRequest] = Field(..., description="样本列表")


# Variant prediction schemas (FEAT-01, FEAT-02)

class PTMEffect(BaseModel):
    """PTM effect prediction for a specific modification type."""
    ptm_type: str = Field(..., description="PTM type (Phosphorylation, Ubiquitination, etc.)")
    wildtype_prob: float = Field(..., description="PTM probability in wildtype", ge=0.0, le=1.0)
    mutant_prob: float = Field(..., description="PTM probability in mutant", ge=0.0, le=1.0)
    delta_prob: float = Field(..., description="Change in probability", ge=-1.0, le=1.0)
    effect: str = Field(..., description="Effect classification: gain/loss/neutral/error")


class PathwayImpact(BaseModel):
    """Signal pathway impact from variant."""
    pathway_name: str = Field(..., description="Pathway name")
    activity_change: float = Field(..., description="Predicted activity change")
    confidence: str = Field(..., description="Confidence level: high/medium/low")
    key_genes: List[str] = Field(default_factory=list, description="Affected key genes")


class PredictionResponse(BaseModel):
    """
    单样本预测响应模型
    """
    predicted_cell_state: str = Field(..., description="预测的细胞状态")
    cell_state: str = Field(..., description="预测的细胞状态")
    confidence: float = Field(..., description="预测置信度", ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(..., description="各类别的概率分布")
    pathway_impacts: Optional[List["PathwayImpact"]] = Field(None, description="信号通路影响分析")
    processing_time_ms: Optional[float] = Field(None, description="处理时间（毫秒）")


class BatchPredictionResponse(BaseModel):
    """
    批量预测响应模型
    """
    predictions: List[PredictionResponse] = Field(..., description="预测结果列表")
    total_processing_time_ms: Optional[float] = Field(None, description="总处理时间（毫秒）")
    sample_count: int = Field(..., description="处理的样本数量")


class HealthResponse(BaseModel):
    """
    健康检查响应模型
    """
    status: str = Field(..., description="服务状态，'healthy' 或 'unhealthy'")
    version: str = Field(..., description="API版本")
    model_loaded: bool = Field(..., description="模型是否已加载")
    timestamp: str = Field(..., description="响应时间戳")


class LiveResponse(BaseModel):
    """
    存活检查响应模型
    """
    status: str = Field(..., description="服务存活状态")


class ReadyResponse(BaseModel):
    """
    就绪检查响应模型
    """
    status: str = Field(..., description="服务就绪状态")
    model_loaded: bool = Field(..., description="模型是否已加载")


class VariantReadyResponse(BaseModel):
    """
    变体预测就绪探针响应模型（P1-1）

    核心模型已加载但变体 workflow 不可用时，``/predict`` 可用而
    ``/predict/variant`` 不可用。该响应让运维/K8s 单独判断变体能力就绪状态。
    """
    status: str = Field(..., description="变体能力就绪状态：'ready' 或 'not_ready'")
    variant_workflow_loaded: bool = Field(..., description="变体预测 workflow 是否已加载")
    model_loaded: bool = Field(..., description="核心模型是否已加载")


class ModelInfoResponse(BaseModel):
    """
    模型信息响应模型
    """
    model_name: str = Field(..., description="模型名称")
    model_version: str = Field(..., description="模型版本")
    encoder_type: str = Field(..., description="编码器类型")
    embed_dim: int = Field(..., description="嵌入维度")
    num_classes: int = Field(..., description="类别数量")
    cell_states: List[str] = Field(..., description="支持的细胞状态列表")
    supported_ptm_types: List[str] = Field(..., description="支持的PTM类型列表")
    # P1-1: capability flags so callers know which endpoints are usable.
    variant_workflow_loaded: bool = Field(
        False, description="变体预测 workflow 是否已加载"
    )
    pathway_mapper_loaded: bool = Field(
        False, description="信号通路分析器是否已加载"
    )
    # P1-3: provenance — surface demo vs real so the model is not misused.
    model_kind: Optional[str] = Field(
        None, description="模型性质：'demo' / 'real' / None（未知）"
    )
    is_demo_model: bool = Field(
        False, description="是否为合成数据训练的 demo 模型（不可用于真实生物学预测）"
    )
    checkpoint_path: Optional[str] = Field(None, description="当前加载的 checkpoint 路径")
    config_path: Optional[str] = Field(None, description="当前加载的配置文件路径")


class VariantInfo(BaseModel):
    """Variant information."""
    hgvs: str = Field(..., description="HGVS notation")
    gene_symbol: Optional[str] = Field(None, description="Gene symbol")
    uniprot_id: Optional[str] = Field(None, description="UniProt ID")
    position: int = Field(..., description="Variant position (1-based)", ge=1)
    ref_aa: str = Field(..., description="Reference amino acid")
    alt_aa: str = Field(..., description="Alternate amino acid")


class VariantPredictionRequest(BaseModel):
    """Variant effect prediction request."""
    hgvs: str = Field(..., description="HGVS variant notation (e.g., 'BRAF:p.V600E')")
    sequence: Optional[str] = Field(None, description="Protein sequence (optional)")
    uniprot_id: Optional[str] = Field(None, description="UniProt ID for sequence lookup")
    include_pathways: bool = Field(True, description="Include pathway impact analysis")


class VariantPredictionResponse(BaseModel):
    """Variant effect prediction response."""
    variant: VariantInfo = Field(..., description="Parsed variant information")
    ptm_effects: List[PTMEffect] = Field(..., description="PTM effects by type")
    pathway_impacts: Optional[List[PathwayImpact]] = Field(None, description="Pathway impacts")
    cell_state_prediction: Optional[str] = Field(None, description="Predicted cell state change")
    confidence: float = Field(..., description="Overall confidence", ge=0.0, le=1.0)
    processing_time_ms: Optional[float] = Field(None, description="Processing time")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")


# 解析 PredictionResponse 中对 PathwayImpact 的前向引用。
# 在 Pydantic v1 下未调用 update_forward_refs() 会抛出 ConfigError，
# 在 Pydantic v2 下则通过 model_rebuild() 完成解析。两者均安全调用。
if hasattr(PredictionResponse, "update_forward_refs"):
    PredictionResponse.update_forward_refs()
if hasattr(PredictionResponse, "model_rebuild"):
    PredictionResponse.model_rebuild()
