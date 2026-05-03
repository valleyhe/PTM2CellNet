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
    """
    position: int = Field(..., description="PTM位点在序列中的位置（从1开始）", ge=1)
    type: str = Field(..., description="PTM类型，如phosphorylation、acetylation等")
    amino_acid: Optional[str] = Field(None, description="该位点的氨基酸")


class PredictionRequest(BaseModel):
    """
    单样本预测请求模型
    """
    sequence: str = Field(..., description="蛋白质氨基酸序列")
    ptm_sites: Optional[List[PTMSite]] = Field(default_factory=list, description="PTM位点列表")


class BatchPredictionRequest(BaseModel):
    """
    批量预测请求模型
    """
    samples: List[PredictionRequest] = Field(..., description="样本列表")


class PredictionResponse(BaseModel):
    """
    单样本预测响应模型
    """
    cell_state: str = Field(..., description="预测的细胞状态")
    confidence: float = Field(..., description="预测置信度", ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(..., description="各类别的概率分布")
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
