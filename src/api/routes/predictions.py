# mypy: ignore-errors
"""Prediction endpoints."""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, status

from ..schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PTMEffect,
    PathwayImpact,
    PredictionRequest,
    PredictionResponse,
    VariantInfo,
    VariantPredictionRequest,
    VariantPredictionResponse,
)
from ...data.features import DEFAULT_AMINO_ACIDS
from ...utils.helpers import clean_sequence, validate_ptm_site, validate_sequence
from ...utils.logging import setup_logger
from .state import STATE

logger = setup_logger(__name__)

router = APIRouter()


def preprocess_request(request: PredictionRequest) -> Dict[str, torch.Tensor]:
    """
    预处理预测请求

    参数:
        request: 预测请求

    返回:
        预处理后的批次字典
    """
    sequence = clean_sequence(request.sequence)

    amino_acids = STATE.feature_extractor.amino_acids if STATE.feature_extractor else DEFAULT_AMINO_ACIDS
    # Always use 1-based encoding (A=1, Y=20) to match FeatureExtractor
    aa_to_idx = (
        STATE.feature_extractor.aa_to_idx
        if STATE.feature_extractor
        else {aa: i + 1 for i, aa in enumerate(amino_acids)}
    )

    is_valid, error_msg = validate_sequence(
        sequence,
        max_length=STATE.max_sequence_length,
        valid_amino_acids=set(amino_acids),
    )
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效序列: {error_msg}",
        )

    max_len = STATE.max_sequence_length
    seq_len = min(len(sequence), max_len)
    seq_tensor = torch.zeros(max_len, dtype=torch.long)

    for i in range(seq_len):
        aa = sequence[i]
        aa_idx = aa_to_idx.get(aa)
        if aa_idx is not None:
            seq_tensor[i] = aa_idx

    ptm_mask = torch.zeros(max_len, dtype=torch.float32)
    ptm_types = torch.zeros(max_len, dtype=torch.long)

    for ptm_site in request.ptm_sites or []:
        site_dict = ptm_site.model_dump() if hasattr(ptm_site, "model_dump") else ptm_site.dict()
        is_valid, error_msg = validate_ptm_site(site_dict, seq_len)
        if not is_valid:
            logger.warning("跳过无效PTM位点: %s", error_msg)
            continue

        pos = ptm_site.position - 1
        if 0 <= pos < max_len:
            ptm_mask[pos] = 1.0
            ptm_idx = STATE.ptm_type_to_idx.get(ptm_site.type)
            if ptm_idx is not None:
                ptm_types[pos] = ptm_idx

    return {
        "sequence": seq_tensor,
        "ptm_mask": ptm_mask,
        "ptm_types": ptm_types,
    }


def _run_prediction_on_batch(
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run model inference on a batched tensor dict (already on device).

    Returns:
        (probabilities, predictions, confidence_scores) — all on CPU.
        probabilities: (N, num_classes)
        predictions: (N,) argmax class indices
        confidence_scores: (N,) max probability per sample
    """
    with torch.no_grad():
        outputs = STATE.model(batch)

        if isinstance(outputs, dict):
            probabilities = outputs.get("probabilities")
        else:
            probabilities = torch.softmax(outputs, dim=-1)

        if not isinstance(probabilities, torch.Tensor):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="模型输出缺少有效概率张量",
            )

    probs = probabilities.cpu()
    predictions = torch.argmax(probs, dim=-1)
    confidence_scores = probs.gather(dim=-1, index=predictions.unsqueeze(-1)).squeeze(-1)
    return probs, predictions, confidence_scores


def _build_prediction_response(
    probs_np: np.ndarray,
    pred_idx: int,
    timing_ms: float,
    pathway_impacts: Optional[List[PathwayImpact]] = None,
) -> PredictionResponse:
    """Build a single PredictionResponse from model output tensors."""
    pred_label = STATE.idx_to_label.get(pred_idx, "unknown")
    confidence = float(probs_np[pred_idx])
    prob_dict = {
        STATE.idx_to_label.get(i, f"class_{i}"): float(probs_np[i])
        for i in range(len(probs_np))
    }
    return PredictionResponse(
        predicted_cell_state=pred_label,
        cell_state=pred_label,
        confidence=confidence,
        probabilities=prob_dict,
        pathway_impacts=pathway_impacts,
        processing_time_ms=round(timing_ms, 2),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    预测细胞状态

    参数:
        request: 预测请求，包含序列和PTM位点

    返回:
        预测响应，包含细胞状态、置信度和概率分布
    """
    if STATE.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型未初始化",
        )

    start_time = time.time()

    try:
        batch = preprocess_request(request)

        # Add batch dimension for single-sample inference
        batch = {key: val.unsqueeze(0).to(STATE.device) for key, val in batch.items()}

        probs_tensor, pred_tensor, _ = _run_prediction_on_batch(batch)

        probs_np = probs_tensor[0].numpy()
        pred_idx = int(pred_tensor[0].item())

        # Optional pathway analysis via SignalingNetworkMapper
        pathway_impacts = None
        if STATE.pathway_mapper is not None and request.ptm_sites:
            try:
                import pandas as pd

                rows = []
                for ptm_site in request.ptm_sites:
                    site = ptm_site.model_dump() if hasattr(ptm_site, "model_dump") else ptm_site.dict()
                    rows.append({
                        "gene_symbol": "",
                        "ptm_type": site.get("type", ""),
                        "effect": "gain",
                        "delta_prob": 1.0,
                    })
                if rows:
                    ptm_df = pd.DataFrame(rows)
                    report = STATE.pathway_mapper.generate_network_report(ptm_df)
                    if report and report.get("pathway_activities"):
                        pathway_impacts = [
                            PathwayImpact(
                                pathway_name=name,
                                activity_change=activity,
                                confidence="high" if abs(activity) > 0.5 else "medium",
                                key_genes=list(
                                    STATE.pathway_mapper.pathways.get(name, {}).get("output_genes", [])
                                ),
                            )
                            for name, activity in report["pathway_activities"].items()
                        ]
            except Exception as exc:
                logger.warning("Optional pathway analysis failed: %s", exc)

        processing_time_ms = (time.time() - start_time) * 1000

        return _build_prediction_response(probs_np, pred_idx, processing_time_ms, pathway_impacts)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("预测错误: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="预测过程中发生错误",
        ) from e


@router.post("/predict/batch", response_model=BatchPredictionResponse)
@router.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    批量预测细胞状态

    参数:
        request: 批量预测请求，包含样本列表

    返回:
        批量预测响应
    """
    if STATE.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型未初始化",
        )

    # M4: 空样本列表直接返回空结果，而非在 torch.stack([]) 时抛 500
    if not request.samples:
        return BatchPredictionResponse(
            predictions=[],
            total_processing_time_ms=0.0,
            sample_count=0,
        )

    start_time = time.time()

    try:
        batches = [preprocess_request(sample) for sample in request.samples]

        batch = {
            "sequence": torch.stack([b["sequence"] for b in batches], dim=0).to(STATE.device),
            "ptm_mask": torch.stack([b["ptm_mask"] for b in batches], dim=0).to(STATE.device),
            "ptm_types": torch.stack([b["ptm_types"] for b in batches], dim=0).to(STATE.device),
        }

        probs_tensor, pred_tensor, _ = _run_prediction_on_batch(batch)

        total_time_ms = (time.time() - start_time) * 1000
        per_sample_ms = total_time_ms / len(request.samples) if request.samples else 0

        predictions = [
            _build_prediction_response(
                probs_tensor[i].numpy(),
                int(pred_tensor[i].item()),
                per_sample_ms,
            )
            for i in range(len(request.samples))
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=round(total_time_ms, 2),
            sample_count=len(predictions),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("批量预测错误: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="批量预测过程中发生错误",
        ) from e


@router.post("/predict/variant", response_model=VariantPredictionResponse)
async def predict_variant(request: VariantPredictionRequest):
    """
    Predict cell state changes from protein variant.

    Args:
        request: Variant prediction request with HGVS notation

    Returns:
        Variant effect prediction with PTM and pathway impacts
    """
    if STATE.variant_workflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variant prediction workflow not initialized",
        )

    start_time = time.time()
    warnings_list = []

    try:
        sequence = request.sequence
        if sequence is None and request.uniprot_id:
            try:
                sequence = STATE.variant_workflow.fetch_sequence_from_uniprot(request.uniprot_id)
                logger.info(f"Fetched sequence from UniProt for {request.uniprot_id}")
            except Exception as e:
                warnings_list.append(f"Failed to fetch sequence from UniProt: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not fetch sequence for {request.uniprot_id}: {e}",
                ) from e

        if sequence is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sequence required (either provide directly or UniProt ID)",
            )

        result = STATE.variant_workflow.predict_from_hgvs(
            hgvs_string=request.hgvs,
            sequence=sequence,
            include_pathways=request.include_pathways,
        )

        ptm_effects = [
            PTMEffect(
                ptm_type=ptm_type,
                wildtype_prob=effect["wildtype_prob"],
                mutant_prob=effect["mutant_prob"],
                delta_prob=effect["delta_prob"],
                effect=effect["effect"],
            )
            for ptm_type, effect in result.ptm_effects.items()
        ]

        pathway_impacts = None
        if request.include_pathways and result.pathway_impacts:
            pathway_impacts = [
                PathwayImpact(
                    pathway_name=name,
                    activity_change=data.get("activity", 0.0),
                    confidence="high" if abs(data.get("activity", 0)) > 0.5 else "medium",
                    key_genes=data.get("genes", []),
                )
                for name, data in result.pathway_impacts.items()
            ]

        max_delta = max((abs(e.delta_prob) for e in ptm_effects), default=0.0)
        confidence = min(max_delta * 2, 1.0)

        processing_time_ms = (time.time() - start_time) * 1000

        return VariantPredictionResponse(
            variant=VariantInfo(
                hgvs=request.hgvs,
                gene_symbol=result.variant.get("gene_symbol"),
                uniprot_id=request.uniprot_id,
                position=result.variant.get("position", 0),
                ref_aa=result.variant.get("ref_aa", ""),
                alt_aa=result.variant.get("alt_aa", ""),
            ),
            ptm_effects=ptm_effects,
            pathway_impacts=pathway_impacts,
            confidence=confidence,
            processing_time_ms=round(processing_time_ms, 2),
            warnings=warnings_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Variant prediction error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Variant prediction failed: {str(e)}",
        ) from e


__all__ = ["router", "preprocess_request", "predict", "batch_predict", "predict_variant"]
