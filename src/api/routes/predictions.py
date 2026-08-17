"""Prediction endpoints."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool
from typing_extensions import TypedDict

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

# ---------------------------------------------------------------------------
# TypedDict definitions for structured dicts used in this module
# ---------------------------------------------------------------------------

class DAVFSiteDict(TypedDict):
    """Shape of a single DAVF site dict passed to the model."""

    position: int
    ptm_type: str


class DAVFInputs(TypedDict, total=False):
    """Shape of the dict returned by ``_collect_davf_inputs``."""

    davf_positions: List[int]
    davf_ptm_types: List[str]
    davf_gene_names: List[str]


# Mapping from PTM type to expected signaling effect direction.
# Lowercase keys match the PTM types used in DEFAULT_PTM_TYPES / API requests.
# "gain" = functional gain (e.g. activation), "loss" = functional loss (e.g. degradation).
PTM_TYPE_TO_EFFECT: Dict[str, str] = {
    "phosphorylation": "gain",    # phosphorylation often activates kinase/protein
    "ubiquitination": "loss",     # ubiquitination often targets protein for degradation
    "acetylation": "gain",        # acetylation can regulate function; default to gain
    "methylation": "gain",        # methylation typically modulates interactions
    "sumoylation": "gain",        # sumoylation regulates nuclear localization/transcription
    "succinylation": "gain",      # succinylation regulates metabolism
}

# Default effect/delta_prob when PTM type is unknown or missing
_DEFAULT_EFFECT = "loss"
_DEFAULT_DELTA_PROB = 0.5

router = APIRouter()


def _collect_davf_inputs(
    ptm_sites: List,
    max_len: int,
) -> DAVFInputs:
    """Collect DAVF inputs (sites + gene names) from validated PTM sites.

    Shared by single-sample and batch preprocessing. Each PTM site that
    carries both a position and a ``gene_symbol`` becomes one DAVF input:

    - ``davf_positions``: 1-based positions of the PTM sites (per sample list).
    - ``davf_ptm_types``: PTM type string per site (matches DAVF direction mapper).
    - ``davf_gene_names``: gene symbol per site.

    Sites without a ``gene_symbol`` are skipped (DAVF cannot map them) and a
    debug log is emitted so the caller can see why DAVF fell back to zeros.

    Returns an empty dict when there are no DAVF-eligible sites.
    """
    davf_positions: List[int] = []
    davf_ptm_types: List[str] = []
    davf_gene_names: List[str] = []
    for ptm_site in ptm_sites or []:
        site = ptm_site.model_dump() if hasattr(ptm_site, "model_dump") else ptm_site.dict()
        gene = site.get("gene_symbol")
        if not gene:
            logger.debug(
                "PTM site at position %s skipped by DAVF: gene_symbol missing",
                site.get("position"),
            )
            continue
        pos = site.get("position")
        if not isinstance(pos, int) or pos < 1 or pos > max_len:
            continue
        davf_positions.append(int(pos))
        davf_ptm_types.append(str(site.get("type", "")).lower())
        davf_gene_names.append(str(gene))

    if not davf_positions:
        return {}

    return {
        "davf_positions": davf_positions,
        "davf_ptm_types": davf_ptm_types,
        "davf_gene_names": davf_gene_names,
    }


def _model_supports_davf(request_flag: bool) -> bool:
    """True when DAVF features should be produced for this request.

    DAVF is enabled when the request explicitly opts in via ``use_davf`` *or*
    the loaded model was built with a DAVF branch (``STATE.model.use_davf``).
    The latter ensures the model receives DAVF features whenever it expects
    them, even if the caller forgot to set ``use_davf`` on the request.
    """
    if request_flag:
        return True
    model = STATE.model
    return bool(getattr(model, "use_davf", False))


def batch_preprocess(samples: List[PredictionRequest]) -> Dict[str, torch.Tensor]:
    """Vectorized preprocessing for a list of prediction requests.

    Builds sequence / ptm_mask / ptm_types tensors for all samples in a single
    pass, avoiding N separate ``preprocess_request`` Python loops. PTM site
    validation still runs per-sample (preserving the skip-on-invalid semantics
    of :func:`preprocess_request`), but sequence encoding (clean_sequence /
    validate_sequence / per-position tensor fill) is shared so the expensive
    parts are not repeated.

    When any sample requests DAVF (``request.use_davf``) or the loaded model
    has a DAVF branch, this also collects per-sample DAVF inputs and emits
    ``davf_sites`` / ``davf_gene_names`` batch keys expected by the DAVF-aware
    ``forward()`` in :class:`~src.models.architectures.PTM2CellNet`.

    参数:
        samples: 预测请求样本列表

    返回:
        堆叠后的批次字典，每个张量形状为 (N, max_sequence_length)，
        未移动到设备（调用方负责 ``.to(STATE.device)``）。

    抛出:
        HTTPException: 当某条样本序列无效时（与 preprocess_request 一致）。
    """
    if not samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="批量预处理样本列表为空",
        )

    amino_acids = STATE.feature_extractor.amino_acids if STATE.feature_extractor else DEFAULT_AMINO_ACIDS
    aa_to_idx = (
        STATE.feature_extractor.aa_to_idx
        if STATE.feature_extractor
        else {aa: i + 1 for i, aa in enumerate(amino_acids)}
    )
    max_len = STATE.max_sequence_length

    n = len(samples)
    seq_tensor = torch.zeros((n, max_len), dtype=torch.long)
    ptm_mask = torch.zeros((n, max_len), dtype=torch.float32)
    ptm_types = torch.zeros((n, max_len), dtype=torch.long)

    # DAVF per-sample inputs (heterogeneous lists — the DAVF branch consumes
    # a list-of-lists rather than a padded tensor).
    collect_davf = any(_model_supports_davf(getattr(s, "use_davf", False)) for s in samples)
    davf_sites_per_sample: List[List[DAVFSiteDict]] = []
    davf_gene_names_per_sample: List[List[str]] = []

    valid_amino_acids = set(amino_acids)
    for i, request in enumerate(samples):
        sequence = clean_sequence(request.sequence)
        is_valid, error_msg = validate_sequence(
            sequence,
            max_length=max_len,
            valid_amino_acids=valid_amino_acids,
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"样本 {i} 无效序列: {error_msg}",
            )

        seq_len = min(len(sequence), max_len)
        for j in range(seq_len):
            aa_idx = aa_to_idx.get(sequence[j])
            if aa_idx is not None:
                seq_tensor[i, j] = aa_idx

        valid_ptm_sites = []
        for ptm_site in request.ptm_sites or []:
            site_dict = ptm_site.model_dump() if hasattr(ptm_site, "model_dump") else ptm_site.dict()
            is_valid, error_msg = validate_ptm_site(site_dict, seq_len)
            if not is_valid:
                logger.warning("样本 %d 跳过无效PTM位点: %s", i, error_msg)
                continue

            pos = ptm_site.position - 1
            if 0 <= pos < max_len:
                ptm_mask[i, pos] = 1.0
                ptm_idx = STATE.ptm_type_to_idx.get(ptm_site.type)
                if ptm_idx is not None:
                    ptm_types[i, pos] = ptm_idx
            valid_ptm_sites.append(ptm_site)

        if collect_davf:
            davf_inputs = _collect_davf_inputs(valid_ptm_sites, max_len)
            if davf_inputs:
                sites_list = [
                    {"position": p, "ptm_type": t}
                    for p, t in zip(
                        davf_inputs["davf_positions"],
                        davf_inputs["davf_ptm_types"], strict=False,
                    )
                ]
                davf_sites_per_sample.append(sites_list)
                davf_gene_names_per_sample.append(davf_inputs["davf_gene_names"])
            else:
                davf_sites_per_sample.append([])
                davf_gene_names_per_sample.append([])

    out: Dict[str, torch.Tensor] = {
        "sequence": seq_tensor,
        "ptm_mask": ptm_mask,
        "ptm_types": ptm_types,
    }
    if collect_davf:
        out["davf_sites"] = davf_sites_per_sample
        out["davf_gene_names"] = davf_gene_names_per_sample
    return out


def _compute_pathway_impacts(
    ptm_sites: Optional[List],
    pathway_mapper,
) -> Optional[List[PathwayImpact]]:
    """Compute pathway impacts from PTM sites using a SignalingNetworkMapper.

    Shared by the single-sample ``/predict`` and batch endpoints so both return
    populated ``pathway_impacts`` when a mapper is initialized and PTM sites are
    present. Returns ``None`` when there are no PTM sites or the analysis fails.
    """
    if pathway_mapper is None or not ptm_sites:
        return None

    try:
        import pandas as pd

        rows = []
        for ptm_site in ptm_sites:
            site = ptm_site.model_dump() if hasattr(ptm_site, "model_dump") else ptm_site.dict()
            ptm_type_raw = site.get("type", "")
            ptm_type_lower = ptm_type_raw.lower() if ptm_type_raw else ""
            effect = PTM_TYPE_TO_EFFECT.get(ptm_type_lower, _DEFAULT_EFFECT)
            delta_prob = _DEFAULT_DELTA_PROB
            rows.append({
                "gene_symbol": site.get("gene_symbol", ""),
                "ptm_type": ptm_type_raw.capitalize() if ptm_type_raw else "",
                "effect": effect,
                "delta_prob": delta_prob,
            })
        if not rows:
            return None
        ptm_df = pd.DataFrame(rows)
        report = pathway_mapper.generate_network_report(ptm_df)
        if report and report.get("pathway_activities"):
            return [
                PathwayImpact(
                    pathway_name=name,
                    activity_change=activity,
                    confidence="high" if abs(activity) > 0.5 else "medium",
                    key_genes=list(
                        pathway_mapper.pathways.get(name, {}).get("output_genes", [])
                    ),
                )
                for name, activity in report["pathway_activities"].items()
            ]
    except (ValueError, KeyError, RuntimeError) as exc:
        logger.warning(
            "Optional pathway analysis failed [%s]: %s",
            type(exc).__name__, exc,
        )
    except (TypeError, AttributeError, OSError) as exc:
        logger.warning(
            "Optional pathway analysis failed with unexpected error [%s]: %s",
            type(exc).__name__, exc, exc_info=True,
        )
    return None


def preprocess_request(request: PredictionRequest) -> Dict[str, torch.Tensor]:
    """
    预处理预测请求

    参数:
        request: 预测请求

    返回:
        预处理后的批次字典。当 DAVF 启用时会附加 ``davf_sites`` 和
        ``davf_gene_names`` 列表字段，供 DAVF 模型分支使用。
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

    valid_ptm_sites = []
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
        valid_ptm_sites.append(ptm_site)

    out: Dict[str, Union[torch.Tensor, List[DAVFSiteDict], List[str]]] = {
        "sequence": seq_tensor,
        "ptm_mask": ptm_mask,
        "ptm_types": ptm_types,
    }

    # DAVF inputs: produced when the request opts in OR the loaded model has a
    # DAVF branch (otherwise the model would receive zero-features for a path
    # that needs real DAVF features).
    if _model_supports_davf(getattr(request, "use_davf", False)):
        davf_inputs = _collect_davf_inputs(valid_ptm_sites, max_len)
        if davf_inputs:
            out["davf_sites"] = [
                {"position": p, "ptm_type": t}
                for p, t in zip(
                    davf_inputs["davf_positions"],
                    davf_inputs["davf_ptm_types"], strict=False,
                )
            ]
            out["davf_gene_names"] = davf_inputs["davf_gene_names"]

    return out


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
        model = STATE.model
        assert model is not None  # guarded by callers
        outputs = model(batch)

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
        cell_state=pred_label,
        predicted_cell_state=pred_label,
        confidence=confidence,
        probabilities=prob_dict,
        pathway_impacts=pathway_impacts,
        model_kind=STATE.model_kind,
        is_demo_model=STATE.is_demo_model,
        processing_time_ms=round(timing_ms, 2),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
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
        # N01: 推理（预处理 + torch 前向 + pathway 分析）是阻塞 CPU/GPU 工作，
        # 通过 run_in_threadpool 卸载出事件循环，避免单 worker 串行阻塞
        # 健康检查与并发请求。
        def _infer() -> Tuple[torch.Tensor, torch.Tensor, Optional[List[PathwayImpact]]]:
            batch = preprocess_request(request)

            # Add batch dimension for single-sample inference. DAVF inputs
            # (``davf_sites`` / ``davf_gene_names``) are Python lists, not tensors,
            # so wrap them as single-element lists instead of calling ``.unsqueeze``.
            tensorized: Dict[str, Union[torch.Tensor, List[Any]]] = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    tensorized[key] = val.unsqueeze(0).to(STATE.device)
                elif key in ("davf_sites", "davf_gene_names"):
                    tensorized[key] = [val]  # one sample -> list of one
                else:
                    tensorized[key] = val

            probs_tensor, pred_tensor, _ = _run_prediction_on_batch(tensorized)
            pathway_impacts = _compute_pathway_impacts(request.ptm_sites, STATE.pathway_mapper)
            return probs_tensor, pred_tensor, pathway_impacts

        probs_tensor, pred_tensor, pathway_impacts = await run_in_threadpool(_infer)

        probs_np = probs_tensor[0].numpy()
        pred_idx = int(pred_tensor[0].item())

        processing_time_ms = (time.time() - start_time) * 1000

        return _build_prediction_response(probs_np, pred_idx, processing_time_ms, pathway_impacts)

    except HTTPException:
        raise
    except (ValueError, KeyError) as e:
        logger.error("预测请求错误 [%s]: %s", type(e).__name__, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数错误: {e}",
        ) from e
    except RuntimeError as e:
        logger.error("预测运行时错误 [%s]: %s", type(e).__name__, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="预测过程中发生运行时错误",
        ) from e
    except Exception as e:
        logger.error(
            "预测未知错误 [%s]: %s", type(e).__name__, e, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="预测过程中发生错误",
        ) from e


@router.post("/predict/batch", response_model=BatchPredictionResponse)
@router.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
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
        # N01: 批量推理（预处理 + torch 前向 + 逐样本 pathway 分析）卸载出事件循环。
        def _batch_infer() -> Tuple[torch.Tensor, torch.Tensor, List[Optional[List[PathwayImpact]]]]:
            batch = batch_preprocess(request.samples)
            # Move tensors to device; leave DAVF list inputs untouched.
            batch = {
                key: (val.to(STATE.device) if isinstance(val, torch.Tensor) else val)
                for key, val in batch.items()
            }
            probs_tensor, pred_tensor, _ = _run_prediction_on_batch(batch)
            impacts = [
                _compute_pathway_impacts(
                    request.samples[i].ptm_sites, STATE.pathway_mapper
                )
                for i in range(len(request.samples))
            ]
            return probs_tensor, pred_tensor, impacts

        probs_tensor, pred_tensor, impacts = await run_in_threadpool(_batch_infer)

        total_time_ms = (time.time() - start_time) * 1000
        per_sample_ms = total_time_ms / len(request.samples) if request.samples else 0

        predictions = [
            _build_prediction_response(
                probs_tensor[i].numpy(),
                int(pred_tensor[i].item()),
                per_sample_ms,
                impacts[i],
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
    except (ValueError, KeyError) as e:
        logger.error("批量预测请求错误 [%s]: %s", type(e).__name__, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"批量预测请求参数错误: {e}",
        ) from e
    except RuntimeError as e:
        logger.error("批量预测运行时错误 [%s]: %s", type(e).__name__, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="批量预测过程中发生运行时错误",
        ) from e
    except Exception as e:
        logger.error(
            "批量预测未知错误 [%s]: %s", type(e).__name__, e, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="批量预测过程中发生错误",
        ) from e


def _ptm_effects_to_models(ptm_effects: Dict[str, Dict[str, Any]]) -> List[PTMEffect]:
    """Convert workflow PTM effect dicts into ``PTMEffect`` schema objects."""
    return [
        PTMEffect(
            ptm_type=ptm_type,
            wildtype_prob=effect["wildtype_prob"],
            mutant_prob=effect["mutant_prob"],
            delta_prob=effect["delta_prob"],
            effect=effect["effect"],
        )
        for ptm_type, effect in ptm_effects.items()
    ]


def _pathway_impacts_to_models(
    pathway_impacts: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[List[PathwayImpact]]:
    """Convert workflow pathway impact dicts into ``PathwayImpact`` objects.

    Confidence is "high" when |activity| > 0.5, otherwise "medium" — the
    threshold contract locked by the variant route characterization tests.
    """
    if not pathway_impacts:
        return None
    return [
        PathwayImpact(
            pathway_name=name,
            activity_change=data.get("activity", 0.0),
            confidence="high" if abs(data.get("activity", 0)) > 0.5 else "medium",
            key_genes=data.get("genes", []),
        )
        for name, data in pathway_impacts.items()
    ]


def _variant_confidence(ptm_effects: List[PTMEffect]) -> float:
    """Confidence from the strongest PTM probability shift (capped at 1.0)."""
    max_delta = max((abs(e.delta_prob) for e in ptm_effects), default=0.0)
    return min(max_delta * 2, 1.0)


async def _resolve_variant_sequence(
    request: VariantPredictionRequest,
    warnings_list: List[str],
) -> Optional[str]:
    """Resolve the variant sequence from the request or UniProt.

    Uses the request sequence when present; otherwise fetches from UniProt via
    the variant workflow. Network failures map to 504, parse/unexpected errors
    to 400. Raises HTTPException when no sequence can be resolved.
    """
    sequence = request.sequence
    if sequence is None and request.uniprot_id:
        try:
            # N01: UniProt fetch 是阻塞网络 I/O，卸载出事件循环。
            sequence = await run_in_threadpool(
                STATE.variant_workflow.fetch_sequence_from_uniprot,
                request.uniprot_id,
            )
            logger.info(f"Fetched sequence from UniProt for {request.uniprot_id}")
        except (ConnectionError, TimeoutError) as e:
            logger.warning("UniProt fetch network error [%s]: %s", type(e).__name__, e)
            warnings_list.append(f"Failed to fetch sequence from UniProt: {e}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Could not reach UniProt for {request.uniprot_id}: {e}",
            ) from e
        except (ValueError, KeyError) as e:
            logger.warning("UniProt fetch parse error [%s]: %s", type(e).__name__, e)
            warnings_list.append(f"Failed to fetch sequence from UniProt: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not fetch sequence for {request.uniprot_id}: {e}",
            ) from e
        except (TypeError, AttributeError, OSError) as e:
            logger.warning(
                "UniProt fetch unexpected error [%s]: %s",
                type(e).__name__, e, exc_info=True,
            )
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
    return sequence


async def _variant_cell_state_prediction(
    sequence: Optional[str],
    ptm_types: List[str],
    warnings_list: List[str],
) -> Optional[str]:
    """Predict a qualitative cell-state label for the variant sequence.

    Runs the variant-aware sequence (plus PTM sites derived from the variant's
    PTM effects) through the initialized cell-state model. Degrades to a
    warning instead of an error when the model is unavailable or fails — the
    variant effect itself has already been computed at this point.
    """
    if STATE.model is None or not sequence:
        return None

    try:
        # Build PTM sites from the derived PTM effects so the model sees
        # the variant-induced modifications.
        def _sync() -> int:
            variant_ptm_sites = [{"position": 1, "type": ptm_type} for ptm_type in ptm_types]
            variant_request = PredictionRequest(
                sequence=sequence,
                ptm_sites=variant_ptm_sites,
                use_davf=False,
            )
            variant_batch = preprocess_request(variant_request)
            variant_batch = {
                key: val.unsqueeze(0).to(STATE.device) for key, val in variant_batch.items()
            }
            _, variant_pred_tensor, _ = _run_prediction_on_batch(variant_batch)
            return int(variant_pred_tensor[0].item())

        variant_pred_idx = await run_in_threadpool(_sync)
        return STATE.idx_to_label.get(variant_pred_idx, "unknown")
    except (ValueError, KeyError, RuntimeError) as exc:
        logger.warning(
            "Variant cell-state prediction failed [%s]: %s",
            type(exc).__name__, exc,
        )
        warnings_list.append(f"Cell-state prediction unavailable: {exc}")
    except (TypeError, AttributeError, OSError) as exc:
        logger.warning(
            "Variant cell-state prediction unexpected error [%s]: %s",
            type(exc).__name__, exc, exc_info=True,
        )
        warnings_list.append(f"Cell-state prediction unavailable: {exc}")
    return None


@router.post("/predict/variant", response_model=VariantPredictionResponse)
async def predict_variant(request: VariantPredictionRequest) -> VariantPredictionResponse:
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
            detail="Variant prediction workflow not initialized. Initialize via /initialize first.",
        )

    start_time = time.time()
    warnings_list = []

    try:
        sequence = await _resolve_variant_sequence(request, warnings_list)

        # N01: HGVS 解析 + 效应预测是 CPU 密集工作，卸载出事件循环。
        result = await run_in_threadpool(
            STATE.variant_workflow.predict_from_hgvs,
            hgvs_string=request.hgvs,
            sequence=sequence,
            include_pathways=request.include_pathways,
        )

        ptm_effects = _ptm_effects_to_models(result.ptm_effects)
        pathway_impacts = (
            _pathway_impacts_to_models(result.pathway_impacts)
            if request.include_pathways
            else None
        )
        confidence = _variant_confidence(ptm_effects)
        cell_state_prediction = await _variant_cell_state_prediction(
            sequence, list(result.ptm_effects.keys()), warnings_list
        )

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
            cell_state_prediction=cell_state_prediction,
            confidence=confidence,
            processing_time_ms=round(processing_time_ms, 2),
            warnings=warnings_list,
        )

    except HTTPException:
        raise
    except (ValueError, KeyError) as e:
        logger.error("Variant prediction request error [%s]: %s", type(e).__name__, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Variant prediction request error: {e}",
        ) from e
    except RuntimeError as e:
        logger.error("Variant prediction runtime error [%s]: %s", type(e).__name__, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Variant prediction runtime error",
        ) from e
    except Exception as e:
        logger.error(
            "Variant prediction unexpected error [%s]: %s",
            type(e).__name__, e, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Variant prediction failed: {str(e)}",
        ) from e


__all__ = ["router", "preprocess_request", "predict", "batch_predict", "predict_variant"]
