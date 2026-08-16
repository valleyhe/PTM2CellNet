#!/usr/bin/env python3
"""Precompute pLM sequence embeddings for cross-scale training (TD-H02).

Consumes raw sequence records (``sample_id`` + amino-acid sequence) and
produces the per-backbone embedding arrays that the cross-scale NPZ
assembler consumes (``src/data/cross_scale_dataset.py``,
``ptm2cellnet.cross-scale.npz.v1``)::

    {output_dir}/embeddings.npz      # {backbone}_embeddings [N,L,D] + masks + sample_id
    {output_dir}/manifest.json       # provenance (model dirs, input sha256, params, failures)
    {output_dir}/failed_samples.jsonl # per-sample failure ledger (resume-able)

Input formats
-------------
* TSV (default): one record per line, ``sample_id<TAB>sequence``,
  blank lines and lines starting with ``#`` are skipped.
* ``--genes-json``: a ``sequence_requests.json``-style record
  (``{"dataset": ..., "genes": [...]}``) recorded into the manifest as
  provenance; actual sequences must still come from the TSV.

Backbones
---------
* ``esm2``    — local HuggingFace ESM-2 checkpoint dir (``esm2*/`` under
  ``--model-dir``, or explicit ``--esm2-dir``).
* ``prott5``  — local ProtT5 encoder checkpoint dir (``prot_t5*/``).
* ``ankh39``  — local Ankh checkpoint dir (``ankh*/``; requires the
  ``ankh`` package at runtime, fails loudly when absent).
* ``mock``    — deterministic fake encoder (testing / CI only).

Design notes
------------
* Deterministic, resumable: ``--resume`` reuses an existing manifest with
  the same input sha256 and skips already-processed sample ids.
* Fail-soft per sample: an unprocessable record (empty sequence, tokenizer
  failure) is written to the failure ledger and the run continues.
* All models run under ``torch.no_grad()``; outputs are CPU float32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EMBEDDING_CACHE_SCHEMA_VERSION = "ptm2cellnet.embedding-cache.v1"
DEFAULT_BACKBONES = ("esm2", "prott5")
DEFAULT_MODEL_DIR = "data/weights/plm"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 8

# 与 src/data/cross_scale_dataset.py 的 backbone 命名约定一致。
BACKBONE_DIR_PREFIXES = {
    "esm2": "esm2",
    "prott5": "prot_t5",
    "ankh39": "ankh",
}


class EmbeddingPrecomputeError(ValueError):
    """Raised for invalid CLI inputs / unsupported backbone configs."""


@dataclass
class BackboneSpec:
    name: str
    model_dir: Optional[str]
    embedding_dim: int
    provenance: str
    encode: Callable[[Sequence[str]], Tuple[np.ndarray, np.ndarray]]


@dataclass
class PrecomputeResult:
    output_dir: Path
    sample_ids: List[str]
    n_processed: int
    n_failed: int
    backbone_dims: Dict[str, int]
    sequence_length: int
    failures: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def read_sequence_records(tsv_path: Path) -> List[Tuple[str, str]]:
    """Parse ``sample_id<TAB>sequence`` records.

    Blank lines and ``#``-prefixed lines are skipped; malformed lines
    (not exactly two columns) raise with the offending line number.
    """
    records: List[Tuple[str, str]] = []
    seen_ids: set[str] = set()
    if not tsv_path.is_file():
        raise EmbeddingPrecomputeError(f"输入文件不存在: {tsv_path}")
    for lineno, raw in enumerate(tsv_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip("\r")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise EmbeddingPrecomputeError(
                f"{tsv_path}:{lineno} 行格式错误（需要 sample_id<TAB>sequence）"
            )
        sample_id, sequence = parts[0].strip(), parts[1].strip()
        if not sample_id:
            raise EmbeddingPrecomputeError(f"{tsv_path}:{lineno} sample_id 为空")
        if sample_id in seen_ids:
            raise EmbeddingPrecomputeError(f"{tsv_path}:{lineno} sample_id 重复: {sample_id}")
        seen_ids.add(sample_id)
        records.append((sample_id, sequence))
    if not records:
        raise EmbeddingPrecomputeError(f"{tsv_path} 无有效记录")
    return records


def _fasta_aliases(header: str) -> List[str]:
    """Return stable lookup aliases for a UniProt-style FASTA header."""
    aliases: List[str] = []
    first_token = header.split(maxsplit=1)[0] if header else ""
    aliases.extend(part for part in first_token.split("|") if part)
    for token in header.split():
        if token.startswith("GN="):
            aliases.extend(gene for gene in token[3:].split(",") if gene)
    return aliases


def read_fasta_sequences(fasta_path: Path) -> Dict[str, str]:
    """Load sequences and index common accession/gene aliases.

    The first record wins when a source contains several isoforms for the same
    alias.  This makes the selection deterministic and leaves the chosen
    source/header in the manifest for auditability.
    """
    if not fasta_path.is_file():
        raise EmbeddingPrecomputeError(f"序列 FASTA 不存在: {fasta_path}")
    sequences: Dict[str, str] = {}
    header = ""
    chunks: List[str] = []

    def flush() -> None:
        if not header:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            return
        for alias in _fasta_aliases(header):
            sequences.setdefault(alias, sequence)

    for raw in fasta_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:]
            chunks = []
        else:
            chunks.append(line)
    flush()
    if not sequences:
        raise EmbeddingPrecomputeError(f"序列 FASTA 无有效记录: {fasta_path}")
    return sequences


def _json_sequence_records(payload: Any, source: Path) -> Optional[List[Tuple[str, str]]]:
    """Extract direct ``[{id, sequence}]`` records from a JSON payload."""
    records_value: Any = payload.get("records") if isinstance(payload, Mapping) else payload
    if isinstance(payload, Mapping) and isinstance(payload.get("sequences"), Mapping):
        records_value = [
            {"sample_id": sample_id, "sequence": sequence}
            for sample_id, sequence in payload["sequences"].items()
        ]
    if not isinstance(records_value, list):
        return None
    records: List[Tuple[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(records_value, start=1):
        if not isinstance(item, Mapping):
            raise EmbeddingPrecomputeError(f"{source}: records[{index}] 必须是对象")
        sample_id = item.get("sample_id", item.get("id", item.get("gene")))
        sequence = item.get("sequence")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise EmbeddingPrecomputeError(f"{source}: records[{index}] 缺少 sample_id")
        if not isinstance(sequence, str):
            raise EmbeddingPrecomputeError(f"{source}: records[{index}] 缺少 sequence")
        sample_id = sample_id.strip()
        if sample_id in seen_ids:
            raise EmbeddingPrecomputeError(f"{source}: sample_id 重复: {sample_id}")
        seen_ids.add(sample_id)
        records.append((sample_id, sequence.strip()))
    return records


def read_gene_request_records(
    genes_json: Path,
    *,
    sequence_records: Optional[Sequence[Tuple[str, str]]] = None,
    sequence_fasta: Optional[Path] = None,
) -> List[Tuple[str, str]]:
    """Resolve a ``sequence_requests.json`` file into embedding records.

    A request file may contain direct records/sequences.  The Norman/Adamson
    form contains only gene symbols, so it must be paired with either the
    ``--input`` TSV or ``--sequence-fasta``.  Missing genes are emitted with an
    empty sequence and therefore appear in the normal fail-soft ledger.
    """
    if not genes_json.is_file():
        raise EmbeddingPrecomputeError(f"基因请求 JSON 不存在: {genes_json}")
    try:
        payload = json.loads(genes_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingPrecomputeError(f"无法读取基因请求 JSON: {genes_json}: {exc}") from exc

    direct_records = _json_sequence_records(payload, genes_json)
    if direct_records is not None:
        return direct_records
    if not isinstance(payload, Mapping) or not isinstance(payload.get("genes"), list):
        raise EmbeddingPrecomputeError(f"{genes_json} 必须包含 genes 或 records 字段")
    genes = [str(gene).strip() for gene in payload["genes"] if str(gene).strip()]
    if not genes:
        raise EmbeddingPrecomputeError(f"{genes_json} 的 genes 为空")

    source_map: Dict[str, str] = {}
    if sequence_records is not None:
        source_map.update({sample_id: sequence for sample_id, sequence in sequence_records})
    if sequence_fasta is not None:
        source_map.update(read_fasta_sequences(sequence_fasta))
    if not source_map:
        raise EmbeddingPrecomputeError(
            "sequence_requests.json 只包含基因名；请同时提供 --input 序列 TSV 或 --sequence-fasta"
        )
    return [(gene, source_map.get(gene, "")) for gene in genes]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def _find_model_dir(model_dir: Path, prefix: str, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return str(Path(explicit).resolve())
    matches = sorted(p for p in model_dir.glob(f"{prefix}*") if p.is_dir())
    return str(matches[0].resolve()) if matches else None


def _load_hf_backbone(
    name: str,
    model_path: str,
    device: str,
    max_length: int,
    model_class: str,
) -> Tuple[Callable[[Sequence[str]], Tuple[np.ndarray, np.ndarray]], int, str]:
    """Load a HuggingFace encoder checkpoint and return (encode_fn, dim, provenance).

    ``model_class`` is either ``"auto"`` (AutoModel + AutoTokenizer, used by
    ESM-2/Ankh) or ``"t5-encoder"`` (T5EncoderModel, used by ProtT5).
    """
    try:
        from transformers import AutoModel, AutoTokenizer, T5EncoderModel  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env guard
        raise EmbeddingPrecomputeError(
            "需要 transformers 才能加载 pLM backbone；"
            "请安装 requirements-pretrained.txt"
        ) from exc

    if model_class == "t5-encoder":
        model = T5EncoderModel.from_pretrained(model_path)
    else:
        model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    model.to(device)
    dim = int(model.config.hidden_size)

    def encode(sequences: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        tokenized = tokenizer(
            list(sequences),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch_no_grad():
            inputs = {k: v.to(device) for k, v in tokenized.items() if k in ("input_ids", "attention_mask")}
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state.detach().cpu().float().numpy()
        mask = inputs["attention_mask"].detach().cpu().numpy().astype(np.float32)
        return hidden, mask

    return encode, dim, str(model_path)


def torch_no_grad():
    import torch  # noqa: PLC0415

    return torch.no_grad()


def _make_mock_backbone(name: str, dim: int = 8, seed: int = 42) -> BackboneSpec:
    """Deterministic fake encoder for tests/CI: embeddings derive from the
    sequence content only (positional hash), so identical input sequences
    always yield identical arrays."""

    def encode(sequences: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        max_len = max((len(s) for s in sequences), default=0) or 1
        embeddings = np.zeros((len(sequences), max_len, dim), dtype=np.float32)
        masks = np.zeros((len(sequences), max_len), dtype=np.float32)
        rng = np.random.default_rng(seed)
        for i, seq in enumerate(sequences):
            masks[i, : len(seq)] = 1.0
            for pos in range(len(seq)):
                rng = np.random.default_rng(seed + sum(ord(c) for c in seq[: pos + 1]))
                embeddings[i, pos] = rng.standard_normal(dim).astype(np.float32)
        return embeddings, masks

    return BackboneSpec(
        name=name,
        model_dir=None,
        embedding_dim=dim,
        provenance="mock",
        encode=encode,
    )


def load_backbone(
    name: str,
    model_dir: Path,
    device: str,
    max_length: int,
    explicit_dirs: Dict[str, Optional[str]],
) -> BackboneSpec:
    name = name.lower()
    if name == "mock":
        return _make_mock_backbone(name)
    prefix = BACKBONE_DIR_PREFIXES.get(name)
    if prefix is None:
        raise EmbeddingPrecomputeError(
            f"不支持的 backbone: {name!r}（可选: esm2 / prott5 / ankh39 / mock）"
        )
    model_path = _find_model_dir(model_dir, prefix, explicit_dirs.get(name))
    if model_path is None:
        raise EmbeddingPrecomputeError(
            f"backbone {name!r}: 在 {model_dir} 下未找到 {prefix}* 模型目录"
        )
    model_class = "t5-encoder" if name == "prott5" else "auto"
    encode, dim, provenance = _load_hf_backbone(name, model_path, device, max_length, model_class)
    return BackboneSpec(name=name, model_dir=model_path, embedding_dim=dim, provenance=provenance, encode=encode)


# ---------------------------------------------------------------------------
# Encoding + persistence
# ---------------------------------------------------------------------------

def _pad_to(arrays: List[np.ndarray], length: int) -> np.ndarray:
    """Pad a list of [Li, D] arrays to [B, length, D] with zeros."""
    out = np.zeros((len(arrays), length, arrays[0].shape[1]), dtype=np.float32)
    masks = np.zeros((len(arrays), length), dtype=np.float32)
    for i, array in enumerate(arrays):
        Li = array.shape[0]
        out[i, :Li] = array
        masks[i, :Li] = 1.0
    return out, masks


def _trim_embedding_pair(
    embedding: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove only tokenizer padding while retaining special tokens."""
    embedding = np.asarray(embedding, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if embedding.ndim != 2 or mask.ndim != 1 or embedding.shape[0] != mask.shape[0]:
        raise EmbeddingPrecomputeError(
            f"backbone 输出 shape 无效: embedding={embedding.shape}, mask={mask.shape}"
        )
    active = np.flatnonzero(mask > 0)
    end = int(active[-1]) + 1 if active.size else 0
    return embedding[:end].copy(), mask[:end].copy()


def _encode_batch(
    spec: BackboneSpec,
    sequences: Sequence[str],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Encode a batch and validate its leading dimension and masks."""
    embeddings, masks = spec.encode(sequences)
    embeddings = np.asarray(embeddings)
    masks = np.asarray(masks)
    if embeddings.ndim != 3 or embeddings.shape[0] != len(sequences):
        raise EmbeddingPrecomputeError(
            f"{spec.name} embedding 输出必须为 [B,L,D]，实际 {embeddings.shape}"
        )
    if embeddings.shape[2] != spec.embedding_dim:
        raise EmbeddingPrecomputeError(
            f"{spec.name} embedding 维度不匹配: 期望 {spec.embedding_dim}，实际 {embeddings.shape[2]}"
        )
    if masks.ndim != 2 or masks.shape[:2] != embeddings.shape[:2]:
        raise EmbeddingPrecomputeError(
            f"{spec.name} attention mask 必须为 [B,L]，实际 {masks.shape}"
        )
    return [
        _trim_embedding_pair(embeddings[index], masks[index])
        for index in range(len(sequences))
    ]


def _load_resume_cache(
    output_dir: Path,
    manifest: Mapping[str, Any],
    records_sha: str,
    backbones: Sequence[BackboneSpec],
) -> Optional[Tuple[Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], List[str], int]]:
    """Load a valid prior cache keyed by sample id, or return ``None``."""
    if manifest.get("input", {}).get("sha256") != records_sha:
        return None
    processed_ids = [str(sample_id) for sample_id in manifest.get("processed_sample_ids", [])]
    npz_path = output_dir / "embeddings.npz"
    if not processed_ids or not npz_path.is_file():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            sample_ids = [str(value) for value in archive["sample_id"].tolist()]
            sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
            if len(sample_index) != len(sample_ids) or not set(processed_ids) <= set(sample_index):
                return None
            cache: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
            sequence_length = 0
            for sample_id in processed_ids:
                index = sample_index[sample_id]
                cache[sample_id] = {}
                for spec in backbones:
                    embedding_key = f"{spec.name}_embeddings"
                    mask_key = f"{spec.name}_attention_mask"
                    if embedding_key not in archive or mask_key not in archive:
                        return None
                    embedding = np.array(archive[embedding_key][index], copy=True)
                    mask = np.array(archive[mask_key][index], copy=True)
                    cache[sample_id][spec.name] = _trim_embedding_pair(embedding, mask)
                    sequence_length = max(sequence_length, embedding.shape[0])
            return cache, processed_ids, sequence_length
    except (KeyError, OSError, ValueError, EmbeddingPrecomputeError):
        return None


def precompute(
    records: Sequence[Tuple[str, str]],
    backbones: Sequence[BackboneSpec],
    output_dir: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    resume: bool = False,
    device: str = "cpu",
) -> PrecomputeResult:
    if not backbones:
        raise EmbeddingPrecomputeError("至少需要一个 backbone")
    if batch_size <= 0:
        raise EmbeddingPrecomputeError("batch_size 必须大于 0")
    if max_length < 0:
        raise EmbeddingPrecomputeError("max_length 不能小于 0")
    sample_ids_in_input = [sample_id for sample_id, _ in records]
    if len(set(sample_ids_in_input)) != len(sample_ids_in_input):
        raise EmbeddingPrecomputeError("输入中存在重复 sample_id")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    failures_path = output_dir / "failed_samples.jsonl"
    payload = "\n".join(f"{sample_id}\t{sequence}" for sample_id, sequence in records)
    records_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    cache: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    prior_failures: Dict[str, Dict[str, Any]] = {}
    prior_sequence_length = 0
    prior_manifest: Optional[Dict[str, Any]] = None
    if resume and manifest_path.is_file():
        try:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            candidate = None
        if isinstance(candidate, dict):
            loaded = _load_resume_cache(output_dir, candidate, records_sha, backbones)
            if loaded is not None:
                cache, _, prior_sequence_length = loaded
                prior_manifest = candidate
                prior_failures = {
                    str(failure.get("sample_id")): failure
                    for failure in candidate.get("failed_samples", [])
                    if isinstance(failure, Mapping) and failure.get("sample_id")
                }

    covered_ids = set(cache) | set(prior_failures)
    if resume and prior_manifest is not None and all(sample_id in covered_ids for sample_id in sample_ids_in_input):
        return PrecomputeResult(
            output_dir=output_dir,
            sample_ids=[sample_id for sample_id in sample_ids_in_input if sample_id in cache],
            n_processed=len(cache),
            n_failed=sum(sample_id in prior_failures for sample_id in sample_ids_in_input),
            backbone_dims={spec.name: spec.embedding_dim for spec in backbones},
            sequence_length=prior_sequence_length,
            failures=[prior_failures[sample_id] for sample_id in sample_ids_in_input if sample_id in prior_failures],
        )

    failures: Dict[str, Dict[str, Any]] = {
        sample_id: failure for sample_id, failure in prior_failures.items() if sample_id in set(sample_ids_in_input)
    }
    candidates: List[Tuple[str, str]] = [
        (sample_id, sequence)
        for sample_id, sequence in records
        if sample_id not in cache and sample_id not in failures
    ]

    def encode_one(sample_id: str, sequence: str) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        try:
            return {
                spec.name: _encode_batch(spec, [sequence])[0]
                for spec in backbones
            }
        except Exception as exc:  # noqa: BLE001 - per-sample failure ledger
            failures[sample_id] = {"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"}
            return None

    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        valid: List[Tuple[str, str]] = []
        for sample_id, sequence in chunk:
            if not sequence:
                failures[sample_id] = {"sample_id": sample_id, "error": "empty sequence"}
                continue
            if max_length > 0:
                sequence = sequence[:max_length]
            valid.append((sample_id, sequence))
        if not valid:
            continue

        try:
            encoded_by_backbone = {
                spec.name: _encode_batch(spec, [sequence for _, sequence in valid])
                for spec in backbones
            }
        except Exception:
            # A tokenizer/model can reject one sequence and abort the whole
            # batch. Identify that record without losing the other samples.
            for sample_id, sequence in valid:
                sample_arrays = encode_one(sample_id, sequence)
                if sample_arrays is not None:
                    cache[sample_id] = sample_arrays
            continue
        for index, (sample_id, _) in enumerate(valid):
            cache[sample_id] = {
                spec.name: encoded_by_backbone[spec.name][index]
                for spec in backbones
            }

    success_ids = [sample_id for sample_id in sample_ids_in_input if sample_id in cache]
    ordered_failures = [
        failures[sample_id] for sample_id in sample_ids_in_input if sample_id in failures
    ]
    if not success_ids:
        _write_manifest(
            manifest_path,
            {
                "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
                "input": {"format": "records", "sha256": records_sha, "n_samples": len(records)},
                "backbones": {
                    spec.name: {"embedding_dim": spec.embedding_dim, "model_dir": spec.model_dir, "provenance": spec.provenance}
                    for spec in backbones
                },
                "params": {"batch_size": batch_size, "max_length": max_length, "device": device},
                "processed_sample_ids": [],
                "failed_samples": ordered_failures,
                "n_processed": 0,
                "n_failed": len(ordered_failures),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        _write_failures(failures_path, ordered_failures)
        return PrecomputeResult(
            output_dir=output_dir,
            sample_ids=[],
            n_processed=0,
            n_failed=len(ordered_failures),
            backbone_dims={spec.name: spec.embedding_dim for spec in backbones},
            sequence_length=0,
            failures=ordered_failures,
        )

    final_length = max(
        pair[0].shape[0]
        for sample_arrays in cache.values()
        for pair in sample_arrays.values()
    )
    npz_arrays: Dict[str, np.ndarray] = {
        "schema_version": np.array(EMBEDDING_CACHE_SCHEMA_VERSION),
        "sample_id": np.array(success_ids),
    }
    for spec in backbones:
        embeddings: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for sample_id in success_ids:
            embedding, mask = cache[sample_id][spec.name]
            embeddings.append(embedding)
            masks.append(mask)
        padded_embeddings = np.zeros((len(success_ids), final_length, spec.embedding_dim), dtype=np.float32)
        padded_masks = np.zeros((len(success_ids), final_length), dtype=np.float32)
        for index, (embedding, mask) in enumerate(zip(embeddings, masks, strict=True)):
            length = min(final_length, embedding.shape[0])
            padded_embeddings[index, :length, : embedding.shape[1]] = embedding[:length]
            padded_masks[index, :length] = mask[:length]
        npz_arrays[f"{spec.name}_embeddings"] = padded_embeddings
        npz_arrays[f"{spec.name}_attention_mask"] = padded_masks

    npz_path = output_dir / "embeddings.npz"
    np.savez_compressed(npz_path, **npz_arrays)
    _write_manifest(
        manifest_path,
        {
            "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
            "input": {"format": "records", "sha256": records_sha, "n_samples": len(records)},
            "backbones": {
                spec.name: {
                    "embedding_dim": spec.embedding_dim,
                    "model_dir": spec.model_dir,
                    "provenance": spec.provenance,
                }
                for spec in backbones
            },
            "params": {"batch_size": batch_size, "max_length": max_length, "device": device},
            "outputs": {"embeddings_npz": str(npz_path), "sha256": _sha256_file(npz_path)},
            "processed_sample_ids": success_ids,
            "failed_samples": ordered_failures,
            "n_processed": len(success_ids),
            "n_failed": len(ordered_failures),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    _write_failures(failures_path, ordered_failures)
    return PrecomputeResult(
        output_dir=output_dir,
        sample_ids=success_ids,
        n_processed=len(success_ids),
        n_failed=len(ordered_failures),
        backbone_dims={spec.name: spec.embedding_dim for spec in backbones},
        sequence_length=final_length,
        failures=ordered_failures,
    )


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_failures(path: Path, failures: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TD-H02: 预计算 pLM 序列 embedding（cross-scale NPZ 组装前置步骤）"
    )
    parser.add_argument("--input", default=None, help="序列记录 TSV（sample_id<TAB>sequence）")
    parser.add_argument(
        "--genes-json",
        default=None,
        help="sequence_requests.json；可与 --input 或 --sequence-fasta 配合使用",
    )
    parser.add_argument(
        "--sequence-fasta",
        default=None,
        help="为 --genes-json 中的 gene symbol 提供序列的 FASTA 文件",
    )
    parser.add_argument("--output", required=True, help="输出目录（embeddings.npz + manifest.json）")
    parser.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES), help="逗号分隔 backbone 列表")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="本地 pLM 权重根目录")
    parser.add_argument("--esm2-dir", default=None, help="显式 ESM-2 checkpoint 目录（覆盖自动发现）")
    parser.add_argument("--prott5-dir", default=None, help="显式 ProtT5 checkpoint 目录（覆盖自动发现）")
    parser.add_argument("--ankh-dir", default=None, help="显式 Ankh checkpoint 目录（覆盖自动发现）")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="token/残基最大长度（0=不限）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--resume", action="store_true", help="复用已有 manifest，跳过已处理样本")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.input is None and args.genes_json is None:
            raise EmbeddingPrecomputeError("必须提供 --input 或 --genes-json")
        sequence_records = read_sequence_records(Path(args.input)) if args.input else None
        if args.genes_json:
            records = read_gene_request_records(
                Path(args.genes_json),
                sequence_records=sequence_records,
                sequence_fasta=Path(args.sequence_fasta) if args.sequence_fasta else None,
            )
        else:
            records = sequence_records or []
        backbone_names = [name.strip() for name in args.backbones.split(",") if name.strip()]
        if not backbone_names:
            raise EmbeddingPrecomputeError("--backbones 为空")
        if args.batch_size <= 0:
            raise EmbeddingPrecomputeError("--batch-size 必须大于 0")
        if args.max_length < 0:
            raise EmbeddingPrecomputeError("--max-length 不能小于 0")
        explicit = {
            "esm2": args.esm2_dir,
            "prott5": args.prott5_dir,
            "ankh39": args.ankh_dir,
        }
        backbones = [
            load_backbone(name, Path(args.model_dir), args.device, args.max_length, explicit)
            for name in backbone_names
        ]
        result = precompute(
            records,
            backbones,
            Path(args.output),
            batch_size=args.batch_size,
            max_length=args.max_length,
            resume=args.resume,
            device=args.device,
        )
    except EmbeddingPrecomputeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("中断", file=sys.stderr)
        return 130

    print(
        f"完成: {result.n_processed} 样本 / {result.n_failed} 失败 / "
        f"L={result.sequence_length} / backbones={sorted(result.backbone_dims)}"
    )
    if result.failures:
        print(f"失败样本已记录: {result.output_dir / 'failed_samples.jsonl'}", file=sys.stderr)
    return 0 if result.n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
