#!/usr/bin/env python3
"""Export a verified static gene-token matrix from a PerturbGen checkpoint.

The tensor key is mandatory: this tool never guesses a checkpoint layout.
It is intended to run inside the locked PerturbGen environment after Gate-0
has identified the real embedding parameter and confirmed vocabulary rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    return payload


def _resolve_explicit_key(payload: Mapping[str, Any], key: str) -> torch.Tensor:
    if key in payload:
        direct = payload[key]
        if not isinstance(direct, torch.Tensor):
            raise TypeError(f"checkpoint value at {key!r} is not a tensor")
        return direct
    current: Any = payload
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"explicit tensor key not found: {key}")
        current = current[part]
    if not isinstance(current, torch.Tensor):
        raise TypeError(f"checkpoint value at {key!r} is not a tensor")
    return current


def _load_vocabulary(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("vocabulary must be a non-empty JSON object")
    vocab = {str(gene): int(token) for gene, token in raw.items()}
    if any(token < 0 for token in vocab.values()) or len(set(vocab.values())) != len(vocab):
        raise ValueError("source vocabulary token IDs must be unique non-negative integers")
    return vocab


def export_asset(checkpoint: Path, tensor_key: str, vocabulary: Path, output_dir: Path) -> None:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    vocabulary = vocabulary.expanduser().resolve(strict=True)
    matrix = _resolve_explicit_key(_load_checkpoint(checkpoint), tensor_key).detach().cpu()
    vocab = _load_vocabulary(vocabulary)
    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError("selected embedding tensor must be 2-D and floating point")
    if max(vocab.values()) >= matrix.shape[0]:
        raise ValueError("source vocabulary contains a token row outside the embedding matrix")
    if not torch.isfinite(matrix).all():
        raise ValueError("embedding tensor contains non-finite values")

    # The upstream vocabulary may contain gaps/special-token rows.  Export only
    # declared genes in deterministic source-row order and compact the runtime
    # vocabulary to 0..N-1; never assume upstream row IDs are contiguous.
    ordered = sorted(vocab.items(), key=lambda item: (item[1], item[0]))
    source_rows = torch.tensor([row for _, row in ordered], dtype=torch.long)
    matrix = matrix.index_select(0, source_rows)
    compact_vocab = {gene: row for row, (gene, _) in enumerate(ordered)}

    output_dir.mkdir(parents=True, exist_ok=False)
    tensor_path = output_dir / "gene_embeddings.safetensors"
    vocab_path = output_dir / "vocabulary.json"
    manifest_path = output_dir / "manifest.json"
    save_file({"gene_embeddings": matrix.contiguous()}, str(tensor_path))
    vocab_path.write_text(
        json.dumps(compact_vocab, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source": {
            "checkpoint_sha256": _sha256(checkpoint),
            "vocabulary_sha256": _sha256(vocabulary),
            "tensor_key": tensor_key,
        },
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": str(matrix.dtype),
        "files": {
            "gene_embeddings.safetensors": {"sha256": _sha256(tensor_path), "size": tensor_path.stat().st_size},
            "vocabulary.json": {"sha256": _sha256(vocab_path), "size": vocab_path.stat().st_size},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tensor-key", required=True, help="Exact dot-separated checkpoint tensor key")
    parser.add_argument("--vocabulary", type=Path, required=True, help="JSON object mapping gene ID to row")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_asset(args.checkpoint, args.tensor_key, args.vocabulary, args.output_dir)


if __name__ == "__main__":
    main()
