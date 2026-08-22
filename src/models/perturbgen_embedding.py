"""Strict loader for PerturbGen gene-token embedding artifacts.

This module deliberately has no PerturbGen dependency.  The external
environment exports a frozen tensor plus vocabulary and manifest; the main
PTM2CellNet process only verifies and reads those immutable artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


class PerturbGenEmbeddingError(ValueError):
    """Raised when an embedding artifact is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PerturbGenEmbeddingAsset:
    """Verified, row-aligned PerturbGen embedding asset."""

    embeddings: torch.Tensor
    gene_to_token: Mapping[str, int]
    manifest: Mapping[str, object]

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.embeddings.shape[0])


def load_perturbgen_embedding_asset(
    asset_dir: str | Path,
) -> PerturbGenEmbeddingAsset:
    """Load an exported asset after validating hashes, schema and row IDs.

    Unknown genes are not synthesized or hashed.  Callers must use the exact
    vocabulary mapping returned here and treat missing genes as unevaluable.
    """

    root = Path(asset_dir).expanduser().resolve()
    if not root.is_dir():
        raise PerturbGenEmbeddingError(f"embedding asset directory not found: {root}")

    manifest_path = root / "manifest.json"
    vocab_path = root / "vocabulary.json"
    tensor_path = root / "gene_embeddings.safetensors"
    for path in (manifest_path, vocab_path, tensor_path):
        if not path.is_file():
            raise PerturbGenEmbeddingError(f"required embedding artifact missing: {path.name}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise PerturbGenEmbeddingError("unsupported embedding manifest schema_version")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise PerturbGenEmbeddingError("manifest.files must be an object")
    for name, path in (("vocabulary.json", vocab_path), ("gene_embeddings.safetensors", tensor_path)):
        record = files.get(name)
        if not isinstance(record, dict) or record.get("sha256") != _sha256(path):
            raise PerturbGenEmbeddingError(f"sha256 mismatch for {name}")

    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise PerturbGenEmbeddingError("vocabulary.json must be a non-empty object")
    try:
        gene_to_token = {str(gene): int(token) for gene, token in vocabulary.items()}
    except (TypeError, ValueError) as exc:
        raise PerturbGenEmbeddingError("vocabulary token IDs must be integers") from exc
    token_ids = sorted(gene_to_token.values())
    if token_ids != list(range(len(token_ids))):
        raise PerturbGenEmbeddingError("vocabulary token IDs must be unique and contiguous from 0")

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("safetensors is required to load PerturbGen embedding assets") from exc
    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != {"gene_embeddings"}:
        raise PerturbGenEmbeddingError("tensor artifact must contain only 'gene_embeddings'")
    embeddings = tensors["gene_embeddings"]
    if embeddings.ndim != 2 or not embeddings.is_floating_point():
        raise PerturbGenEmbeddingError("gene_embeddings must be a 2-D floating tensor")
    expected_shape = manifest.get("embedding_shape")
    if expected_shape != list(embeddings.shape):
        raise PerturbGenEmbeddingError("manifest embedding_shape does not match tensor")
    if embeddings.shape[0] != len(gene_to_token):
        raise PerturbGenEmbeddingError("vocabulary size does not match embedding rows")
    if not torch.isfinite(embeddings).all():
        raise PerturbGenEmbeddingError("gene_embeddings contains non-finite values")

    return PerturbGenEmbeddingAsset(
        embeddings=embeddings,
        gene_to_token=gene_to_token,
        manifest=manifest,
    )
