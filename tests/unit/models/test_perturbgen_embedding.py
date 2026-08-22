import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from src.models.perturbgen_embedding import (
    PerturbGenEmbeddingError,
    load_perturbgen_embedding_asset,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(tmp_path):
    tensor_path = tmp_path / "gene_embeddings.safetensors"
    vocab_path = tmp_path / "vocabulary.json"
    save_file({"gene_embeddings": torch.arange(6, dtype=torch.float32).reshape(2, 3)}, str(tensor_path))
    vocab_path.write_text(json.dumps({"ENSG000001": 0, "ENSG000002": 1}), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "embedding_shape": [2, 3],
        "files": {
            tensor_path.name: {"sha256": _sha(tensor_path)},
            vocab_path.name: {"sha256": _sha(vocab_path)},
        },
    }), encoding="utf-8")
    return tensor_path, vocab_path


def test_load_verified_embedding_asset(tmp_path):
    _asset(tmp_path)
    asset = load_perturbgen_embedding_asset(tmp_path)
    assert asset.embedding_dim == 3
    assert asset.vocab_size == 2
    assert asset.gene_to_token["ENSG000002"] == 1


def test_rejects_tampered_vocabulary(tmp_path):
    _, vocab_path = _asset(tmp_path)
    vocab_path.write_text(json.dumps({"ENSG000001": 1}), encoding="utf-8")
    with pytest.raises(PerturbGenEmbeddingError, match="sha256 mismatch"):
        load_perturbgen_embedding_asset(tmp_path)


def test_rejects_non_contiguous_rows_even_with_valid_hash(tmp_path):
    tensor_path, vocab_path = _asset(tmp_path)
    vocab_path.write_text(json.dumps({"ENSG000001": 0, "ENSG000002": 2}), encoding="utf-8")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"][vocab_path.name]["sha256"] = _sha(vocab_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PerturbGenEmbeddingError, match="contiguous"):
        load_perturbgen_embedding_asset(tmp_path)
