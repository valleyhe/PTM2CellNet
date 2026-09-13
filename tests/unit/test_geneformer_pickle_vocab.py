"""U-07 / TD-13-11: Geneformer pickle vocab and strict hash fallback."""

from __future__ import annotations

import hashlib
import pickle

import pytest
import torch

from src.models.geneformer_embedding import GeneformerEmbeddingLoader, GeneformerVocabularyError


def test_pickle_vocab_refused_without_pin_or_opt_in(tmp_path, monkeypatch) -> None:
    path = tmp_path / "token_dictionary.pkl"
    path.write_bytes(pickle.dumps({"ENSG00000168610": 0}))
    monkeypatch.delenv("PTM2CELLNET_GENEFORMER_VOCAB_SHA256", raising=False)
    monkeypatch.delenv("PTM2CELLNET_ALLOW_GENEFORMER_PICKLE", raising=False)
    monkeypatch.setenv("PTM2CELLNET_ENV", "production")
    loader = GeneformerEmbeddingLoader.__new__(GeneformerEmbeddingLoader)
    loader.strict = False
    with pytest.raises(GeneformerVocabularyError, match="refusing pickle.load"):
        loader._load_official_pickle_vocab(path)


def test_pickle_vocab_refused_in_strict_mode(tmp_path, monkeypatch) -> None:
    path = tmp_path / "token_dictionary.pkl"
    path.write_bytes(pickle.dumps({"ENSG00000168610": 0}))
    monkeypatch.delenv("PTM2CELLNET_GENEFORMER_VOCAB_SHA256", raising=False)
    monkeypatch.delenv("PTM2CELLNET_ALLOW_GENEFORMER_PICKLE", raising=False)
    monkeypatch.delenv("PTM2CELLNET_ENV", raising=False)
    loader = GeneformerEmbeddingLoader.__new__(GeneformerEmbeddingLoader)
    loader.strict = True
    with pytest.raises(GeneformerVocabularyError, match="refusing pickle.load"):
        loader._load_official_pickle_vocab(path)


def test_pickle_vocab_accepted_with_sha_pin(tmp_path, monkeypatch) -> None:
    path = tmp_path / "token_dictionary.pkl"
    payload = {"ENSG00000168610": 0, "ENSG00000177606": 1}
    raw = pickle.dumps(payload)
    path.write_bytes(raw)
    monkeypatch.setenv("PTM2CELLNET_GENEFORMER_VOCAB_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.delenv("PTM2CELLNET_ALLOW_GENEFORMER_PICKLE", raising=False)
    loader = GeneformerEmbeddingLoader.__new__(GeneformerEmbeddingLoader)
    loaded = loader._load_official_pickle_vocab(path)
    assert loaded == payload


def test_strict_mode_refuses_hash_fallback() -> None:
    loader = GeneformerEmbeddingLoader.__new__(GeneformerEmbeddingLoader)
    loader.strict = True
    loader.device = torch.device("cpu")
    loader._embeddings = torch.zeros(4, 2)
    loader._gene_to_idx = {"KNOWN": 0}
    loader._vocabulary_is_semantic = False
    loader._vocab_size = 4
    with pytest.raises(KeyError, match="cannot be hashed"):
        loader.get_gene_embedding(["UNKNOWN"])
