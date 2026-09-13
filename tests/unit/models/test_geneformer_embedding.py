"""GeneformerEmbeddingLoader 契约测试（N06 余量：geneformer_embedding.py 原零专属测试）。

覆盖 src/models/geneformer_embedding.py 的公开契约：fallback 语义（显式
警告、随机嵌入形状/词表）、strict 门禁（构造参数与
PTM2CELLNET_STRICT_MODEL_ASSETS 环境覆盖）、稳定哈希基因映射、
embedding 查找（字符串/整数/越界）、设备迁移与模块级单例助手。

不触网：model_path 一律指向不存在的本地绝对路径（transformers 对绝对
路径不会回退到 hub），并强制 HF_HUB_OFFLINE=1 兜底。
"""

import warnings

import pytest
import torch

import src.models.geneformer_embedding as gfe
from src.models.geneformer_embedding import GeneformerEmbeddingLoader


@pytest.fixture(autouse=True)
def _preserve_module_globals(monkeypatch):
    """离线兜底 + 保存/恢复模块级单例与 fallback 标志，避免状态泄漏。"""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    loader = gfe._geneformer_loader
    flag = gfe._geneformer_is_fallback
    yield
    gfe._geneformer_loader = loader
    gfe._geneformer_is_fallback = flag


def _make_fallback_loader(tmp_path) -> GeneformerEmbeddingLoader:
    """构造 fallback loader（不触网，吞掉已知 UserWarning）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return GeneformerEmbeddingLoader(
            model_path=str(tmp_path / "absent_geneformer"),
            device=torch.device("cpu"),
        )


class TestFallbackContract:
    def test_missing_model_falls_back_with_explicit_warning(self, tmp_path):
        with pytest.warns(UserWarning, match="NOT suitable for production"):
            loader = GeneformerEmbeddingLoader(
                model_path=str(tmp_path / "absent_geneformer"),
                device=torch.device("cpu"),
            )
        assert loader.model_source == "fallback_random"
        assert loader.model_provenance == "fallback_random"
        assert loader.get_vocab_size() == 30000
        assert loader.embedding_dim == 1152
        assert loader.embeddings.shape == (30000, 1152)

    def test_strict_constructor_raises_fail_fast(self, tmp_path):
        with pytest.raises(RuntimeError, match="PTM2CELLNET_STRICT_MODEL_ASSETS"):
            GeneformerEmbeddingLoader(
                model_path=str(tmp_path / "absent_geneformer"),
                device=torch.device("cpu"),
                strict=True,
            )

    def test_env_var_forces_strict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTM2CELLNET_STRICT_MODEL_ASSETS", "1")
        with pytest.raises(RuntimeError, match="failed to load"):
            GeneformerEmbeddingLoader(
                model_path=str(tmp_path / "absent_geneformer"),
                device=torch.device("cpu"),
                strict=False,
            )


class TestGeneEmbeddingLookup:
    def test_string_lookup_is_deterministic_and_cached(self, tmp_path):
        loader = _make_fallback_loader(tmp_path)
        e1 = loader.get_gene_embedding(["TP53"])
        e2 = loader.get_gene_embedding(["TP53"])
        torch.testing.assert_close(e1, e2)
        assert e1.shape == (1, loader.embedding_dim)
        # TD-NEW-02: the fallback hash is never written back into the
        # vocabulary, so repeated lookups cannot pollute later resolution.
        assert "TP53" not in loader._gene_to_idx

    def test_fallback_lookup_does_not_pollute_vocabulary(self, tmp_path):
        loader = _make_fallback_loader(tmp_path)
        before = dict(loader._gene_to_idx)
        loader.get_gene_embedding(["BRCA1", "EGFR"])
        assert loader._gene_to_idx == before

    def test_stable_hash_is_vocab_bounded_and_reproducible(self, tmp_path):
        loader = _make_fallback_loader(tmp_path)
        i1 = loader._stable_gene_index("ENSG00000139618")
        i2 = loader._stable_gene_index("ENSG00000139618")
        assert i1 == i2
        assert 0 <= i1 < loader.get_vocab_size()
        assert i1 != loader._stable_gene_index("BRCA1")

    def test_integer_index_lookup_direct(self, tmp_path):
        loader = _make_fallback_loader(tmp_path)
        emb = loader.get_gene_embedding([5, 7])
        torch.testing.assert_close(emb[0], loader.embeddings[5])
        torch.testing.assert_close(emb[1], loader.embeddings[7])

    def test_out_of_range_index_warns_and_defaults_to_zero(self, tmp_path):
        loader = _make_fallback_loader(tmp_path)
        with pytest.warns(UserWarning, match="out of range"):
            emb = loader.get_gene_embedding([10**9])
        torch.testing.assert_close(emb[0], loader.embeddings[0])

    def test_create_gene_id_mapping_consistent_with_lookup(self, tmp_path):
        loader = _make_fallback_loader(tmp_path)
        genes = ["TP53", "BRCA1", "EGFR"]
        mapping = loader.create_gene_id_mapping(genes)
        assert set(mapping) == set(genes)
        assert all(0 <= v < loader.get_vocab_size() for v in mapping.values())
        for gene in genes:
            torch.testing.assert_close(
                loader.get_gene_embedding([gene]),
                loader.embeddings[[mapping[gene]]],
            )

    def test_embeddings_property_requires_load(self, tmp_path):
        loader = _make_fallback_loader(tmp_path)
        loader._embeddings = None
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = loader.embeddings


class TestRealVocabularySemantics:
    """TD-NEW-02: weights + vocab.json must resolve real gene rows."""

    @staticmethod
    def _write_asset(tmp_path, include_vocab=True):
        from safetensors.torch import save_file

        asset_dir = tmp_path / "geneformer_asset"
        asset_dir.mkdir(exist_ok=True)
        embeddings = torch.randn(6, 8)
        save_file(
            {"bert.embeddings.word_embeddings.weight": embeddings},
            str(asset_dir / "model.safetensors"),
        )
        if include_vocab:
            import json

            vocabulary = {"<pad>": 0, "ENSG00000139618": 1, "ENSG00000168610": 3, "ENSG00000141510": 5}
            (asset_dir / "vocab.json").write_text(
                json.dumps(vocabulary), encoding="utf-8"
            )
        return asset_dir, embeddings

    def test_known_gene_hits_real_row(self, tmp_path):
        asset_dir, embeddings = self._write_asset(tmp_path)
        loader = GeneformerEmbeddingLoader(
            model_path=str(asset_dir), device=torch.device("cpu")
        )
        assert loader._vocabulary_is_semantic is True
        got = loader.get_gene_embedding(["ENSG00000168610"])
        torch.testing.assert_close(got, embeddings[3:4, :])

    def test_unknown_gene_raises_key_error(self, tmp_path):
        asset_dir, _ = self._write_asset(tmp_path)
        loader = GeneformerEmbeddingLoader(
            model_path=str(asset_dir), device=torch.device("cpu")
        )
        with pytest.raises(KeyError, match="not in the Geneformer vocabulary"):
            loader.get_gene_embedding(["TP53"])

    def test_create_gene_id_mapping_rejects_unknown_gene(self, tmp_path):
        asset_dir, _ = self._write_asset(tmp_path)
        loader = GeneformerEmbeddingLoader(
            model_path=str(asset_dir), device=torch.device("cpu")
        )
        with pytest.raises(KeyError, match="not in the Geneformer vocabulary"):
            loader.create_gene_id_mapping(["TP53"])

    def test_missing_vocab_fails_fast_without_random_fallback(self, tmp_path):
        asset_dir, _ = self._write_asset(tmp_path, include_vocab=False)
        with pytest.raises(RuntimeError, match="vocab.json"):
            GeneformerEmbeddingLoader(
                model_path=str(asset_dir), device=torch.device("cpu")
            )

    def test_vocab_out_of_range_fails_fast(self, tmp_path):
        asset_dir, embeddings = self._write_asset(tmp_path)
        (asset_dir / "vocab.json").write_text(
            '{"ENSG00000139618": 99}', encoding="utf-8"
        )
        assert embeddings.shape[0] == 6
        with pytest.raises(RuntimeError, match="outside the embedding matrix range"):
            GeneformerEmbeddingLoader(
                model_path=str(asset_dir), device=torch.device("cpu")
            )


class TestDeviceAndSingletonHelpers:
    def test_to_moves_embeddings_and_returns_self(self, tmp_path):
        loader = _make_fallback_loader(tmp_path)
        returned = loader.to(torch.device("cpu"))
        assert returned is loader
        assert loader.device == torch.device("cpu")
        assert loader.embeddings.device.type == "cpu"

    def test_get_provenance_not_loaded_before_any_loader(self, monkeypatch):
        monkeypatch.setattr(gfe, "_geneformer_loader", None)
        monkeypatch.setattr(gfe, "_geneformer_is_fallback", False)
        assert gfe.get_provenance() == "not_loaded"
        assert gfe.is_fallback_mode() is False

    def test_singleton_cached_and_recreated_on_path_change(self, tmp_path):
        gfe._geneformer_loader = None
        path_a = str(tmp_path / "absent_a")
        path_b = str(tmp_path / "absent_b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            first = gfe.get_geneformer_loader(model_path=path_a, device=torch.device("cpu"))
            again = gfe.get_geneformer_loader(model_path=path_a, device=torch.device("cpu"))
            other = gfe.get_geneformer_loader(model_path=path_b, device=torch.device("cpu"))
        assert again is first
        assert other is not first
        assert other.model_path == path_b

    def test_singleton_device_reuses_and_moves(self, tmp_path):
        gfe._geneformer_loader = None
        path = str(tmp_path / "absent_c")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            first = gfe.get_geneformer_loader(model_path=path, device=torch.device("cpu"))
            moved = gfe.get_geneformer_loader(model_path=path, device=torch.device("cpu"))
        assert moved is first
