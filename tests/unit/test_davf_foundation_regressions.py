import hashlib
import pickle
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.models.davf_checkpoint_utils import load_checkpoint, save_checkpoint
from src.models.geneformer_embedding import GeneformerEmbeddingLoader


class TestGeneformerEmbeddingLoaderRegressions:
    def test_fallback_hash_lookup_does_not_pollute_vocabulary(self):
        """TD-NEW-02: fallback hashes stay stable but are never written back."""
        loader = GeneformerEmbeddingLoader.__new__(GeneformerEmbeddingLoader)
        loader._vocab_size = 30000
        loader._gene_to_idx = {}
        loader._idx_to_gene = {}
        loader._embeddings = torch.randn(loader._vocab_size, 8)
        loader.device = torch.device("cpu")
        loader._vocabulary_is_semantic = False

        gene_id = "ENSG00000139618"
        embeddings = loader.get_gene_embedding([gene_id])

        expected_idx = int(hashlib.sha256(gene_id.encode("utf-8")).hexdigest(), 16) % loader._vocab_size

        assert gene_id not in loader._gene_to_idx
        assert torch.equal(embeddings, loader._embeddings[torch.tensor([expected_idx])])


class TestDAVFCheckpointUtilsRegressions:
    def test_save_checkpoint_load_checkpoint_round_trip_without_explicit_config(self, tmp_path):
        model = nn.Linear(2, 2)
        checkpoint_path = tmp_path / "round_trip.pt"

        save_checkpoint(str(checkpoint_path), model)
        checkpoint = load_checkpoint(str(checkpoint_path))

        assert "model_state_dict" in checkpoint
        assert checkpoint["config"] == {}

    def test_load_checkpoint_supports_phase10_bundled_legacy_checkpoint(self):
        """Legacy pickle checkpoints are rejected with a clear migration error."""
        msg = (
            r"Legacy checkpoint .* cannot be loaded with weights_only=True\. "
            r"Run: python scripts/tools/migrate_legacy_checkpoint\.py"
        )
        with pytest.raises(RuntimeError, match=msg):
            load_checkpoint(
                "checkpoints/davf/model_a_finetuned/best_model.pt",
                strict=False,
            )

    def test_load_checkpoint_does_not_unsafe_fallback_without_opt_in(self, tmp_path):
        checkpoint_path = tmp_path / "legacy.pt"
        checkpoint_path.write_bytes(b"placeholder")

        with patch("src.models.davf_checkpoint_utils.torch.load") as mock_load:
            mock_load.side_effect = pickle.UnpicklingError("legacy pickle")

            with pytest.raises(RuntimeError):
                load_checkpoint(str(checkpoint_path))

        assert mock_load.call_count == 1

    def test_load_checkpoint_rejects_legacy_checkpoint_even_without_opt_in(self, tmp_path):
        """Legacy checkpoints are rejected regardless of any opt-in — SC-01 removed unsafe fallback."""
        checkpoint_path = tmp_path / "legacy.pt"
        checkpoint_path.write_bytes(b"placeholder")

        with patch("src.models.davf_checkpoint_utils.torch.load") as mock_load:
            mock_load.side_effect = pickle.UnpicklingError("legacy pickle")

            with pytest.raises(RuntimeError):
                load_checkpoint(str(checkpoint_path))

        assert mock_load.call_count == 1

    def test_load_checkpoint_rejects_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("checkpoints/does-not-exist.pt")
