"""
Geneformer Embedding Loader

Loads pretrained Geneformer embeddings from HuggingFace and provides
gene embedding lookup functionality.

Geneformer: https://huggingface.co/ctheodoris/Geneformer
Embedding dimension: 1152 (Geneformer V1)
"""

import logging
import hashlib
import os
import warnings
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

_geneformer_is_fallback = False


class GeneformerEmbeddingLoader:
    """
    Loader for Geneformer pretrained gene embeddings.

    Loads the Geneformer model from HuggingFace and extracts the
    gene embedding layer for use in DAVF and other models.

    Attributes:
        embedding_dim: Dimension of gene embeddings (1152 for Geneformer V1)
        gene_to_idx: Mapping from gene names to embedding indices
        idx_to_gene: Reverse mapping from indices to gene names
    """

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (1152 for Geneformer V1)."""
        return self._embedding_dim

    def __init__(
        self,
        model_path: str = "ctheodoris/Geneformer",
        device: Optional[torch.device] = None,
        strict: bool = False
    ):
        """
        Initialize Geneformer embedding loader.

        Args:
            model_path: HuggingFace model path or local directory.
                       Default: "ctheodoris/Geneformer"
            device: Device to load embeddings on. Default: CUDA if available.
            strict: Fail fast if Geneformer cannot be loaded, instead of
                   falling back to random embeddings.
                   Overridden to True if PTM2CELLNET_STRICT_MODEL_ASSETS is set.
        """
        self.model_path = model_path
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Strict mode: fail-fast instead of random fallback
        _global_strict = os.environ.get("PTM2CELLNET_STRICT_MODEL_ASSETS", "").lower() in ("1", "true", "yes")
        if _global_strict:
            strict = True
        self.strict = strict

        # Geneformer settings
        self._embedding_dim = 1152  # Geneformer V1 uses 1152 dimensions
        self.max_seq_len = 2048  # Geneformer's max sequence length

        # Load model and extract embeddings
        self._model = None
        self._embeddings = None
        self._gene_to_idx: dict[str, int] = {}
        self._idx_to_gene: dict[int, str] = {}
        self._vocab_size = 0

        self._load_model()

    def _load_model(self):
        """Load Geneformer model and extract gene embeddings."""
        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise ImportError(
                "transformers library required for Geneformer. "
                "Install with: pip install transformers"
            ) from None

        logger.info(f"Loading Geneformer from {self.model_path}...")

        try:
            import os
            from pathlib import Path

            # Define local safetensors paths to check
            project_root = Path(__file__).parent.parent.parent
            local_safetensors_paths = [
                project_root / "models" / "geneformer_model.safetensors",
                project_root / "models" / "model.safetensors",
                Path("models/geneformer_model.safetensors"),
                Path("models/model.safetensors"),
            ]

            # Check if model_path is a direct path to a safetensors file
            direct_safetensors = os.path.isfile(self.model_path)

            # Check local safetensors paths
            local_safetensors_path = None
            for path in local_safetensors_paths:
                if os.path.isfile(path):
                    local_safetensors_path = str(path)
                    break

            # Check if local directory with safetensors
            local_dir = os.path.join(self.model_path, 'model.safetensors')
            use_safetensors_dir = os.path.isfile(local_dir)

            if local_safetensors_path:
                # Load from local safetensors file (priority)
                logger.info(f"Loading from local safetensors file: {local_safetensors_path}")
                from safetensors.torch import load_file
                state_dict = load_file(local_safetensors_path)
                logger.info(f"Loaded safetensors with {len(state_dict)} keys")

                # Extract word embeddings using correct key for Geneformer
                embedding_key = 'bert.embeddings.word_embeddings.weight'
                if embedding_key not in state_dict:
                    # Try alternative key names
                    for key in state_dict.keys():
                        if 'word_embeddings' in key and 'weight' in key:
                            embedding_key = key
                            logger.info(f"Using alternative embedding key: {embedding_key}")
                            break
                    else:
                        raise KeyError(f"Could not find word_embeddings weight. Available keys: {list(state_dict.keys())[:10]}...")

                self._embeddings = state_dict[embedding_key].detach()
                logger.info(f"Extracted embeddings from {embedding_key}: {self._embeddings.shape}")

                # Get vocab size and dim from embeddings
                self._vocab_size = self._embeddings.shape[0]
                self._embedding_dim = self._embeddings.shape[1]

                # Build gene index mappings (using integer indices as Geneformer uses hash-based mapping)
                self._gene_to_idx = {str(i): i for i in range(self._vocab_size)}
                self._idx_to_gene = {i: str(i) for i in range(self._vocab_size)}

                self._embeddings = self._embeddings.to(self.device)
                logger.info(f"Geneformer loaded. Vocab: {self._vocab_size}, Dim: {self.embedding_dim}")
            elif direct_safetensors or use_safetensors_dir:
                # Load from local safetensors file
                safetensors_path = self.model_path if direct_safetensors else local_dir
                logger.info(f"Loading from local safetensors file: {safetensors_path}")
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path)
                logger.info(f"Loaded safetensors with {len(state_dict)} keys")

                # Extract word embeddings directly from state dict
                embedding_key = 'bert.embeddings.word_embeddings.weight'
                if embedding_key not in state_dict:
                    # Try alternative key names
                    for key in state_dict.keys():
                        if 'word_embeddings' in key and 'weight' in key:
                            embedding_key = key
                            break

                self._embeddings = state_dict[embedding_key].detach()
                logger.info(f"Extracted embeddings from {embedding_key}: {self._embeddings.shape}")

                # Get vocab size and dim from embeddings
                self._vocab_size = self._embeddings.shape[0]
                self._embedding_dim = self._embeddings.shape[1]

                # Build gene index mappings
                self._gene_to_idx = {str(i): i for i in range(self._vocab_size)}
                self._idx_to_gene = {i: str(i) for i in range(self._vocab_size)}

                self._embeddings = self._embeddings.to(self.device)
                logger.info(f"Geneformer loaded. Vocab: {self._vocab_size}, Dim: {self.embedding_dim}")
            else:
                # Load from HuggingFace
                config = AutoConfig.from_pretrained(self.model_path)
                self._vocab_size = config.vocab_size
                model = AutoModel.from_pretrained(self.model_path)
                model.to(self.device)
                model.eval()
                self._model = model

                # Extract gene embedding layer
                # Geneformer uses 'embeddings.word_embedding' for gene tokens
                if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embedding'):
                    self._embeddings = model.embeddings.word_embedding.weight.detach()
                else:
                    # Try to find embedding layer in state dict
                    state_dict = model.state_dict()
                    embedding_key = None
                    for key in state_dict.keys():
                        if 'word_embedding' in key or 'embedding' in key:
                            if 'LayerNorm' not in key and 'position' not in key:
                                embedding_key = key
                                break

                    if embedding_key is None:
                        # Fallback: look for any large embedding matrix
                        for key in state_dict.keys():
                            tensor = state_dict[key]
                            if tensor.dim() == 2 and tensor.shape[0] > 1000:
                                embedding_key = key
                                break

                    if embedding_key:
                        self._embeddings = state_dict[embedding_key].detach()
                        logger.info(f"Extracted embeddings from key: {embedding_key}")
                    else:
                        raise ValueError("Could not find gene embedding layer in Geneformer model")

                self._embeddings = self._embeddings.to(self.device)

                # Update vocab size and embedding dim from actual embedding matrix
                self._vocab_size = self._embeddings.shape[0]
                self._embedding_dim = self._embeddings.shape[1]

                # Build gene index mappings
                # Geneformer uses ENSEMBL IDs internally, indexed 0 to vocab_size-1
                self._gene_to_idx = {str(i): i for i in range(self._vocab_size)}
                self._idx_to_gene = {i: str(i) for i in range(self._vocab_size)}

                logger.info(
                    f"Geneformer loaded successfully. "
                    f"Vocab size: {self._vocab_size}, Embedding dim: {self.embedding_dim}"
                )

        except (OSError, ValueError, RuntimeError) as e:
            if self.strict:
                raise RuntimeError(
                    f"Geneformer model '{self.model_path}' failed to load "
                    f"and PTM2CELLNET_STRICT_MODEL_ASSETS is set. "
                    f"Error: {e}. Install the model or unset PTM2CELLNET_STRICT_MODEL_ASSETS."
                ) from e
            logger.warning(
                "Geneformer model '%s' failed to load: %s. "
                "Falling back to random embeddings.",
                self.model_path, e
            )
            self._use_fallback()

    def _stable_gene_index(self, gene_id: str) -> int:
        """Map unknown gene identifiers to a stable fallback index."""
        digest = hashlib.sha256(gene_id.encode("utf-8")).hexdigest()
        return int(digest, 16) % self._vocab_size

    def _use_fallback(self):
        """Use random embeddings as fallback when Geneformer cannot be loaded."""
        global _geneformer_is_fallback
        _geneformer_is_fallback = True

        warnings.warn(
            "Geneformer could not be loaded. Using random embeddings. "
            "This is NOT suitable for production use. "
            "Set PTM2CELLNET_STRICT_MODEL_ASSETS=1 to enforce fail-fast.",
            UserWarning,
            stacklevel=2,
        )
        # Create random embeddings with correct dimension
        self._vocab_size = 30000  # Typical Geneformer vocab size
        self._embeddings = torch.randn(
            self._vocab_size,
            self._embedding_dim,
            device=self.device
        )
        self._gene_to_idx = {str(i): i for i in range(self._vocab_size)}
        self._idx_to_gene = {i: str(i) for i in range(self._vocab_size)}

    def get_gene_embedding(self, gene_ids: Union[List[str], List[int]]) -> torch.Tensor:
        """
        Get gene embeddings for a list of gene identifiers.

        Args:
            gene_ids: List of gene identifiers. Can be:
                     - ENSEMBL IDs (strings like "ENSG00000139618")
                     - Integer indices into Geneformer vocabulary

        Returns:
            embeddings: [num_genes, embedding_dim] tensor of gene embeddings
        """
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded. Call _load_model() first.")

        indices = []
        for gene_id in gene_ids:
            if isinstance(gene_id, str):
                # Try to parse as ENSEMBL ID or use as-is
                # Geneformer internally maps all genes to indices 0..vocab_size-1
                # We use a simple hash-based mapping for unknown genes
                if gene_id in self._gene_to_idx:
                    indices.append(self._gene_to_idx[gene_id])
                else:
                    # Stable hash fallback for genes not in vocabulary.
                    gene_hash = self._stable_gene_index(gene_id)
                    indices.append(gene_hash)
                    if gene_id not in self._gene_to_idx:
                        self._gene_to_idx[gene_id] = gene_hash
                        self._idx_to_gene[gene_hash] = gene_id
            else:
                # Already an integer index
                idx = int(gene_id)
                if 0 <= idx < self._vocab_size:
                    indices.append(idx)
                else:
                    warnings.warn(f"Gene index {idx} out of range [0, {self._vocab_size})", stacklevel=2)
                    indices.append(0)  # Default to index 0

        # Lookup embeddings
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        embeddings = self._embeddings[indices_tensor]

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension (192 for Geneformer)."""
        return self.embedding_dim

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._vocab_size

    def create_gene_id_mapping(
        self,
        dataset_genes: List[str],
        gene_name_type: str = "ensembl"
    ) -> Dict[str, int]:
        """
        Create a mapping from dataset gene IDs to Geneformer indices.

        Args:
            dataset_genes: List of gene identifiers from the dataset.
            gene_name_type: Type of gene identifiers ("ensembl", "symbol", "entrez").
                          Currently treats all as strings for hashing.

        Returns:
            mapping: Dictionary mapping dataset gene ID to Geneformer index.
        """
        mapping = {}
        for gene in dataset_genes:
            if gene in self._gene_to_idx:
                mapping[gene] = self._gene_to_idx[gene]
            else:
                gene_hash = self._stable_gene_index(gene)
                mapping[gene] = gene_hash

        return mapping

    @property
    def embeddings(self) -> torch.Tensor:
        """Return the full embedding matrix."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded.")
        return self._embeddings

    def to(self, device: torch.device):
        """Move embeddings to specified device."""
        self.device = device
        if self._embeddings is not None:
            self._embeddings = self._embeddings.to(device)
        return self

    @property
    def model_provenance(self) -> str:
        """Return provenance information for the loaded model.

        Returns:
            'real' if genuine Geneformer weights are loaded.
            'fallback_random' if using random fallback embeddings.
        """
        return "fallback_random" if _geneformer_is_fallback else "real"


# Singleton instance for shared use across the project
_geneformer_loader: Optional[GeneformerEmbeddingLoader] = None


def is_fallback_mode() -> bool:
    """Return True if the Geneformer loader is using random (non-production) embeddings."""
    return _geneformer_is_fallback


def get_provenance() -> str:
    """Return the provenance of the global Geneformer loader.

    Returns:
        'real' if genuine Geneformer weights are loaded.
        'fallback_random' if using random fallback embeddings.
        'not_loaded' if no Geneformer loader has been created yet.
    """
    if _geneformer_loader is not None:
        return _geneformer_loader.model_provenance
    if _geneformer_is_fallback:
        return "fallback_random"
    return "not_loaded"


def get_geneformer_loader(
    model_path: str = "ctheodoris/Geneformer",
    device: Optional[torch.device] = None
) -> GeneformerEmbeddingLoader:
    """
    Get or create the global GeneformerEmbeddingLoader instance.

    Args:
        model_path: HuggingFace model path.
        device: Device for embeddings.

    Returns:
        Global GeneformerEmbeddingLoader instance.
    """
    global _geneformer_loader
    target_device = torch.device(device) if device is not None else None

    if _geneformer_loader is None or _geneformer_loader.model_path != model_path:
        _geneformer_loader = GeneformerEmbeddingLoader(model_path, target_device)
    elif target_device is not None and _geneformer_loader.device != target_device:
        _geneformer_loader.to(target_device)
    return _geneformer_loader
