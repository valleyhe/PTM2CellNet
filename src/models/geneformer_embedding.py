"""
Geneformer Embedding Loader

Loads pretrained Geneformer embeddings from HuggingFace and provides
gene embedding lookup functionality.

Geneformer: https://huggingface.co/ctheodoris/Geneformer
Embedding dimension: 1152 (Geneformer V1)
"""

import logging
import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

_geneformer_is_fallback = False


class GeneformerVocabularyError(RuntimeError):
    """Weights loaded but the gene token vocabulary is missing or invalid.

    This is a data-correctness failure, not a load failure: falling back to
    random embeddings here would silently strip every gene lookup of its
    semantics, so the error is never downgraded to the random fallback.
    """


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

    @property
    def gene_to_idx(self) -> Dict[str, int]:
        """Public read-only view of the token vocabulary (TD-NEW-04)."""
        return dict(self._gene_to_idx)

    def __init__(
        self, model_path: str = "ctheodoris/Geneformer", device: Optional[torch.device] = None, strict: bool = False
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
        # TD-NEW-02: True only when the token vocabulary carries gene
        # semantics (vocab.json was found next to the loaded weights).
        self._vocabulary_is_semantic = False

        self._load_model()

    def _load_vocabulary(self, candidate_dirs: List[Path], model_path_for_hub: Optional[str] = None) -> None:
        """TD-NEW-02: build the real gene vocabulary or fail fast.

        The token matrix only has gene semantics when ``vocab.json`` is
        available next to the weights (or from the hub for a remote repo).
        A successfully loaded model without a vocabulary is a silent
        data-correctness hazard — every gene lookup would miss — so this
        raises instead of leaving an integer-string vocabulary in place.
        """
        candidates: List[Path] = [directory / "vocab.json" for directory in candidate_dirs]
        for vocab_path in candidates:
            if vocab_path.is_file():
                payload = json.loads(vocab_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or not payload:
                    raise GeneformerVocabularyError(f"Invalid vocab.json (must be a non-empty object): {vocab_path}")
                self._gene_to_idx = {str(token): int(index) for token, index in payload.items()}
                self._idx_to_gene = {index: token for token, index in self._gene_to_idx.items()}
                out_of_range = [token for token, index in self._gene_to_idx.items() if index >= self._vocab_size]
                if out_of_range:
                    raise GeneformerVocabularyError(
                        f"vocab.json contains {len(out_of_range)} tokens outside the embedding "
                        f"matrix range [0, {self._vocab_size}): {vocab_path}"
                    )
                self._vocabulary_is_semantic = True
                logger.info("Loaded gene vocabulary from %s (%d tokens)", vocab_path, len(self._gene_to_idx))
                return

        if model_path_for_hub and not Path(model_path_for_hub).is_dir():
            # The official ctheodoris/Geneformer repository stores its token
            # vocabulary as a pickled dict under geneformer/, not as a root
            # vocab.json; both locations are tried in order.
            hub_candidates = ("vocab.json", "geneformer/token_dictionary_gc104M.pkl")
            for filename in hub_candidates:
                try:
                    from huggingface_hub import hf_hub_download

                    hub_path = Path(hf_hub_download(repo_id=model_path_for_hub, filename=filename))
                except Exception:  # noqa: BLE001 - try the next known location
                    continue
                if hub_path.suffix == ".json":
                    payload = json.loads(hub_path.read_text(encoding="utf-8"))
                else:
                    payload = self._load_official_pickle_vocab(hub_path)
                if not isinstance(payload, dict) or not payload:
                    raise GeneformerVocabularyError(
                        f"Invalid vocabulary file downloaded for {model_path_for_hub!r}: {filename}"
                    )
                self._gene_to_idx = {str(token): int(index) for token, index in payload.items()}
                self._idx_to_gene = {index: token for token, index in self._gene_to_idx.items()}
                self._vocabulary_is_semantic = True
                logger.info(
                    "Loaded gene vocabulary from hub for %s (%s, %d tokens)",
                    model_path_for_hub,
                    filename,
                    len(self._gene_to_idx),
                )
                return
            raise GeneformerVocabularyError(
                f"Geneformer weights loaded from {model_path_for_hub!r} but none of the "
                f"known vocabulary files {hub_candidates} could be located or read. "
                "Without the real token vocabulary every gene lookup would silently "
                "miss; provide vocab.json next to the weights."
            )

        raise GeneformerVocabularyError(
            "Geneformer weights loaded but no vocab.json was found in "
            f"{[str(directory) for directory in candidate_dirs]}. Without the real "
            "token vocabulary every gene lookup would silently miss; provide "
            "vocab.json next to the weights."
        )

    def _load_official_pickle_vocab(self, path: Path) -> dict:
        """Load the official Geneformer pickle vocab only with an explicit SHA pin."""

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = os.environ.get("PTM2CELLNET_GENEFORMER_VOCAB_SHA256", "").strip().lower()
        allow = os.environ.get("PTM2CELLNET_ALLOW_GENEFORMER_PICKLE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        production = os.environ.get("PTM2CELLNET_ENV", "").strip().lower() == "production"
        strict = bool(getattr(self, "strict", False))
        if expected:
            if digest != expected:
                raise GeneformerVocabularyError(
                    f"Geneformer pickle vocab sha256 mismatch for {path}: got {digest}, expected {expected}"
                )
        elif not allow and (production or strict):
            raise GeneformerVocabularyError(
                "refusing pickle.load of Geneformer vocabulary without "
                "PTM2CELLNET_GENEFORMER_VOCAB_SHA256 or PTM2CELLNET_ALLOW_GENEFORMER_PICKLE=1; "
                "provide vocab.json instead (M7 still requires Gate-E before deleting this loader)"
            )
        elif not allow:
            logger.warning(
                "Loading unpinned Geneformer pickle vocabulary %s (sha256=%s). "
                "This is forbidden when PTM2CELLNET_ENV=production or strict=True; "
                "pin PTM2CELLNET_GENEFORMER_VOCAB_SHA256 or provide vocab.json.",
                path,
                digest,
            )
        import pickle

        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - pinned official artifact or explicit opt-in
        if not isinstance(payload, dict):
            raise GeneformerVocabularyError(f"Geneformer pickle vocab must be a dict: {path}")
        logger.warning("Loaded Geneformer vocabulary from pickle %s (sha256=%s)", path, digest)
        return payload

    def _load_model(self):
        """Load Geneformer model and extract gene embeddings."""
        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise ImportError(
                "transformers library required for Geneformer. Install with: pip install transformers"
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
            local_dir = os.path.join(self.model_path, "model.safetensors")
            use_safetensors_dir = os.path.isfile(local_dir)

            if local_safetensors_path:
                # Load from local safetensors file (priority)
                logger.info(f"Loading from local safetensors file: {local_safetensors_path}")
                from safetensors.torch import load_file

                state_dict = load_file(local_safetensors_path)
                logger.info(f"Loaded safetensors with {len(state_dict)} keys")

                # Extract word embeddings using correct key for Geneformer
                embedding_key = "bert.embeddings.word_embeddings.weight"
                if embedding_key not in state_dict:
                    # Try alternative key names
                    for key in state_dict.keys():
                        if "word_embeddings" in key and "weight" in key:
                            embedding_key = key
                            logger.info(f"Using alternative embedding key: {embedding_key}")
                            break
                    else:
                        raise KeyError(
                            f"Could not find word_embeddings weight. Available keys: {list(state_dict.keys())[:10]}..."
                        )

                self._embeddings = state_dict[embedding_key].detach()
                logger.info(f"Extracted embeddings from {embedding_key}: {self._embeddings.shape}")

                # Get vocab size and dim from embeddings
                self._vocab_size = self._embeddings.shape[0]
                self._embedding_dim = self._embeddings.shape[1]

                # TD-NEW-02: the vocabulary must come from vocab.json; the
                # embedding row indices alone carry no gene semantics.
                vocab_dirs = [Path(local_safetensors_path).parent]
                if os.path.isdir(self.model_path):
                    vocab_dirs.append(Path(self.model_path))
                self._load_vocabulary(vocab_dirs)

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
                embedding_key = "bert.embeddings.word_embeddings.weight"
                if embedding_key not in state_dict:
                    # Try alternative key names
                    for key in state_dict.keys():
                        if "word_embeddings" in key and "weight" in key:
                            embedding_key = key
                            break

                self._embeddings = state_dict[embedding_key].detach()
                logger.info(f"Extracted embeddings from {embedding_key}: {self._embeddings.shape}")

                # Get vocab size and dim from embeddings
                self._vocab_size = self._embeddings.shape[0]
                self._embedding_dim = self._embeddings.shape[1]

                vocab_dirs = [Path(safetensors_path).parent]
                self._load_vocabulary(vocab_dirs)

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
                if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embedding"):
                    self._embeddings = model.embeddings.word_embedding.weight.detach()
                else:
                    # Try to find embedding layer in state dict
                    state_dict = model.state_dict()
                    embedding_key = None
                    for key in state_dict.keys():
                        if "word_embedding" in key or "embedding" in key:
                            if "LayerNorm" not in key and "position" not in key:
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

                # Geneformer ships its gene tokens in vocab.json; the row
                # indices alone carry no gene semantics (TD-NEW-02).
                vocab_dirs = [Path(self.model_path)] if os.path.isdir(self.model_path) else []
                self._load_vocabulary(vocab_dirs, model_path_for_hub=self.model_path)

                logger.info(
                    f"Geneformer loaded successfully. "
                    f"Vocab size: {self._vocab_size}, Embedding dim: {self.embedding_dim}"
                )

        except (OSError, ValueError, RuntimeError) as e:
            if isinstance(e, GeneformerVocabularyError):
                # A missing/invalid vocabulary is a data-correctness failure;
                # it must never be downgraded to random fallback embeddings.
                raise
            if self.strict:
                raise RuntimeError(
                    f"Geneformer model '{self.model_path}' failed to load "
                    f"and PTM2CELLNET_STRICT_MODEL_ASSETS is set. "
                    f"Error: {e}. Install the model or unset PTM2CELLNET_STRICT_MODEL_ASSETS."
                ) from e
            logger.warning(
                "Geneformer model '%s' failed to load: %s. Falling back to random embeddings.", self.model_path, e
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
        self._embeddings = torch.randn(self._vocab_size, self._embedding_dim, device=self.device)
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
                if gene_id in self._gene_to_idx:
                    indices.append(self._gene_to_idx[gene_id])
                elif self._vocabulary_is_semantic:
                    # TD-NEW-02: with a real vocabulary an unknown gene is an
                    # error; hashing it onto a random row would fabricate an
                    # embedding and polluting the vocabulary would hide the
                    # miss from every later lookup.
                    raise KeyError(f"Gene id {gene_id!r} is not in the Geneformer vocabulary")
                else:
                    if self.strict:
                        raise KeyError(f"Gene id {gene_id!r} cannot be hashed onto a Geneformer row in strict mode")
                    # Explicit non-production fallback mode (random weights,
                    # loud warning, provenance flag): keep shape compatibility
                    # via a stable hash, but never write it back.
                    indices.append(self._stable_gene_index(gene_id))
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

    def create_gene_id_mapping(self, dataset_genes: List[str], gene_name_type: str = "ensembl") -> Dict[str, int]:
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
            elif self._vocabulary_is_semantic:
                raise KeyError(f"Gene id {gene!r} is not in the Geneformer vocabulary")
            elif self.strict:
                raise KeyError(f"Gene id {gene!r} cannot be hashed onto a Geneformer row in strict mode")
            else:
                mapping[gene] = self._stable_gene_index(gene)

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
    def model_source(self) -> str:
        """Return the source of the loaded model."""
        if _geneformer_is_fallback:
            return "fallback_random"
        return "real"

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


def get_model_source() -> str:
    """Return the model source for the global Geneformer loader."""
    loader = get_geneformer_loader()
    return loader.model_source


def get_geneformer_loader(
    model_path: str = "ctheodoris/Geneformer", device: Optional[torch.device] = None
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
