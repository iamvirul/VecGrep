"""Local embedding using sentence-transformers."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Embedding model: fine-tuned for semantic code search by @isuruwijesiri
# Citation:
#   isuruwijesiri. (2026). all-MiniLM-L6-v2-code-search-512 [Model].
#   Hugging Face. https://huggingface.co/isuruwijesiri/all-MiniLM-L6-v2-code-search-512
MODEL_NAME = "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"

# Batch sizes tuned per device — GPU/MPS can saturate with larger batches
_BATCH_SIZE: dict[str, int] = {
    "cuda": 256,
    "mps": 256,
    "cpu": 64,
}

_model = None
_device: str | None = None


def _detect_device() -> str:
    """Return the best available compute device: cuda > mps > cpu."""
    if HAS_TORCH:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _get_model():
    global _model, _device
    if _model is None:
        _device = _detect_device()
        _model = SentenceTransformer(MODEL_NAME, device=_device)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning a float32 array of shape (N, 384)."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = _get_model()
    batch_size = _BATCH_SIZE.get(_device or "cpu", 64)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)
