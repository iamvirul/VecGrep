"""Local embedding with auto-detected backend (fastembed ONNX or sentence-transformers)."""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

# Model to use. Default is the fine-tuned code search model.
# Set VECGREP_MODEL=sentence-transformers/all-MiniLM-L6-v2 (or any HF model)
# to use a different model.
#
# Citation for default model:
#   isuruwijesiri. (2026). all-MiniLM-L6-v2-code-search-512 [Model].
#   Hugging Face. https://huggingface.co/isuruwijesiri/all-MiniLM-L6-v2-code-search-512
DEFAULT_MODEL = "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"
MODEL_NAME = os.environ.get("VECGREP_MODEL", DEFAULT_MODEL)

# Backend to use for embedding.
# - "onnx"  (default) — fastembed + ONNX Runtime, ~100ms startup, no PyTorch needed
# - "torch" — sentence-transformers + PyTorch, ~2-3s startup, supports any HF model
BACKEND = os.environ.get("VECGREP_BACKEND", "onnx").lower()

# Batch sizes tuned per device for the torch backend
_TORCH_BATCH_SIZE: dict[str, int] = {
    "cuda": 256,
    "mps": 256,
    "cpu": 64,
}

# ---------------------------------------------------------------------------
# ONNX backend (fastembed) — default
# ---------------------------------------------------------------------------

_onnx_model = None


def _get_onnx_model():
    global _onnx_model
    if _onnx_model is None:
        from fastembed import TextEmbedding  # type: ignore
        from fastembed.common.model_description import ModelSource, PoolingType  # type: ignore

        # Register the default model as a custom model so fastembed can fetch
        # its ONNX files directly from HuggingFace.
        if MODEL_NAME == DEFAULT_MODEL:
            TextEmbedding.add_custom_model(
                model=DEFAULT_MODEL,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=DEFAULT_MODEL),
                dim=384,
                model_file="onnx/model.onnx",
                description="Fine-tuned MiniLM for semantic code search",
            )

        _onnx_model = TextEmbedding(MODEL_NAME)
    return _onnx_model


# ---------------------------------------------------------------------------
# Torch backend (sentence-transformers) — opt-in via VECGREP_BACKEND=torch
# ---------------------------------------------------------------------------

_torch_model = None
_torch_device: str | None = None


def _detect_device() -> str:
    """Return the best available compute device: cuda > mps > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _get_torch_model():
    global _torch_model, _torch_device
    if _torch_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _torch_device = _detect_device()
        _torch_model = SentenceTransformer(MODEL_NAME, device=_torch_device)
    return _torch_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning a normalised float32 array of shape (N, 384)."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    if BACKEND == "torch":
        model = _get_torch_model()
        batch_size = _TORCH_BATCH_SIZE.get(_torch_device or "cpu", 64)
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    else:
        model = _get_onnx_model()
        vecs = np.array(list(model.embed(texts)))

    # Normalise for cosine similarity via dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)
