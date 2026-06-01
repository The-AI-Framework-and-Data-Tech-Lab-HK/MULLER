# SPDX-License-Identifier: MPL-2.0
"""
CLIP image/text embedding helper for the MULLER vector-search demo.

Single source of truth shared by:
  * the Streamlit UI (query-by-image / query-by-text in the Vector Search tab),
  * ``sigmod_demo_revision/vector_search/embed_column.py`` (populate / backfill
    the ``clip_embedding`` column).

Why CLIP ViT-B/32 (openai/clip-vit-base-patch32, 512-d): see
``sigmod_demo_revision/vector_search/ANALYSIS.md`` §3 — image->image P@10≈0.70
(a conservative lower bound; many "misses" are mislabeled images) and near
perfect text->image, all from a single joint embedding space so one 512-d
column serves both modalities. Radford et al., ICML 2021, arXiv:2103.00020.

Design notes
------------
* Lazy, cached model load (first call pays the cost; later calls reuse).
* Offline by default: if the CLIP weights are already cached (standard
  ``~/.cache/huggingface``, or wherever ``HF_HOME`` points), sets
  HF_HUB_OFFLINE so a sandboxed / air-gapped run never tries the network.
* Device defaults to **cpu** — on this machine MPS + the torch×MKL OpenMP
  collision segfaults during batched inference (documented in ANALYSIS.md).
  Override with env ``MULLER_CLIP_DEVICE=mps`` if your box is happy with it.
* Embeddings are L2-normalized float32, so a FAISS ``l2``/``ip`` index gives
  cosine-equivalent ranking.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBED_DIM = 512

ImageLike = Union[str, Path, "np.ndarray", "object"]  # path | array | PIL.Image


def _hub_dir() -> Path:
    """Resolve the HuggingFace hub cache dir (honors HF_HOME, else default)."""
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home) if hf_home else (Path.home() / ".cache" / "huggingface")
    return base / "hub"


def _configure_env() -> None:
    """Defuse the torch×MKL/faiss OpenMP duplicate-runtime abort and, when the
    CLIP weights are already cached, force offline so we never hit the network.
    Idempotent / non-destructive: never overrides values the caller already set,
    and never forces a custom cache path (uses HF's standard
    ``~/.cache/huggingface`` unless the caller set ``HF_HOME``)."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # torch and faiss each ship their own OpenMP runtime; on this machine
    # loading both and running multi-threaded segfaults. Pinning to a single
    # OMP thread (set BEFORE torch/faiss initialize their pools) avoids it and
    # costs little at demo scale (~60 img/s CLIP on CPU). Override if your box
    # is happy multi-threaded.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # If the CLIP weights are visibly cached, stay offline (no network needed).
    model_cache = _hub_dir() / ("models--" + MODEL_NAME.replace("/", "--"))
    if model_cache.is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def get_device() -> str:
    forced = os.environ.get("MULLER_CLIP_DEVICE")
    if forced:
        return forced
    return "cpu"  # safe default; see module docstring


@lru_cache(maxsize=1)
def _load():
    """Load (model, processor, device) once and cache."""
    _configure_env()
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = get_device()
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, proc, device


def _to_pil_list(images: Sequence[ImageLike]):
    from PIL import Image

    out = []
    for im in images:
        if isinstance(im, (str, Path)):
            out.append(Image.open(im).convert("RGB"))
        elif isinstance(im, np.ndarray):
            arr = im
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            if arr.ndim == 2:
                out.append(Image.fromarray(arr, mode="L").convert("RGB"))
            elif arr.ndim == 3 and arr.shape[-1] == 4:
                out.append(Image.fromarray(arr, mode="RGBA").convert("RGB"))
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                out.append(Image.fromarray(arr[..., 0], mode="L").convert("RGB"))
            else:
                out.append(Image.fromarray(arr).convert("RGB"))
        else:  # assume PIL.Image
            out.append(im.convert("RGB"))
    return out


def _as_feature_tensor(out):
    import torch
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"Cannot extract feature tensor from {type(out)}")


def encode_images(images: Sequence[ImageLike], batch_size: int = 32) -> np.ndarray:
    """Return an (N, 512) float32 L2-normalized array of CLIP image embeddings.

    ``images`` may mix file paths, HxWx3 uint8 numpy arrays, and PIL images.
    """
    import torch

    model, proc, device = _load()
    vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            pil = _to_pil_list(list(images[i:i + batch_size]))
            inputs = proc(images=pil, return_tensors="pt").to(device)
            feats = _as_feature_tensor(model.get_image_features(**inputs))
            feats = torch.nn.functional.normalize(feats, dim=-1)
            vecs.append(feats.cpu().numpy().astype(np.float32))
    if not vecs:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(vecs, axis=0), dtype=np.float32)


def encode_texts(texts: Sequence[str]) -> np.ndarray:
    """Return an (N, 512) float32 L2-normalized array of CLIP text embeddings.

    Same vector space as ``encode_images`` -> enables text->image search.
    """
    import torch

    model, proc, device = _load()
    with torch.no_grad():
        inputs = proc(text=list(texts), return_tensors="pt", padding=True).to(device)
        feats = _as_feature_tensor(model.get_text_features(**inputs))
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return np.ascontiguousarray(feats.cpu().numpy().astype(np.float32), dtype=np.float32)


def install_to_numpy_thread_shim() -> None:
    """Swap MULLER's ProcessPoolExecutor-based full-tensor read for a thread
    pool.

    ``Tensor.numpy()`` (used by ``create_vector_index`` and by bulk image
    reads) fans out over chunks with ``ProcessPoolExecutor`` under the default
    ``strategy="processed"``. ``spawn`` re-imports ``__main__`` (which under
    ``streamlit run`` is the Streamlit CLI, not our script) and POSIX-semaphore
    limits can be unavailable in sandboxes — either way it dies. Threads share
    the address space and are plenty for demo-scale data. Idempotent.
    """
    try:
        import concurrent.futures as _cf
        import muller.core.chunk.operations.to_numpy as _tonumpy
    except Exception:
        return
    if getattr(_tonumpy, "_muller_ui_thread_shim", False):
        return
    _tonumpy.ProcessPoolExecutor = _cf.ThreadPoolExecutor
    _tonumpy._muller_ui_thread_shim = True
