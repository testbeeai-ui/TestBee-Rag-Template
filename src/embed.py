"""
embed.py — Local BGE-M3 embedding phase.
Loads strictly from data/models/bge-m3/ — no internet downloads at runtime.
"""

import os
import gc
import torch
from sentence_transformers import SentenceTransformer
from config import BGE_M3_MODEL_PATH

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_model() -> SentenceTransformer:
    if not BGE_M3_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"BGE-M3 model not found at {BGE_M3_MODEL_PATH}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. A GPU is required.")

    # Aggressively free all cached VRAM before loading
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {free_gb:.2f} GB free / {total_gb:.2f} GB total")

    if free_gb < 1.3:
        raise RuntimeError(
            f"Only {free_gb:.2f} GB VRAM free. BGE-M3 fp16 needs ~1.3 GB.\n"
            "Close other GPU-heavy apps (browser, games, other Python processes) and retry."
        )

    model = SentenceTransformer(
        str(BGE_M3_MODEL_PATH),
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )
    print("  BGE-M3 loaded on GPU (fp16).")
    return model


def embed_chunks(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    # Free VRAM after each batch
    torch.cuda.empty_cache()
    return embeddings.tolist()
