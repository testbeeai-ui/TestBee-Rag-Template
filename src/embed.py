"""
embed.py — Local BGE-M3 embedding phase.
Loads strictly from data/models/bge-m3/ — no internet downloads at runtime.
"""

from sentence_transformers import SentenceTransformer
from config import BGE_M3_MODEL_PATH


def load_model() -> SentenceTransformer:
    if not BGE_M3_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"BGE-M3 model not found at {BGE_M3_MODEL_PATH}. "
            "Run: python -c \"from sentence_transformers import SentenceTransformer; "
            "SentenceTransformer('BAAI/bge-m3').save('data/models/bge-m3')\" "
            "from inside testbee_local/"
        )
    return SentenceTransformer(str(BGE_M3_MODEL_PATH))


def embed_chunks(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()
