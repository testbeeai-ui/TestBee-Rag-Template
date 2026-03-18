"""
retrieve.py — ChromaDB vector store: write after ingestion, read at query time.
"""

import chromadb
from chromadb import Collection
from config import VECTOR_STORE_DIR, CHROMA_COLLECTION_NAME


def get_or_create_collection() -> Collection:
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_collection(
    collection: Collection,
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    source_file: str,
) -> None:
    ids = [f"{source_file}_chunk_{i}" for i in range(len(texts))]
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_collection(
    collection: Collection,
    query_embedding: list[float],
    filters: dict | None = None,
    n_results: int = 5,
) -> list[dict]:
    # ChromaDB v1.x requires $and operator for multiple filters
    if filters and len(filters) > 1:
        where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
    elif filters and len(filters) == 1:
        k, v = next(iter(filters.items()))
        where = {k: {"$eq": v}}
    else:
        where = None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": text, "metadata": meta, "distance": dist})
    return output
