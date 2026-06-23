"""Local embedding adapter for LangGraph PostgresStore semantic memory search."""
from __future__ import annotations

from functools import lru_cache
from typing import List


class LocalMemoryEmbeddings:
    """Small callable wrapper around SentenceTransformer for store indexing."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


@lru_cache(maxsize=1)
def get_memory_embeddings(model_name: str) -> LocalMemoryEmbeddings:
    return LocalMemoryEmbeddings(model_name)
