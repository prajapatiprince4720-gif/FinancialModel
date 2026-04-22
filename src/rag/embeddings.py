"""
Embedding model wrapper.

Uses sentence-transformers (runs 100% locally, no API key, no cost).
Default model: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - ~80MB download on first use
  - ~14,000 sentences/second on CPU

For higher quality (Q2+), consider:
  - all-mpnet-base-v2        (768-dim, slower, better)
  - BAAI/bge-small-en-v1.5   (384-dim, stronger on retrieval tasks)
  - intfloat/multilingual-e5-small  (if you need Hindi text support)
"""

from functools import lru_cache
from typing import Any

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingModel:
    """
    Thin wrapper around sentence-transformers.
    Lazy-loads the model on first call to avoid startup cost.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} (first time may download ~80MB)")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings. Returns list of float vectors."""
        if not texts:
            return []
        vectors = self.model.encode(texts, show_progress_bar=len(texts) > 50, convert_to_numpy=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for this model."""
        return self.model.get_sentence_embedding_dimension()


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
