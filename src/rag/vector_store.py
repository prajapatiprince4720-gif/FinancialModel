"""
ChromaDB vector store wrapper.

ChromaDB stores:
  - The raw text chunks
  - Their vector embeddings
  - Metadata (ticker, section, period, source)

It runs entirely locally — no cloud, no API key, no cost.
Data persists to disk at config.chroma_persist_dir.
"""

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VectorStore:

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore ready — collection '{self.collection_name}' "
            f"({self._collection.count()} docs) at {self.persist_dir}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────────────────────

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        ids: list[str],
    ) -> None:
        """
        Upsert documents into the collection.
        Uses upsert so re-running the pipeline doesn't create duplicates.
        """
        if not texts:
            return
        self._collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Upserted {len(texts)} chunks into '{self.collection_name}'")

    def delete_ticker(self, ticker: str) -> None:
        """Remove all chunks for a specific ticker (useful for refreshing stale data)."""
        self._collection.delete(where={"ticker": ticker})
        logger.info(f"Deleted all chunks for {ticker}")

    # ──────────────────────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search. Returns top-k results sorted by cosine similarity.
        `where` filters by metadata, e.g. {"ticker": "RELIANCE.NS"}
        """
        n = top_k or settings.retrieval_top_k
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "text": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),  # convert distance → similarity
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ]

    def get_all_tickers(self) -> list[str]:
        """Return list of unique tickers currently in the store."""
        try:
            results = self._collection.get(include=["metadatas"])
            tickers = {m.get("ticker", "") for m in results["metadatas"]}
            return sorted(t for t in tickers if t)
        except Exception:
            return []

    def count(self) -> int:
        return self._collection.count()
