"""
Retriever: given a natural-language query, return the most relevant text chunks.

Supports two modes:
  1. Dense (semantic) — pure vector similarity via ChromaDB
  2. Hybrid — dense + keyword re-rank (BM25-style)

Start with dense for Q1. Add hybrid in Q2 for better precision on
financial terms (e.g. "EBITDA margin Q3FY24" needs exact match too).
"""

from typing import Any

from src.rag.embeddings import EmbeddingModel, get_embedding_model
from src.rag.vector_store import VectorStore
from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class Retriever:

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.store = vector_store or VectorStore()
        self.embedder = embedding_model or get_embedding_model()

    # ──────────────────────────────────────────────────────────────────────────
    # Main retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        ticker: str | None = None,
        section: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:   Natural language question
            ticker:  Filter to a specific stock (e.g. 'RELIANCE.NS')
            section: Filter to a section ('news', 'income_statement', etc.)
            top_k:   Number of results (default from settings)

        Returns:
            List of {text, metadata, score} dicts, sorted by relevance.
        """
        query_vec = self.embedder.embed_one(query)

        where: dict[str, Any] | None = None
        if ticker and section:
            where = {"$and": [{"ticker": ticker}, {"section": section}]}
        elif ticker:
            where = {"ticker": ticker}
        elif section:
            where = {"section": section}

        results = self.store.query(query_vec, top_k=top_k, where=where)
        results = self._deduplicate(results)
        logger.debug(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
        return results

    def retrieve_multi_section(
        self,
        query: str,
        ticker: str,
        sections: list[str],
        top_k_per_section: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve from multiple sections independently, then merge.
        Useful when you want balanced coverage (e.g. 3 financial chunks + 3 news chunks).
        """
        all_results: list[dict[str, Any]] = []
        for section in sections:
            results = self.retrieve(query, ticker=ticker, section=section, top_k=top_k_per_section)
            all_results.extend(results)

        # Re-sort by score after merging
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results

    def format_context(self, results: list[dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM prompt.
        Each chunk is numbered and labelled with its source metadata.
        """
        if not results:
            return "No relevant context found."

        lines = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            label = f"[{i}] {meta.get('ticker', '')} | {meta.get('section', '')} | {meta.get('period', meta.get('published_at', ''))}"
            lines.append(label)
            lines.append(r["text"])
            lines.append("")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────

    def _deduplicate(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for r in results:
            key = r["text"][:120]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
