"""
RAG ingestion: ties DataPipeline → EmbeddingModel → VectorStore.

Run this to build / refresh the knowledge base for one or all tickers:
    python -m src.rag.ingestion --ticker RELIANCE.NS
    python -m src.rag.ingestion --all
    python -m src.rag.ingestion --ticker RELIANCE.NS --refresh
"""

import argparse
import hashlib
from typing import Any

from src.data_pipeline.pipeline import DataPipeline
from src.rag.embeddings import get_embedding_model
from src.rag.vector_store import VectorStore
from config.nifty50_tickers import NIFTY50_TICKERS
from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

BATCH_SIZE = 64  # embed this many chunks at once to manage memory


class RAGIngestion:

    def __init__(self) -> None:
        self.pipeline = DataPipeline()
        self.embedder = get_embedding_model()
        self.store = VectorStore()

    # ──────────────────────────────────────────────────────────────────────────

    def ingest(self, ticker_yf: str, refresh: bool = False) -> int:
        """
        Fetch, process, embed, and store all data for one ticker.
        Returns the number of chunks ingested.
        """
        if refresh:
            logger.info(f"Refreshing {ticker_yf} — deleting existing chunks")
            self.store.delete_ticker(ticker_yf)

        logger.info(f"Ingesting {ticker_yf}...")
        chunks = self.pipeline.run(ticker_yf)

        if not chunks:
            logger.warning(f"No chunks generated for {ticker_yf}")
            return 0

        total = 0
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start: batch_start + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            embeddings = self.embedder.embed(texts)
            metadatas = [
                {
                    "ticker": c.get("ticker", ticker_yf),
                    "section": c.get("section", "unknown"),
                    "period": c.get("period", ""),
                    "source": c.get("source", ""),
                    "published_at": c.get("published_at", ""),
                }
                for c in batch
            ]
            ids = [self._make_id(ticker_yf, i + batch_start, t) for i, t in enumerate(texts)]
            self.store.add(texts, embeddings, metadatas, ids)
            total += len(batch)
            logger.info(f"  Embedded batch {batch_start}–{batch_start+len(batch)} ({total}/{len(chunks)})")

        logger.info(f"Ingestion complete for {ticker_yf}: {total} chunks stored")
        return total

    def ingest_all(self, refresh: bool = False) -> dict[str, int]:
        results: dict[str, int] = {}
        tickers = list(NIFTY50_TICKERS.keys())
        logger.info(f"Ingesting all {len(tickers)} Nifty 50 tickers...")
        for symbol in tickers:
            ticker_yf = f"{symbol}.NS"
            try:
                count = self.ingest(ticker_yf, refresh=refresh)
                results[ticker_yf] = count
            except Exception as exc:
                logger.error(f"Ingestion failed for {ticker_yf}: {exc}")
                results[ticker_yf] = -1
        logger.info(f"All ingestions done. Total docs in store: {self.store.count()}")
        return results

    @staticmethod
    def _make_id(ticker: str, index: int, text: str) -> str:
        """Deterministic ID so upsert is idempotent."""
        content_hash = hashlib.md5(text[:64].encode()).hexdigest()[:8]
        clean = ticker.replace(".", "_")
        return f"{clean}_{index:05d}_{content_hash}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Ingestion — build/refresh the vector knowledge base")
    parser.add_argument("--ticker", type=str, help="Single ticker, e.g. RELIANCE.NS")
    parser.add_argument("--all", action="store_true", help="Ingest all Nifty 50 tickers")
    parser.add_argument("--refresh", action="store_true", help="Delete existing data for ticker before re-ingesting")
    args = parser.parse_args()

    rag = RAGIngestion()

    if args.all:
        results = rag.ingest_all(refresh=args.refresh)
        for t, c in results.items():
            status = f"{c} chunks" if c >= 0 else "FAILED"
            print(f"  {t:25s} {status}")
    elif args.ticker:
        n = rag.ingest(args.ticker, refresh=args.refresh)
        print(f"\nIngested {n} chunks for {args.ticker}")
        print(f"Total in store: {rag.store.count()}")
    else:
        parser.print_help()
