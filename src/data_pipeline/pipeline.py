"""
Entry point for the full data ingestion pipeline.

Usage (CLI):
    python -m src.data_pipeline.pipeline --ticker RELIANCE.NS

Usage (Python):
    pipeline = DataPipeline()
    pipeline.run("RELIANCE.NS")
"""

import argparse
import json
import os
from typing import Any

from config.settings import get_settings
from config.nifty50_tickers import NIFTY50_TICKERS
from src.data_pipeline.fetchers import YFinanceFetcher, NewsFetcher, NSEFetcher, ScreenerFetcher
from src.data_pipeline.processors import FinancialProcessor, TextProcessor, ScreenerProcessor
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)
settings = get_settings()


class DataPipeline:
    """
    Orchestrates: fetch raw data → process → return structured chunks.
    The RAG ingestion layer (src/rag/ingestion.py) calls this and then embeds.
    """

    def __init__(self) -> None:
        self.yf_fetcher = YFinanceFetcher()
        self.news_fetcher = NewsFetcher()
        self.nse_fetcher = NSEFetcher()
        self.screener_fetcher = ScreenerFetcher()
        self.fin_processor = FinancialProcessor()
        self.text_processor = TextProcessor()
        self.screener_processor = ScreenerProcessor()
        ensure_dir(settings.processed_data_dir)

    def run(self, ticker_yf: str) -> list[dict[str, str]]:
        """
        Full pipeline for one ticker.
        ticker_yf: yfinance format, e.g. 'RELIANCE.NS'
        Returns list of processed text chunks ready for embedding.
        """
        nse_symbol = ticker_yf.replace(".NS", "").replace(".BO", "")
        company_name = NIFTY50_TICKERS.get(nse_symbol, nse_symbol)

        logger.info(f"=== Pipeline start: {ticker_yf} ({company_name}) ===")
        all_chunks: list[dict[str, str]] = []

        # 1. Financial fundamentals
        logger.info("Step 1/3: Fetching financial fundamentals")
        fin_raw = self.yf_fetcher.fetch(ticker_yf)
        all_chunks += self.fin_processor.process(fin_raw)

        # 2. News
        logger.info("Step 2/3: Fetching news")
        news_raw = self.news_fetcher.fetch(company_name, ticker_yf, days_back=60)
        all_chunks += self.text_processor.process_news({"ticker": ticker_yf, "articles": news_raw})

        # 3. NSE data (best-effort — NSE frequently blocks automated requests)
        try:
            self.nse_fetcher.fetch_corporate_actions(nse_symbol)
        except Exception:
            pass  # silently skip — NSE blocks are expected

        # 4. Screener.in — 10 years of financials (best-effort)
        logger.info("Step 4/4: Fetching 10-year data from Screener.in")
        try:
            screener_raw = self.screener_fetcher.fetch(nse_symbol)
            screener_chunks = self.screener_processor.process(screener_raw)
            all_chunks += screener_chunks
            logger.info(f"Screener.in added {len(screener_chunks)} chunks for {ticker_yf}")
        except Exception as exc:
            logger.warning(f"Screener fetch skipped for {ticker_yf}: {exc}")

        # Save processed chunks
        self._save_chunks(ticker_yf, all_chunks)
        logger.info(f"=== Pipeline complete: {len(all_chunks)} chunks for {ticker_yf} ===")
        return all_chunks

    def run_all(self) -> dict[str, int]:
        """Run pipeline for all Nifty 50 tickers. Returns {ticker: chunk_count}."""
        results: dict[str, int] = {}
        for symbol in NIFTY50_TICKERS:
            ticker_yf = f"{symbol}.NS"
            try:
                chunks = self.run(ticker_yf)
                results[ticker_yf] = len(chunks)
            except Exception as exc:
                logger.error(f"Pipeline failed for {ticker_yf}: {exc}")
                results[ticker_yf] = -1
        return results

    def _save_chunks(self, ticker: str, chunks: list[dict[str, str]]) -> None:
        clean = ticker.replace(".", "_")
        path = os.path.join(settings.processed_data_dir, f"{clean}_chunks.json")
        with open(path, "w") as f:
            json.dump(chunks, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EquityLens Data Pipeline")
    parser.add_argument("--ticker", type=str, help="NSE ticker in yfinance format, e.g. RELIANCE.NS")
    parser.add_argument("--all", action="store_true", help="Run for all Nifty 50 tickers")
    args = parser.parse_args()

    pipeline = DataPipeline()

    if args.all:
        results = pipeline.run_all()
        for t, count in results.items():
            status = "OK" if count > 0 else "FAILED"
            print(f"  {t:25s} {status:8s} {count} chunks")
    elif args.ticker:
        chunks = pipeline.run(args.ticker)
        print(f"\nGenerated {len(chunks)} chunks for {args.ticker}")
    else:
        parser.print_help()
