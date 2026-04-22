"""
Processes unstructured text (news articles, earnings call transcripts)
into clean, deduplicated chunks ready for embedding.
"""

import re
from typing import Any

from src.utils.logger import get_logger
from src.utils.helpers import chunk_text
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Boilerplate patterns to strip from scraped text
_BOILERPLATE = re.compile(
    r"(subscribe now|click here|read more|advertisement|related articles|"
    r"follow us on|disclaimer:|©\s*\d{4}|all rights reserved)",
    re.IGNORECASE,
)


class TextProcessor:

    def __init__(self) -> None:
        self.chunk_size = settings.embedding_chunk_size
        self.overlap = settings.embedding_chunk_overlap

    def process_news(self, raw_data: dict[str, Any]) -> list[dict[str, str]]:
        """
        Process raw news JSON (from NewsFetcher) into embedding-ready chunks.
        Returns list of dicts with: text, ticker, section, source, published_at
        """
        ticker = raw_data.get("ticker", "UNKNOWN")
        articles = raw_data.get("articles", [])
        chunks: list[dict[str, str]] = []

        for article in articles:
            content = self._clean(
                " ".join(filter(None, [
                    article.get("title", ""),
                    article.get("description", ""),
                    article.get("content", ""),
                ]))
            )
            if len(content.split()) < 20:
                continue

            for chunk in chunk_text(content, self.chunk_size, self.overlap):
                chunks.append({
                    "text": chunk,
                    "ticker": ticker,
                    "section": "news",
                    "source": article.get("source", "unknown"),
                    "published_at": article.get("published_at", ""),
                })

        logger.info(f"Processed {len(articles)} news articles → {len(chunks)} chunks for {ticker}")
        return chunks

    def process_transcript(self, ticker: str, transcript_text: str, period: str = "") -> list[dict[str, str]]:
        """
        Process an earnings call transcript (raw string) into chunks.
        Transcripts are long — chunking preserves speaker turns where possible.
        """
        cleaned = self._clean(transcript_text)
        # Try to split on speaker turns first (e.g. "Operator:", "CEO:", "Analyst:")
        speaker_chunks = self._split_by_speaker(cleaned)

        all_chunks: list[dict[str, str]] = []
        for sc in speaker_chunks:
            for chunk in chunk_text(sc, self.chunk_size, self.overlap):
                all_chunks.append({
                    "text": chunk,
                    "ticker": ticker,
                    "section": "earnings_transcript",
                    "source": "earnings_call",
                    "published_at": period,
                })

        logger.info(f"Processed transcript → {len(all_chunks)} chunks for {ticker}")
        return all_chunks

    def process_annual_report(self, ticker: str, report_text: str, fiscal_year: str = "") -> list[dict[str, str]]:
        """Process text extracted from an annual report PDF."""
        cleaned = self._clean(report_text)
        all_chunks: list[dict[str, str]] = []
        for chunk in chunk_text(cleaned, self.chunk_size, self.overlap):
            all_chunks.append({
                "text": chunk,
                "ticker": ticker,
                "section": "annual_report",
                "source": "annual_report",
                "published_at": fiscal_year,
            })
        return all_chunks

    # ──────────────────────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
        text = re.sub(_BOILERPLATE, "", text)          # strip boilerplate
        text = re.sub(r"\s+", " ", text)               # collapse whitespace
        text = re.sub(r"[^\x00-\x7F]+", " ", text)    # strip non-ASCII
        return text.strip()

    def _split_by_speaker(self, text: str) -> list[str]:
        """Split transcript on lines that look like 'SPEAKER NAME:' markers."""
        parts = re.split(r"\n(?=[A-Z][A-Z\s\.]+:)", text)
        return [p.strip() for p in parts if p.strip()]
