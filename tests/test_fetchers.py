"""
Tests for data fetchers.
These are integration tests — they hit real APIs.
Run with: pytest tests/test_fetchers.py -v
"""

import pytest
from src.data_pipeline.fetchers.yfinance_fetcher import YFinanceFetcher
from src.data_pipeline.processors.financial_processor import FinancialProcessor
from src.data_pipeline.processors.text_processor import TextProcessor


class TestYFinanceFetcher:

    def test_fetch_returns_expected_keys(self, tmp_path):
        fetcher = YFinanceFetcher(raw_data_dir=str(tmp_path))
        data = fetcher.fetch("RELIANCE.NS")
        assert "ticker" in data
        assert "key_ratios" in data
        assert "income_statement" in data
        assert "balance_sheet" in data
        assert "cash_flow" in data

    def test_fetch_saves_file(self, tmp_path):
        import os
        fetcher = YFinanceFetcher(raw_data_dir=str(tmp_path))
        fetcher.fetch("TCS.NS")
        assert os.path.exists(os.path.join(str(tmp_path), "financials", "TCS_NS.json"))

    def test_safe_info_handles_error(self, tmp_path):
        fetcher = YFinanceFetcher(raw_data_dir=str(tmp_path))
        data = fetcher.fetch("INVALID_TICKER_XYZ.NS")
        # Should not raise, should return empty info gracefully
        assert isinstance(data, dict)


class TestFinancialProcessor:

    def test_process_returns_list_of_dicts(self, tmp_path):
        fetcher = YFinanceFetcher(raw_data_dir=str(tmp_path))
        raw = fetcher.fetch("INFY.NS")
        processor = FinancialProcessor()
        chunks = processor.process(raw)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "ticker" in chunk
            assert "section" in chunk

    def test_chunks_contain_ticker(self, tmp_path):
        fetcher = YFinanceFetcher(raw_data_dir=str(tmp_path))
        raw = fetcher.fetch("INFY.NS")
        processor = FinancialProcessor()
        chunks = processor.process(raw)
        for chunk in chunks:
            assert chunk["ticker"] == "INFY.NS"


class TestTextProcessor:

    def test_chunk_text_basic(self):
        processor = TextProcessor()
        text = " ".join(["word"] * 1000)
        chunks = processor._clean(text)
        assert isinstance(chunks, str)

    def test_process_news_empty_articles(self):
        processor = TextProcessor()
        result = processor.process_news({"ticker": "TEST.NS", "articles": []})
        assert result == []

    def test_process_news_filters_short_articles(self):
        processor = TextProcessor()
        result = processor.process_news({
            "ticker": "TEST.NS",
            "articles": [{"title": "Short", "description": "", "content": ""}]
        })
        assert result == []

    def test_process_transcript_creates_chunks(self):
        processor = TextProcessor()
        transcript = "MODERATOR: Welcome to the call. CEO: We had a great quarter with strong revenue growth. " * 50
        chunks = processor.process_transcript("TEST.NS", transcript, period="Q4FY24")
        assert len(chunks) > 0
        assert all(c["section"] == "earnings_transcript" for c in chunks)
