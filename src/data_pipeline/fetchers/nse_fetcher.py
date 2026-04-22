"""
Fetches NSE-specific data:
  - Corporate actions (dividends, splits, bonuses)
  - FII/DII activity (monthly)
  - 52-week high/low from NSE directly

NSE has a public JSON API (no key required).
Rate-limit: ~2 req/s — always add a small sleep.
"""

import json
import os
import time
from typing import Any

import requests

from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)
settings = get_settings()

NSE_BASE = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}


class NSEFetcher:
    """
    Thin wrapper around the public NSE JSON API.

    NOTE: NSE occasionally blocks automated access. If you hit 403s:
      1. Use a residential proxy
      2. Add longer sleeps
      3. Fall back to yfinance data (which also sources from NSE)
    """

    def __init__(self, raw_data_dir: str | None = None) -> None:
        self.raw_data_dir = raw_data_dir or settings.raw_data_dir
        ensure_dir(os.path.join(self.raw_data_dir, "nse"))
        self._session = self._init_session()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_quote(self, symbol: str) -> dict[str, Any]:
        """Fetch real-time quote and basic info for an NSE symbol (e.g. 'RELIANCE')."""
        data = self._get(f"/api/quote-equity?symbol={symbol}")
        self._save(symbol, "quote", data)
        return data

    def fetch_corporate_actions(self, symbol: str) -> dict[str, Any]:
        """Fetch dividends, splits, and bonus history."""
        data = self._get(f"/api/corporates-corporateActions?index=equities&symbol={symbol}")
        self._save(symbol, "corporate_actions", data)
        return data

    def fetch_shareholding(self, symbol: str) -> dict[str, Any]:
        """Fetch latest shareholding pattern (promoters, FII, DII, retail)."""
        data = self._get(f"/api/corporate-shareholding-patterns?symbol={symbol}&industry=&isinCode=")
        self._save(symbol, "shareholding", data)
        return data

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _init_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        # Warm up session — NSE requires a cookie from the homepage
        try:
            session.get(NSE_BASE, timeout=10)
            time.sleep(1)
        except Exception as exc:
            logger.warning(f"NSE session warmup failed: {exc}")
        return session

    def _get(self, path: str) -> dict[str, Any]:
        url = NSE_BASE + path
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            time.sleep(0.5)
            return resp.json()
        except Exception as exc:
            logger.error(f"NSE GET {path} failed: {exc}")
            return {"error": str(exc)}

    def _save(self, symbol: str, data_type: str, data: dict[str, Any]) -> None:
        path = os.path.join(self.raw_data_dir, "nse", f"{symbol}_{data_type}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
