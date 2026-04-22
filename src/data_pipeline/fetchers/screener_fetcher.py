"""
Scrapes 10 years of financial data from Screener.in for NSE-listed companies.

Screener.in provides free access to:
  - P&L: 10 years of Sales, Operating Profit, Net Profit, EPS
  - Balance Sheet: 10 years of Assets, Liabilities, Equity, Debt
  - Cash Flow: 10 years of CFO, CFI, CFF, Free Cash Flow
  - Key Ratios: 10 years of ROCE, ROE, OPM%, Debt/Equity

Usage:
    fetcher = ScreenerFetcher()
    data = fetcher.fetch("RELIANCE")   # NSE symbol without .NS
"""

import json
import os
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir, safe_float

logger = get_logger(__name__)
settings = get_settings()

# Some NSE symbols map to a different Screener URL slug
# Discovered via: https://www.screener.in/api/company/search/?q=<name>&v=3&fts=1
SCREENER_SLUG_OVERRIDES: dict[str, str] = {
    "TATAMOTORS": "500570",     # Tata Motors parent uses BSE code; TMCV is CV subsidiary only
    "ZOMATO":     "ETERNAL",    # Zomato rebranded to Eternal Ltd
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Screener uses these section IDs in the HTML
SECTION_IDS = {
    "profit_loss": "profit-loss",
    "balance_sheet": "balance-sheet",
    "cash_flow": "cash-flow",
    "ratios": "ratios",
}


class ScreenerFetcher:
    """Scrapes Screener.in for 10-year financial history."""

    def __init__(self, raw_data_dir: str | None = None) -> None:
        self.raw_data_dir = raw_data_dir or settings.raw_data_dir
        ensure_dir(os.path.join(self.raw_data_dir, "screener"))

    def fetch(self, nse_symbol: str, rate_limit_delay: float = 1.5) -> dict[str, Any]:
        """
        Fetch 10-year financial data for an NSE symbol (e.g. 'RELIANCE').
        Tries consolidated view first, falls back to standalone.
        Respects SCREENER_SLUG_OVERRIDES for companies with non-standard URLs.
        Returns structured dict and saves to disk.
        """
        slug = SCREENER_SLUG_OVERRIDES.get(nse_symbol, nse_symbol)

        data = self._fetch_page(slug, consolidated=True)
        time.sleep(rate_limit_delay)
        if not self._has_data(data):
            logger.info(f"[Screener] No consolidated data for {nse_symbol} (slug={slug}), trying standalone")
            data = self._fetch_page(slug, consolidated=False)
            time.sleep(rate_limit_delay)

        if data and self._has_data(data):
            self._save(nse_symbol, data)
        elif data:
            logger.warning(f"[Screener] {nse_symbol}: page fetched but no financial tables found")
        return data or {}

    @staticmethod
    def _has_data(data: dict | None) -> bool:
        """Return True only if at least one section contains actual rows."""
        if not data:
            return False
        for key in ("profit_loss", "balance_sheet", "cash_flow", "ratios"):
            section = data.get(key, {})
            if section.get("rows"):
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_page(self, nse_symbol: str, consolidated: bool) -> dict[str, Any] | None:
        suffix = "consolidated/" if consolidated else ""
        url = f"https://www.screener.in/company/{nse_symbol}/{suffix}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"[Screener] HTTP error for {nse_symbol}: {exc}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Detect login wall (Screener gates some data behind login for exports,
        # but the HTML tables are public)
        if "Sign in" in resp.text and "profit-loss" not in resp.text:
            logger.warning(f"[Screener] Possible login wall for {nse_symbol}")
            return None

        data: dict[str, Any] = {
            "symbol": nse_symbol,
            "consolidated": consolidated,
            "url": url,
            "profit_loss": self._parse_section(soup, "profit-loss"),
            "balance_sheet": self._parse_section(soup, "balance-sheet"),
            "cash_flow": self._parse_section(soup, "cash-flow"),
            "ratios": self._parse_section(soup, "ratios"),
            "current_price": self._parse_current_price(soup),
            "peers": self._parse_peers(soup),
        }

        rows_found = sum(len(v) for v in data.values() if isinstance(v, dict))
        logger.info(
            f"[Screener] {nse_symbol} ({'consolidated' if consolidated else 'standalone'}): "
            f"{rows_found} data rows"
        )
        return data

    def _parse_section(self, soup: BeautifulSoup, section_id: str) -> dict[str, Any]:
        """
        Parse a financial section table into:
        {
          "years": ["Mar 2015", "Mar 2016", ...],
          "rows": {"Sales": [408299, 370832, ...], ...}
        }
        """
        section = soup.find("section", {"id": section_id})
        if not section:
            return {}

        table = section.find("table")
        if not table:
            return {}

        # Extract year headers — strip all internal whitespace for robustness
        thead = table.find("thead")
        years: list[str] = []
        if thead:
            header_cells = thead.find_all("th")
            for th in header_cells[1:]:
                raw = " ".join(th.get_text().split())  # collapse all whitespace
                if raw:
                    years.append(raw)

        # Extract data rows
        rows: dict[str, list[Any]] = {}
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                cells = tr.find_all("td")
                if not cells:
                    continue
                row_label = cells[0].get_text(strip=True).rstrip(" +")
                values: list[Any] = []
                for td in cells[1:]:
                    raw = td.get_text(strip=True).replace(",", "").replace("%", "")
                    values.append(safe_float(raw))
                if row_label and values:
                    rows[row_label] = values

        return {"years": years, "rows": rows}

    def _parse_current_price(self, soup: BeautifulSoup) -> float | None:
        try:
            price_el = soup.select_one(".number-profit, .current-price, [id='current-price']")
            if price_el:
                return safe_float(price_el.get_text(strip=True).replace(",", ""))
        except Exception:
            pass
        return None

    def _parse_peers(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """Parse the peer comparison table."""
        peers: list[dict[str, Any]] = []
        peer_section = soup.find("section", {"id": "peers"})
        if not peer_section:
            return peers
        table = peer_section.find("table")
        if not table:
            return peers

        headers: list[str] = []
        thead = table.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all("th")]

        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                cells = tr.find_all("td")
                if len(cells) >= 2:
                    row = {headers[i]: cells[i].get_text(strip=True) for i in range(min(len(headers), len(cells)))}
                    peers.append(row)

        return peers[:10]  # top 10 peers

    def _save(self, nse_symbol: str, data: dict[str, Any]) -> None:
        path = os.path.join(self.raw_data_dir, "screener", f"{nse_symbol}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"[Screener] Saved data for {nse_symbol} → {path}")
