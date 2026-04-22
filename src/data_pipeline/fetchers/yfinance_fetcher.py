"""
Fetches fundamental financial data from Yahoo Finance via yfinance.

Data available for free:
  - Income statement (annual + quarterly)
  - Balance sheet
  - Cash flow statement
  - Key ratios (PE, PB, EPS, dividend yield, etc.)
  - Historical price data
  - Company info and description
"""

import json
import os
from datetime import datetime
from typing import Any

import yfinance as yf

from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir, safe_float

logger = get_logger(__name__)
settings = get_settings()


class YFinanceFetcher:
    """Fetches and persists financial fundamentals for a given NSE ticker."""

    def __init__(self, raw_data_dir: str | None = None) -> None:
        self.raw_data_dir = raw_data_dir or settings.raw_data_dir
        ensure_dir(os.path.join(self.raw_data_dir, "financials"))

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fetch(self, ticker: str) -> dict[str, Any]:
        """
        Fetch all available fundamental data for `ticker` (e.g. 'RELIANCE.NS').
        Returns a structured dict and saves it to disk.
        """
        logger.info(f"Fetching fundamentals for {ticker}")
        stock = yf.Ticker(ticker)

        data: dict[str, Any] = {
            "ticker": ticker,
            "fetched_at": datetime.utcnow().isoformat(),
            "info": self._safe_info(stock),
            "income_statement": self._df_to_dict(stock.financials),
            "balance_sheet": self._df_to_dict(stock.balance_sheet),
            "cash_flow": self._df_to_dict(stock.cashflow),
            "quarterly_income": self._df_to_dict(stock.quarterly_financials),
            "quarterly_balance": self._df_to_dict(stock.quarterly_balance_sheet),
            "key_ratios": self._extract_ratios(stock),
        }

        self._save(ticker, data)
        logger.info(f"Saved fundamentals for {ticker}")
        return data

    def fetch_price_history(
        self, ticker: str, period: str = "5y", interval: str = "1d"
    ) -> dict[str, Any]:
        """Fetch OHLCV history. period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"""
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "data": hist.reset_index().to_dict(orient="records"),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _safe_info(self, stock: yf.Ticker) -> dict[str, Any]:
        try:
            info = stock.info
            # Keep only serialisable scalar values
            return {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool, type(None)))}
        except Exception as exc:
            logger.warning(f"Could not fetch info: {exc}")
            return {}

    def _df_to_dict(self, df) -> dict[str, Any]:
        """Convert a pandas DataFrame to a JSON-serialisable dict."""
        try:
            if df is None or df.empty:
                return {}
            return {
                str(col): {str(idx): self._serialize(val) for idx, val in df[col].items()}
                for col in df.columns
            }
        except Exception as exc:
            logger.warning(f"DataFrame conversion failed: {exc}")
            return {}

    def _serialize(self, val: Any) -> Any:
        """Convert numpy/pandas scalars to Python natives."""
        try:
            import numpy as np
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, (np.floating,)):
                return float(val)
            if isinstance(val, float) and (val != val):  # NaN check
                return None
        except ImportError:
            pass
        return val

    def _extract_ratios(self, stock: yf.Ticker) -> dict[str, float | None]:
        info = self._safe_info(stock)
        return {
            "pe_ratio": safe_float(info.get("trailingPE")) or None,
            "forward_pe": safe_float(info.get("forwardPE")) or None,
            "pb_ratio": safe_float(info.get("priceToBook")) or None,
            "ps_ratio": safe_float(info.get("priceToSalesTrailing12Months")) or None,
            "ev_ebitda": safe_float(info.get("enterpriseToEbitda")) or None,
            "debt_to_equity": safe_float(info.get("debtToEquity")) or None,
            "current_ratio": safe_float(info.get("currentRatio")) or None,
            "roe": safe_float(info.get("returnOnEquity")) or None,
            "roa": safe_float(info.get("returnOnAssets")) or None,
            "profit_margin": safe_float(info.get("profitMargins")) or None,
            "operating_margin": safe_float(info.get("operatingMargins")) or None,
            "revenue_growth": safe_float(info.get("revenueGrowth")) or None,
            "earnings_growth": safe_float(info.get("earningsGrowth")) or None,
            "dividend_yield": safe_float(info.get("dividendYield")) or None,
            "payout_ratio": safe_float(info.get("payoutRatio")) or None,
            "beta": safe_float(info.get("beta")) or None,
            "market_cap": safe_float(info.get("marketCap")) or None,
            "52w_high": safe_float(info.get("fiftyTwoWeekHigh")) or None,
            "52w_low": safe_float(info.get("fiftyTwoWeekLow")) or None,
        }

    def _save(self, ticker: str, data: dict[str, Any]) -> None:
        clean_ticker = ticker.replace(".", "_")
        path = os.path.join(self.raw_data_dir, "financials", f"{clean_ticker}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
