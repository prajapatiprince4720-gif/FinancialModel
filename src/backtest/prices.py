"""
Historical NSE prices for backtesting, downloaded once and cached to disk.

Two series per stock, both from yfinance:
  close      split/bonus-adjusted. Screener restates historical EPS onto today's
             share basis too, so this is the price to value a past view against.
  adj_close  also dividend-adjusted — the series total return is measured on.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.valuation.dcf_engine import _TICKER_OVERRIDES

BENCHMARK = "^NSEI"          # Nifty 50 price index (excludes dividends)
_STALE_DAYS = 7              # a "price on date" older than this is treated as missing


def yf_ticker(symbol: str) -> str:
    """yfinance ticker for an NSE symbol; renamed listings use their new ticker."""
    if symbol == BENCHMARK:
        return symbol
    return (_TICKER_OVERRIDES.get(symbol) or [f"{symbol}.NS"])[0]


class PriceHistory:
    """Daily close and adjusted close, one column per NSE symbol."""

    def __init__(self, close: pd.DataFrame, adj_close: pd.DataFrame):
        self.close     = close.sort_index()
        self.adj_close = adj_close.sort_index()

    @property
    def last_date(self) -> Optional[date]:
        return self.adj_close.index.max().date() if len(self.adj_close) else None

    @classmethod
    def load(cls, symbols: list[str], start: date, end: date,
             cache_dir: str | Path = "data/raw/prices", refresh: bool = False) -> "PriceHistory":
        """Cached history covering `symbols` (plus the benchmark) over [start, end]."""
        cache_dir = Path(cache_dir)
        c_path, a_path = cache_dir / "close.csv", cache_dir / "adj_close.csv"
        wanted = list(dict.fromkeys([*symbols, BENCHMARK]))

        if not refresh and c_path.exists() and a_path.exists():
            close = pd.read_csv(c_path, index_col=0, parse_dates=True)
            adj   = pd.read_csv(a_path, index_col=0, parse_dates=True)
            covered = (set(wanted) <= set(adj.columns)
                       and adj.index.min().date() <= start
                       and adj.index.max().date() >= min(end, date.today()) - timedelta(days=_STALE_DAYS))
            if covered:
                return cls(close, adj)

        close, adj = _download(wanted, start, end)
        cache_dir.mkdir(parents=True, exist_ok=True)
        close.to_csv(c_path)
        adj.to_csv(a_path)
        return cls(close, adj)

    # ── Lookups ───────────────────────────────────────────────────────────────

    def close_on(self, symbol: str, when: date) -> Optional[float]:
        """Split-adjusted close on `when`, or the last trading day before it."""
        return _on_or_before(self.close, symbol, when)

    def total_return(self, symbol: str, start: date, end: date) -> Optional[float]:
        """Dividend-inclusive return from `start` to `end`."""
        a = _on_or_before(self.adj_close, symbol, start)
        b = _on_or_before(self.adj_close, symbol, end)
        if a is None or b is None or a <= 0:
            return None
        return b / a - 1


def _on_or_before(frame: pd.DataFrame, symbol: str, when: date) -> Optional[float]:
    if symbol not in frame.columns:
        return None
    s = frame[symbol].loc[: pd.Timestamp(when)].dropna()
    if s.empty or (when - s.index[-1].date()).days > _STALE_DAYS:
        return None
    return float(s.iloc[-1])


def _download(symbols: list[str], start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    tickers = {yf_ticker(s): s for s in symbols}
    df = yf.download(list(tickers), start=start.isoformat(),
                     end=(end + timedelta(days=1)).isoformat(),
                     auto_adjust=False, progress=False, group_by="column")
    if df.empty:
        raise RuntimeError("yfinance returned no price history — check the network connection")
    close = df["Close"].rename(columns=tickers)
    adj   = df["Adj Close"].rename(columns=tickers)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    adj.index   = pd.to_datetime(adj.index).tz_localize(None)
    return close, adj
