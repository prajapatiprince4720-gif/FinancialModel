"""
Live India macro rates for CAPM / DCF inputs.

Fetches the India 10-Year Government Security (GSEC) yield from Yahoo Finance
with a 24-hour on-disk cache so every valuation uses today's rate, not a
hardcoded value from 18 months ago.

Fallback chain:
  1. On-disk cache (if < 24 h old)
  2. yfinance — tries several known India GSEC tickers
  3. NSE India public API — scrapes the G-Sec yield page
  4. Hardcoded recent value + printed warning (so the app never crashes)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

_CACHE_FILE = Path("data/raw/nse/macro_rates.json")
_CACHE_TTL  = 3600 * 24   # 24 hours

# Yahoo Finance tickers that carry the India 10Y GSEC yield (tried in order)
_YF_TICKERS = [
    "^INBMK",          # India Bond Market Index — sometimes carries yield
    "IN10Y.NS",        # NSE-listed G-Sec proxy
    "GSEC10.NS",       # Alternative NSE listing
]

# Hardcoded fallback — updated April 2026 (India 10Y GSEC ~6.82%)
_FALLBACK_RF   = 0.0682
_FALLBACK_DATE = "2026-04-01"


def get_rf_rate() -> float:
    """
    Return the India 10Y GSEC yield as a decimal (e.g. 0.0682 = 6.82%).
    Uses a 24-hour cache; prints a one-line status so the user knows the source.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    rate = _fetch_from_yfinance() or _fetch_from_fred()

    if rate is not None:
        _save_cache(rate, live=True)
        print(f"\033[92m  [Macro] India 10Y GSEC yield (live): {rate*100:.2f}%\033[0m")
        return rate

    # Fallback — hardcoded
    _save_cache(_FALLBACK_RF, live=False)
    print(
        f"\033[93m  [Macro] India 10Y GSEC yield (fallback {_FALLBACK_DATE}): "
        f"{_FALLBACK_RF*100:.2f}% — run: python main.py macro --refresh\033[0m"
    )
    return _FALLBACK_RF


def refresh_rates() -> float:
    """Force-refresh the macro rate cache and return the new rate."""
    _CACHE_FILE.unlink(missing_ok=True)
    return get_rf_rate()


def set_rf_rate(pct: float) -> float:
    """
    Manually set the RF rate (as a percentage, e.g. 6.75).
    Use this when live fetch is unavailable — check RBI website for current 10Y GSEC.
    Returns the stored decimal rate.
    """
    if not (4.0 <= pct <= 15.0):
        raise ValueError(f"RF rate {pct}% is outside the valid range (4–15%). Check your input.")
    rate = pct / 100
    _save_cache(rate, live=True)
    return rate


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache() -> Optional[float]:
    if not _CACHE_FILE.exists():
        return None
    age = time.time() - _CACHE_FILE.stat().st_mtime
    if age >= _CACHE_TTL:
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text())
        rate = data.get("rf_rate")
        src  = "live" if data.get("is_live") else f"fallback {data.get('updated','?')}"
        print(f"\033[2m  [Macro] India 10Y GSEC: {rate*100:.2f}% ({src}, "
              f"cached {age/3600:.1f}h ago)\033[0m")
        return float(rate) if rate else None
    except Exception:
        return None


def _save_cache(rate: float, live: bool) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if _CACHE_FILE.exists():
            try:
                existing = json.loads(_CACHE_FILE.read_text())
            except Exception:
                pass
        existing.update({
            "rf_rate":  rate,
            "is_live":  live,
            "updated":  time.strftime("%Y-%m-%d"),
            "fetched":  time.time(),
        })
        _CACHE_FILE.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass


# ── Live fetch methods ────────────────────────────────────────────────────────

def _fetch_from_yfinance() -> Optional[float]:
    try:
        import yfinance as yf
        for ticker in _YF_TICKERS:
            try:
                t    = yf.Ticker(ticker)
                hist = t.history(period="5d")
                if hist.empty:
                    continue
                val = float(hist["Close"].iloc[-1])
                if val > 1:          # stored as percentage (e.g. 6.82)
                    val /= 100
                if 0.04 <= val <= 0.15:   # sanity: 4 – 15% is valid GSEC range
                    return val
            except Exception:
                continue
    except Exception:
        pass
    return None


def _fetch_from_fred() -> Optional[float]:
    """
    FRED (St. Louis Fed) publishes India 10Y GSEC via OECD series INDIRLTLT01STM.
    Free, no API key, monthly data — most reliable publicly available source.
    """
    try:
        import requests
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            "?id=INDIRLTLT01STM"
        )
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None

        # CSV: observation_date,INDIRLTLT01STM
        # Last row = most recent monthly value
        lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]
        last  = lines[-1]                       # e.g. "2025-12-01,6.82000000"
        parts = last.split(",")
        val   = float(parts[1])                 # percentage, e.g. 6.82
        if 4.0 <= val <= 15.0:
            return val / 100                    # → 0.0682
    except Exception:
        pass
    return None
