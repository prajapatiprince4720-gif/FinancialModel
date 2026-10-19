"""
Live India macro rates for CAPM / DCF inputs.

The India 10-Year Government Security (GSEC) yield is the risk-free rate in
every CAPM cost-of-equity and therefore in every DCF on this platform. It is
computed from the live NSE G-Sec order book rather than hardcoded.

Source chain (first success wins):
  1. On-disk cache (if < 24 h old)
  2. NSE India public API — live G-Sec prices, YTM solved per bond
  3. yfinance — legacy India GSEC tickers (mostly delisted; kept as a long shot)
  4. FRED / OECD series INDIRLTLT01STM (monthly; unreachable on some networks)
  5. Hardcoded recent value + a loud warning (so the app never crashes)

NSE publishes coupon and maturity inside the instrument symbol (`754GS2036`
= 7.54% Government Stock maturing 2036) plus a live traded price, but no YTM
field. We therefore solve YTM per bond from price, coupon and years-to-maturity,
keep only actively traded central-government bonds maturing near the 10-year
point, and take a volume-weighted average. That mirrors how the street quotes
"the 10-year" — the most liquid benchmark bond near the 10Y tenor.

Every consumer should prefer `get_rf_context()` over `get_rf_rate()` so that
reports can disclose whether the rate is live, cached, manual or hardcoded.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path
from typing import NamedTuple, Optional

_CACHE_FILE = Path("data/raw/nse/macro_rates.json")
_CACHE_TTL  = 3600 * 24   # 24 hours

# Yahoo Finance tickers that once carried the India 10Y GSEC yield (tried in order).
# All three are delisted as of 2026 — retained only because they cost one call.
_YF_TICKERS = [
    "^INBMK",          # India Bond Market Index
    "IN10Y.NS",        # NSE-listed G-Sec proxy
    "GSEC10.NS",       # Alternative NSE listing
]

# Hardcoded last-resort fallback — India 10Y GSEC ~6.82% as of April 2026.
_FALLBACK_RF   = 0.0682
_FALLBACK_DATE = "2026-04-01"

# Sanity band for any India 10Y GSEC yield. Anything outside this is a parse bug.
_RF_MIN, _RF_MAX = 0.04, 0.15

# ── NSE live G-Sec parameters ─────────────────────────────────────────────────
_NSE_HOME   = "https://www.nseindia.com"
_NSE_GSEC   = "https://www.nseindia.com/api/liveBonds-traded-on-cm?type=gsec"
_NSE_UA     = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Central-government stock symbols: coupon (x100) + series + maturity year.
#   667GS2050  → 6.67% GS 2050
#   679GS2034A → 6.79% GS 2034 (re-issue suffix)
#   1018GS2026 → 10.18% GS 2026
_GSEC_SYMBOL = re.compile(r"^(\d{3,4})(G[SRBC])(\d{2,4})([A-Z]?)$")
_TARGET_TENOR = 10.0     # years
_TENOR_WINDOW = 1.5      # accept bonds maturing within +/- this many years
_COUPON_FREQ  = 2        # India G-Secs pay semi-annual coupons


class RFContext(NamedTuple):
    """The risk-free rate plus where it came from, so reports can disclose it."""
    rate:   float          # decimal, e.g. 0.0685
    source: str            # "nse_live" | "yfinance" | "fred" | "manual" | "fallback"
    as_of:  str            # ISO date the rate was established
    is_live: bool          # False means a hardcoded or stale number is in play
    detail: str = ""       # human-readable provenance, e.g. benchmark bond used

    @property
    def label(self) -> str:
        """One-line provenance string for PDF/DOCX/web footers."""
        if self.source == "nse_live":
            return f"{self.rate*100:.2f}% — live NSE G-Sec book, {self.as_of}"
        if self.source == "manual":
            return f"{self.rate*100:.2f}% — manually set {self.as_of}"
        if self.source == "fallback":
            return f"{self.rate*100:.2f}% — HARDCODED fallback dated {self.as_of} (not live)"
        return f"{self.rate*100:.2f}% — {self.source}, {self.as_of}"


def get_rf_context(quiet: bool = False) -> RFContext:
    """
    Return the India 10Y GSEC yield with full provenance.

    Uses a 24-hour on-disk cache. Never raises and never blocks a valuation —
    worst case it returns the hardcoded fallback flagged is_live=False.
    """
    cached = _load_cache(quiet=quiet)
    if cached is not None:
        return cached

    for fetch, source in (
        (_fetch_from_nse,      "nse_live"),
        (_fetch_from_yfinance, "yfinance"),
        (_fetch_from_fred,     "fred"),
    ):
        try:
            result = fetch()
        except Exception:
            result = None
        if result is None:
            continue
        rate, detail = result if isinstance(result, tuple) else (result, "")
        if not _sane(rate):
            continue
        ctx = RFContext(
            rate=rate, source=source, as_of=date.today().isoformat(),
            is_live=True, detail=detail,
        )
        _save_cache(ctx)
        if not quiet:
            print(f"\033[92m  [Macro] India 10Y GSEC yield (live via {source}): "
                  f"{rate*100:.2f}%\033[0m"
                  + (f"\033[2m  {detail}\033[0m" if detail else ""))
        return ctx

    # Every live source failed — fall back loudly.
    ctx = RFContext(
        rate=_FALLBACK_RF, source="fallback", as_of=_FALLBACK_DATE,
        is_live=False, detail="all live sources unreachable",
    )
    _save_cache(ctx)
    if not quiet:
        print(
            f"\033[93m  [Macro] India 10Y GSEC yield NOT live — using hardcoded "
            f"{_FALLBACK_RF*100:.2f}% dated {_FALLBACK_DATE}. "
            f"Valuations will be stale. Fix with: "
            f"python main.py macro --set <pct>\033[0m"
        )
    return ctx


def get_rf_rate(quiet: bool = False) -> float:
    """Return just the India 10Y GSEC yield as a decimal (e.g. 0.0685 = 6.85%)."""
    return get_rf_context(quiet=quiet).rate


def refresh_rates() -> float:
    """Force-refresh the macro rate cache and return the new rate."""
    _CACHE_FILE.unlink(missing_ok=True)
    return get_rf_context().rate


def set_rf_rate(pct: float) -> float:
    """
    Manually set the RF rate (as a percentage, e.g. 6.75).
    Use when every live source is blocked — check the RBI or NSE bond page.
    Returns the stored decimal rate.
    """
    if not (_RF_MIN * 100 <= pct <= _RF_MAX * 100):
        raise ValueError(
            f"RF rate {pct}% is outside the valid range "
            f"({_RF_MIN*100:.0f}–{_RF_MAX*100:.0f}%). Check your input."
        )
    ctx = RFContext(
        rate=pct / 100, source="manual", as_of=date.today().isoformat(),
        is_live=True, detail="operator override",
    )
    _save_cache(ctx)
    return ctx.rate


def _sane(rate: Optional[float]) -> bool:
    return rate is not None and _RF_MIN <= rate <= _RF_MAX


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache(quiet: bool = False) -> Optional[RFContext]:
    if not _CACHE_FILE.exists():
        return None
    age = time.time() - _CACHE_FILE.stat().st_mtime
    if age >= _CACHE_TTL:
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text())
        rate = float(data.get("rf_rate") or 0)
    except Exception:
        return None
    if not _sane(rate):
        return None

    source = data.get("source") or ("yfinance" if data.get("is_live") else "fallback")
    ctx = RFContext(
        rate=rate,
        source=source,
        as_of=data.get("updated", "?"),
        is_live=bool(data.get("is_live")),
        detail=data.get("detail", ""),
    )
    if not quiet:
        colour = "\033[2m" if ctx.is_live else "\033[93m"
        print(f"{colour}  [Macro] India 10Y GSEC: {ctx.label} "
              f"(cached {age/3600:.1f}h ago)\033[0m")
    return ctx


def _save_cache(ctx: RFContext) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if _CACHE_FILE.exists():
            try:
                existing = json.loads(_CACHE_FILE.read_text())
            except Exception:
                pass
        existing.update({
            "rf_rate": ctx.rate,
            "is_live": ctx.is_live,
            "source":  ctx.source,
            "detail":  ctx.detail,
            "updated": ctx.as_of,
            "fetched": time.time(),
        })
        _CACHE_FILE.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass


# ── Live fetch methods ────────────────────────────────────────────────────────

def _bond_ytm(price: float, coupon_pct: float, years: float,
              freq: int = _COUPON_FREQ, face: float = 100.0) -> Optional[float]:
    """
    Solve yield-to-maturity by bisection on the standard bond price identity.

    price = SUM(coupon/freq / (1+y/freq)^k) + face / (1+y/freq)^n

    Bisection (not Newton) because the function is monotonic in y and we want a
    guaranteed answer without derivative bookkeeping or divergence risk.
    """
    n = int(round(years * freq))
    if price <= 0 or n < 1:
        return None
    c = face * coupon_pct / 100.0 / freq

    def pv(y: float) -> float:
        r = y / freq
        return sum(c / (1 + r) ** k for k in range(1, n + 1)) + face / (1 + r) ** n

    lo, hi = 0.0001, 0.50
    if pv(hi) > price:      # yield above 50% — not a real G-Sec quote
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        if pv(mid) > price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _fetch_from_nse() -> Optional[tuple[float, str]]:
    """
    Compute the India 10Y GSEC yield from NSE's live G-Sec order book.

    Returns (rate, detail) or None. Keeps only actively traded central-government
    bonds maturing near the 10-year point and volume-weights their YTMs, so an
    illiquid off-the-run bond with a stale print cannot move the benchmark.
    """
    import requests

    session = requests.Session()
    session.headers.update({
        "User-Agent": _NSE_UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"{_NSE_HOME}/market-data/bonds-traded-in-capital-market",
    })
    # NSE requires a cookie from the home page before its API will answer.
    session.get(_NSE_HOME, timeout=15)
    resp = session.get(_NSE_GSEC, timeout=30)
    if resp.status_code != 200:
        return None
    rows = resp.json().get("data") or []
    if not rows:
        return None

    today = date.today()
    candidates: list[tuple[float, float, float, str]] = []   # (dist, ytm, volume, symbol)

    for row in rows:
        # Central government only — state development loans (SG series) trade at a spread.
        if (row.get("companyName") or "").strip() != "Government of India":
            continue
        match = _GSEC_SYMBOL.match((row.get("symbol") or "").strip())
        if not match:
            continue

        price = row.get("lastPrice") or row.get("previousClose") or 0
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue

        coupon = int(match.group(1)) / 100.0
        year   = int(match.group(3))
        year   = year + 2000 if year < 100 else year
        # NSE omits the exact maturity date; mid-year is within ~6 months, which
        # moves a 10-year YTM by only a few basis points.
        years  = (date(year, 6, 30) - today).days / 365.25
        if years <= 1.0:
            continue

        distance = abs(years - _TARGET_TENOR)
        if distance > _TENOR_WINDOW:
            continue

        ytm = _bond_ytm(price, coupon, years)
        if not _sane(ytm):
            continue

        try:
            volume = float(row.get("totalTradedVolume") or 0)
        except (TypeError, ValueError):
            volume = 0.0
        candidates.append((distance, ytm, volume, match.group(0)))

    if not candidates:
        return None

    # Prefer bonds that actually traded today; an untraded bond's "lastPrice"
    # can be weeks old and produces wild YTMs.
    traded = [c for c in candidates if c[2] > 0]
    pool   = traded or candidates

    total_volume = sum(c[2] for c in pool)
    if total_volume > 0:
        rate = sum(c[1] * c[2] for c in pool) / total_volume
        weighting = "volume-weighted"
    else:
        ordered = sorted(c[1] for c in pool)
        mid     = len(ordered) // 2
        rate    = (ordered[mid] if len(ordered) % 2
                   else (ordered[mid - 1] + ordered[mid]) / 2)
        weighting = "median (no volume reported)"

    if not _sane(rate):
        return None

    benchmark = max(pool, key=lambda c: c[2])
    detail = (
        f"{weighting} across {len(pool)} GoI bond(s) near {_TARGET_TENOR:.0f}Y; "
        f"benchmark {benchmark[3]} @ {benchmark[1]*100:.2f}%"
    )
    return rate, detail


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
                if val > 1:          # quoted as a percentage (e.g. 6.82)
                    val /= 100
                if _sane(val):
                    return val
            except Exception:
                continue
    except Exception:
        pass
    return None


def _fetch_from_fred() -> Optional[float]:
    """
    FRED (St. Louis Fed) publishes India 10Y GSEC via OECD series INDIRLTLT01STM.
    Free and keyless, but monthly, and the host is unreachable from some networks
    (including behind certain ISPs), so it sits late in the chain.
    """
    try:
        import requests
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDIRLTLT01STM"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None

        # CSV: observation_date,INDIRLTLT01STM — last row is the most recent month.
        lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]
        val   = float(lines[-1].split(",")[1])        # percentage, e.g. 6.82
        if _RF_MIN * 100 <= val <= _RF_MAX * 100:
            return val / 100
    except Exception:
        pass
    return None
