"""
One loader for company fundamentals.

Before this module, the DCF engine (three times), the scorer and the report
builder each re-implemented "open the Screener JSON, drop the TTM column, pull a
row" — and they disagreed in small ways. Two concrete defects that shared logic
fixes:

* TTM was dropped from every statement using the P&L's column count. A statement
  with a different layout (ZOMATO's balance sheet ends in a half-year "Sep 2025"
  column) was trimmed by coincidence, not by rule. Here each statement is trimmed
  on its own: TTM and any trailing interim column are removed.
* `[v for v in row if v is not None]` silently shifts years. If 2019 is missing
  from Net Profit but not from EPS, element i of one list no longer matches
  element i of the other. `aligned()` keeps the gaps so year i is year i.

Data can be loaded as it is now (the Screener cache) or as it was on a past date
(the snapshot store), which is what a backtest needs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

SCREENER_DIR = Path("data/raw/screener")

# Lenders: balance-sheet businesses valued on book value / ROE, not cash flow.
BANKS:  frozenset[str] = frozenset({
    "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "INDUSINDBK",
})
NBFCS:  frozenset[str] = frozenset({"BAJFINANCE", "SHRIRAMFIN", "BAJAJFINSV"})
LENDERS: frozenset[str] = BANKS | NBFCS
INSURERS: frozenset[str] = frozenset({"HDFCLIFE", "SBILIFE"})

STATEMENTS = ("profit_loss", "balance_sheet", "cash_flow", "ratios")

# Corporate actions that break the link between historical fundamentals and the
# traded security. Surfaced as caveats wherever the company is reported on.
CORPORATE_ACTIONS: dict[str, str] = {
    "TATAMOTORS": (
        "Tata Motors demerged in 2025 into Tata Motors Passenger Vehicles (TMPV) and "
        "Tata Motors Commercial Vehicles (TMCV). Fundamentals here describe the combined "
        "pre-demerger company and do not match either listed stock; price-based figures "
        "are withheld."
    ),
    "ZOMATO": "Zomato Ltd was renamed Eternal Ltd (NSE: ETERNAL) in 2025.",
}

_MONTH_YEAR = re.compile(r"^([A-Za-z]{3})\s+(\d{4})$")


def _full_year_cut(years: list[Any]) -> int:
    """
    Number of leading columns that are full fiscal years.

    Drops a trailing TTM column and any trailing interim column whose month
    differs from the statement's usual fiscal year-end month.
    """
    labels = [str(y) for y in years]
    cut = len(labels)
    if cut and "TTM" in labels[cut - 1]:
        cut -= 1
    months = [m.group(1) for m in (_MONTH_YEAR.match(l) for l in labels[:cut]) if m]
    if months:
        fy_month = max(set(months), key=months.count)
        while cut:
            m = _MONTH_YEAR.match(labels[cut - 1])
            if m and m.group(1) != fy_month:
                cut -= 1
            else:
                break
    return cut


def fiscal_year_available_date(label: str) -> Optional[date]:
    """
    Earliest date a fiscal year's annual numbers were public.

    SEBI LODR gives listed companies 60 days after year-end to publish audited
    annual results. A backtest that uses FY figures before this date is looking
    into the future. "Mar 2024" → 2024-05-31.
    """
    m = _MONTH_YEAR.match(str(label))
    if not m:
        return None
    month = datetime.strptime(m.group(1), "%b").month
    year  = int(m.group(2))
    # Last day of the month two months after year-end.
    end_month = month + 2
    end_year  = year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    nxt = date(end_year + (end_month // 12), end_month % 12 + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)


@dataclass
class CompanyData:
    """Fundamentals for one company, with consistent full-year trimming."""
    symbol: str
    raw:    dict[str, Any]
    source: str = "cache"              # "cache" or "snapshot"
    _cuts:  dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for sec in STATEMENTS:
            years = (self.raw.get(sec) or {}).get("years", []) or []
            self._cuts[sec] = _full_year_cut(years)

    # ── Classification ────────────────────────────────────────────────────────

    @property
    def is_lender(self) -> bool:
        return self.symbol in LENDERS

    @property
    def is_insurer(self) -> bool:
        return self.symbol in INSURERS

    @property
    def current_price(self) -> Optional[float]:
        return self.raw.get("current_price")

    @property
    def is_historical(self) -> bool:
        """True for a view built by as_of(); its price must never be fetched live."""
        return self.source.startswith("as_of:")

    # ── Series access ─────────────────────────────────────────────────────────

    def years(self, section: str = "profit_loss") -> list[str]:
        """Full fiscal-year labels for a statement (TTM/interim removed)."""
        ys = (self.raw.get(section) or {}).get("years", []) or []
        return [str(y) for y in ys[: self._cuts.get(section, len(ys))]]

    def aligned(self, section: str, *keys: str) -> list[Optional[float]]:
        """
        Full-year values for the first row key present, gaps kept as None so that
        index i is the same fiscal year across every row of the statement.
        """
        rows = (self.raw.get(section) or {}).get("rows", {}) or {}
        cut  = self._cuts.get(section, 0)
        for key in keys:
            vals = rows.get(key)
            if vals:
                return list(vals[:cut])
        return []

    def series(self, section: str, *keys: str) -> list[float]:
        """Full-year values with gaps removed (legacy behaviour, for averages/tails)."""
        return [v for v in self.aligned(section, *keys) if v is not None]

    def as_of(self, when: date, price: Optional[float] = None) -> "CompanyData":
        """
        The statements as an investor could have seen them on `when`.

        Each statement keeps only the full fiscal years whose results were public
        by then (fiscal_year_available_date ≤ when); TTM and interim columns go.
        `price` replaces current_price so nothing downstream fetches today's
        price into a historical valuation.

        Limitation: Screener shows figures as later restated, not as first
        reported. Restatements are usually small, but this is not a true
        point-in-time database for dates before snapshots began.
        """
        raw: dict[str, Any] = {k: v for k, v in self.raw.items() if k not in STATEMENTS}
        raw["current_price"] = price
        for sec in STATEMENTS:
            src_sec = self.raw.get(sec) or {}
            years   = (src_sec.get("years") or [])[: self._cuts.get(sec, 0)]
            keep = 0
            for label in years:
                avail = fiscal_year_available_date(label)
                if avail is None or avail > when:
                    break
                keep += 1
            raw[sec] = {
                "years": list(years[:keep]),
                "rows":  {k: list(v[:keep]) for k, v in (src_sec.get("rows") or {}).items()},
            }
        return CompanyData(self.symbol, raw, source=f"as_of:{when.isoformat()}")

    def ttm(self, section: str, key: str) -> Optional[float]:
        """The trailing-twelve-month value, if the statement has a TTM column."""
        sec   = self.raw.get(section) or {}
        years = sec.get("years", []) or []
        if not years or "TTM" not in str(years[-1]):
            return None
        vals = (sec.get("rows", {}) or {}).get(key) or []
        return vals[len(years) - 1] if len(vals) >= len(years) else None


def load_company(
    symbol: str,
    as_of: Optional[datetime | str] = None,
    cache_dir: str | Path = SCREENER_DIR,
) -> Optional[CompanyData]:
    """
    Load fundamentals for `symbol`.

    as_of=None reads the current Screener cache. A date reads the newest snapshot
    observed on or before it — data we actually had then, not data restated since.
    """
    if as_of is not None:
        from src.data.snapshots import get_snapshot_store
        raw = get_snapshot_store().as_of(symbol, as_of)
        return CompanyData(symbol, raw, source="snapshot") if raw else None

    path = Path(cache_dir) / f"{symbol}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return CompanyData(symbol, raw or {}, source="cache")
