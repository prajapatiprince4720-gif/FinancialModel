"""
Shared report data preparation.

The PDF renderer previously held this extraction inline. The Word renderer needs
exactly the same numbers, and two copies of "which Screener row is revenue for a
bank" would drift apart on the first schema change — so it lives here once and
both renderers consume it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ReportFinancials:
    """Five years of headline P&L / cash-flow lines for one company."""
    years:    list[str]      = field(default_factory=list)
    sales:    list[Any]      = field(default_factory=list)
    net_profit: list[Any]    = field(default_factory=list)
    eps:      list[Any]      = field(default_factory=list)
    opm:      list[Any]      = field(default_factory=list)
    cfo:      list[Any]      = field(default_factory=list)
    roce:     list[Any]      = field(default_factory=list)
    fcf:      list[Any]      = field(default_factory=list)
    is_bank:  bool           = False
    # Label for the margin row, which differs for lenders.
    opm_label: str           = "OPM %"
    roce_label: str          = "ROCE %"
    raw:      dict           = field(default_factory=dict)

    @property
    def has_data(self) -> bool:
        return bool(self.years)


def load_report_financials(
    symbol: str,
    cache_dir: str = "data/raw/screener",
) -> ReportFinancials:
    """
    Load and normalise five years of financials for `symbol` from the Screener cache.

    Banks report "Revenue" rather than "Sales" and a "Financing Margin %" instead of
    "OPM %", and they have no meaningful ROCE — so the keys are chosen per company
    type rather than assumed.
    """
    path = Path(cache_dir) / f"{symbol}.json"
    raw: dict = {}
    if path.exists():
        try:
            with open(path) as f:
                raw = json.load(f)
        except Exception:
            raw = {}

    pl       = raw.get("profit_loss", {}) or {}
    pl_rows  = pl.get("rows", {}) or {}
    pl_years = pl.get("years", []) or []
    cf_rows  = (raw.get("cash_flow", {}) or {}).get("rows", {}) or {}
    rat_rows = (raw.get("ratios", {}) or {}).get("rows", {}) or {}

    # Drop a trailing TTM column so the table shows comparable full years.
    n   = len(pl_years)
    cut = n - 1 if pl_years and "TTM" in str(pl_years[-1]) else n
    idx = list(range(max(0, cut - 5), cut))

    def row(rows: dict, key: Optional[str]) -> list[Any]:
        if not key:
            return [None] * len(idx)
        vals = rows.get(key, []) or []
        return [vals[i] if i < len(vals) else None for i in idx]

    is_bank   = bool(pl_rows.get("Revenue")) and not pl_rows.get("Sales")
    sales_key = "Revenue" if is_bank else "Sales"
    opm_key   = "Financing Margin %" if is_bank else "OPM %"
    roce_key  = "ROE %" if is_bank else "ROCE %"

    return ReportFinancials(
        years=[str(y) for y in pl_years[max(0, cut - 5):cut]],
        sales=row(pl_rows, sales_key),
        net_profit=row(pl_rows, "Net Profit"),
        eps=row(pl_rows, "EPS in Rs"),
        opm=row(pl_rows, opm_key),
        cfo=row(cf_rows, "Cash from Operating Activity"),
        roce=row(rat_rows, roce_key),
        fcf=row(cf_rows, "Free Cash Flow"),
        is_bank=is_bank,
        opm_label=opm_key,
        roce_label=roce_key,
        raw=raw,
    )


def fmt_num(value: Any, decimals: int = 0, dash: str = "—") -> str:
    """Format a number for a report table, tolerating None and junk."""
    if value is None or value == "":
        return dash
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{num:,.{decimals}f}"


def fmt_pct(value: Any, decimals: int = 1, dash: str = "—") -> str:
    if value is None or value == "":
        return dash
    try:
        return f"{float(value):.{decimals}f}%"
    except (TypeError, ValueError):
        return str(value)
