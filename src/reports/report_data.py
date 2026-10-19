"""
Shared report data preparation.

The PDF renderer previously held this extraction inline. The Word renderer needs
exactly the same numbers, and two copies of "which Screener row is revenue for a
bank" would drift apart on the first schema change — so it lives here once and
both renderers consume it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.data.company import CompanyData, load_company


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
    cd  = load_company(symbol, cache_dir=cache_dir) or CompanyData(symbol, {})
    raw = cd.raw
    pl_rows = (raw.get("profit_loss", {}) or {}).get("rows", {}) or {}

    # Full fiscal years only (TTM/interim dropped per statement), last five.
    years = cd.years("profit_loss")[-5:]
    width = len(years)

    def row(section: str, key: str) -> list[Any]:
        vals = cd.aligned(section, key)[-width:] if width else []
        return [None] * (width - len(vals)) + vals

    is_bank   = bool(pl_rows.get("Revenue")) and not pl_rows.get("Sales")
    sales_key = "Revenue" if is_bank else "Sales"
    opm_key   = "Financing Margin %" if is_bank else "OPM %"
    roce_key  = "ROE %" if is_bank else "ROCE %"

    return ReportFinancials(
        years=years,
        sales=row("profit_loss", sales_key),
        net_profit=row("profit_loss", "Net Profit"),
        eps=row("profit_loss", "EPS in Rs"),
        opm=row("profit_loss", opm_key),
        cfo=row("cash_flow", "Cash from Operating Activity"),
        roce=row("ratios", roce_key),
        fcf=row("cash_flow", "Free Cash Flow"),
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
