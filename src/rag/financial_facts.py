"""
Verified financial figures for a question, read directly from Screener.in data.

Semantic search is unreliable for exact numbers: every year's balance-sheet chunk
looks almost identical to the embedder, so "borrowings in FY2022" often retrieves
FY2014–FY2016 instead. This module skips search entirely — it detects the companies
and fiscal years in the question and returns a compact table of the exact figures,
plus pre-computed growth, margin, debt-to-equity and CAGR, so the LLM never has to
do arithmetic or pick numbers out of loosely related text.

Usage:
    facts = FinancialFacts()
    block = facts.for_question("What was TCS's net profit in FY2025?", ticker="TCS.NS")
"""

import json
import os
import re
from functools import lru_cache
from typing import Any

from config.nifty50_tickers import NIFTY50_TICKERS

SCREENER_DIR = "data/raw/screener"

# (display label, statement, candidate row names) — first row present wins
METRICS = [
    ("Revenue", "profit_loss", ["Sales", "Revenue"]),
    ("Operating Profit", "profit_loss", ["Operating Profit", "Financing Profit"]),
    ("Net Profit", "profit_loss", ["Net Profit"]),
    ("EPS (₹)", "profit_loss", ["EPS in Rs"]),
    ("Equity Capital", "balance_sheet", ["Equity Capital"]),
    ("Reserves", "balance_sheet", ["Reserves"]),
    ("Borrowings", "balance_sheet", ["Borrowings", "Borrowing"]),
    ("Total Assets", "balance_sheet", ["Total Assets"]),
    ("Cash from Operations", "cash_flow", ["Cash from Operating Activity"]),
    ("Free Cash Flow", "cash_flow", ["Free Cash Flow"]),
    ("ROCE %", "ratios", ["ROCE %"]),
    ("ROE %", "ratios", ["ROE %"]),
]

_YEAR_RE = re.compile(r"\b(?:FY\s?'?|March\s+|Mar\s+)?(20\d{2})\b", re.IGNORECASE)
_FY_SHORT_RE = re.compile(r"\bFY\s?'?(\d{2})\b", re.IGNORECASE)
_MULTI_YEAR_RE = re.compile(r"cagr|\b\d+[- ]year|trend|over the (?:last|past)|history", re.IGNORECASE)

# Aliases people commonly use that aren't the official Nifty 50 names
_ALIASES = {
    "reliance": "RELIANCE", "infosys": "INFY", "hdfc bank": "HDFCBANK", "icici": "ICICIBANK",
    "sbi": "SBIN", "state bank": "SBIN", "airtel": "BHARTIARTL", "l&t": "LT",
    "larsen": "LT", "hul": "HINDUNILVR", "hindustan unilever": "HINDUNILVR",
    "kotak": "KOTAKBANK", "m&m": "M&M", "mahindra": "M&M", "maruti": "MARUTI",
    "tata motors": "TATAMOTORS", "tata steel": "TATASTEEL", "wipro": "WIPRO",
}


@lru_cache(maxsize=64)
def _load(symbol: str) -> dict[str, Any] | None:
    path = os.path.join(SCREENER_DIR, f"{symbol}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _series(data: dict[str, Any], statement: str, rows: list[str]) -> dict[int, float]:
    block = data.get(statement) or {}
    years = block.get("years") or []
    for row in rows:
        values = (block.get("rows") or {}).get(row)
        if not values:
            continue
        out: dict[int, float] = {}
        for label, value in zip(years, values):
            m = re.fullmatch(r"Mar (\d{4})", label)
            if value is not None and m:
                out[int(m.group(1))] = float(value)
        if out:
            return out
    return {}


def _fmt(label: str, value: float) -> str:
    if label.endswith("%"):
        return f"{value:.1f}%"
    if label.startswith("EPS"):
        return f"₹{value:,.2f}"
    return f"₹{value:,.0f} Cr"


@lru_cache(maxsize=1)
def _short_names() -> dict[str, str]:
    """Two-word prefixes ("hdfc life", "sun pharma") that identify exactly one company."""
    counts: dict[str, list[str]] = {}
    for sym, name in NIFTY50_TICKERS.items():
        words = name.lower().replace("'s", "").split()
        if len(words) > 2 and words[1] != "&":
            counts.setdefault(" ".join(words[:2]), []).append(sym)
    return {short: syms[0] for short, syms in counts.items() if len(syms) == 1}


class FinancialFacts:

    def detect_symbols(self, question: str, ticker: str = "") -> list[str]:
        """All Nifty 50 companies mentioned, in order of first mention; `ticker` is always included."""
        q = question.lower()
        hits: dict[str, int] = {}
        names = {name.lower(): sym for sym, name in NIFTY50_TICKERS.items()}
        names.update({sym.lower(): sym for sym in NIFTY50_TICKERS if len(sym) > 2})
        names.update(_short_names())
        names.update(_ALIASES)
        # longest names first so "hdfc bank" wins over "hdfc"
        for name in sorted(names, key=len, reverse=True):
            pos = re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", q)
            sym = names[name]
            if pos and sym in NIFTY50_TICKERS and sym not in hits:
                hits[sym] = pos.start()
        symbols = sorted(hits, key=hits.get)
        base = ticker.replace(".NS", "").replace(".BO", "")
        if base and base in NIFTY50_TICKERS and base not in symbols:
            symbols.insert(0, base)
        return symbols

    def detect_years(self, question: str) -> list[int]:
        years = {int(y) for y in _YEAR_RE.findall(question)}
        years |= {2000 + int(y) for y in _FY_SHORT_RE.findall(question)}
        return sorted(years)

    def for_question(self, question: str, ticker: str = "") -> str:
        """Return a verified-figures block for the question, or '' if no company was found."""
        symbols = self.detect_symbols(question, ticker)
        if not symbols:
            return ""
        asked = self.detect_years(question)
        multi_year = bool(_MULTI_YEAR_RE.search(question))
        blocks = [self._company_block(s, asked, multi_year) for s in symbols[:4]]
        blocks = [b for b in blocks if b]
        if not blocks:
            return ""
        header = (
            "VERIFIED FIGURES (source: Screener.in, consolidated where available; ₹ Crore unless noted).\n"
            "These are exact reported numbers — quote them as given and prefer them over any other context.\n"
            "State amounts in ₹ Crore exactly as shown; do not convert to lakh crore, billion or trillion.\n"
            "Use the pre-computed growth, margin and ratio figures instead of doing your own arithmetic.\n"
            "FY2025 means the fiscal year ended March 2025."
        )
        return header + "\n\n" + "\n\n".join(blocks)

    def _company_block(self, symbol: str, asked: list[int], multi_year: bool) -> str:
        data = _load(symbol)
        if not data:
            return ""
        series = {label: _series(data, st, rows) for label, st, rows in METRICS}
        series = {k: v for k, v in series.items() if v}
        available = sorted({y for s in series.values() for y in s})
        if not available:
            return ""
        first, last = available[0], available[-1]
        name = NIFTY50_TICKERS.get(symbol, symbol)
        lines = [f"### {name} ({symbol}) — data available FY{first} to FY{last}"]

        missing = [y for y in asked if y not in available]
        for y in missing:
            if y > last:
                lines.append(f"FY{y}: NO DATA — this year has not been reported yet (latest available is FY{last}). "
                             f"Say the figure is not available; do not estimate it.")
            else:
                lines.append(f"FY{y}: NO DATA in the source.")

        # Years to show: asked years (+ prior year for growth), or the latest 3; full window for multi-year asks
        in_range = [y for y in asked if y in available]
        if multi_year:
            years = available[-6:]
        elif in_range:
            years = sorted({y for a in in_range for y in (a - 1, a) if y in available})
        else:
            years = available[-3:]

        lines.append("| Metric | " + " | ".join(f"FY{y}" for y in years) + " |")
        lines.append("|---|" + "---|" * len(years))
        for label, s in series.items():
            cells = [_fmt(label, s[y]) if y in s else "—" for y in years]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        lines.extend(self._derived(series, years, available, multi_year))
        return "\n".join(lines)

    def _derived(self, series: dict[str, dict[int, float]], years: list[int],
                 available: list[int], multi_year: bool) -> list[str]:
        rev, np_ = series.get("Revenue", {}), series.get("Net Profit", {})
        eq, res, debt = series.get("Equity Capital", {}), series.get("Reserves", {}), series.get("Borrowings", {})
        out = ["Pre-computed:"]
        for y in years:
            parts = []
            for label, s in (("Revenue", rev), ("Net profit", np_)):
                if y in s and y - 1 in s and s[y - 1] > 0:
                    parts.append(f"{label} growth YoY {(s[y] / s[y - 1] - 1) * 100:+.2f}%")
            if y in rev and y in np_ and rev[y] > 0:
                parts.append(f"net margin {np_[y] / rev[y] * 100:.2f}%")
            if y in eq and y in res and y in debt and (eq[y] + res[y]) > 0:
                parts.append(f"debt-to-equity {debt[y] / (eq[y] + res[y]):.2f}")
            if parts:
                out.append(f"- FY{y}: " + "; ".join(parts))
        if multi_year or len(available) >= 6:
            yrs = sorted(set(rev) & set(available))
            if len(yrs) >= 6 and rev[yrs[-6]] > 0:
                y0, y1 = yrs[-6], yrs[-1]
                out.append(f"- Revenue 5-year CAGR FY{y0}→FY{y1}: {((rev[y1] / rev[y0]) ** 0.2 - 1) * 100:.2f}%")
            yrs = sorted(set(np_) & set(available))
            if len(yrs) >= 6 and np_[yrs[-6]] > 0 and np_[yrs[-1]] > 0:
                y0, y1 = yrs[-6], yrs[-1]
                out.append(f"- Net profit 5-year CAGR FY{y0}→FY{y1}: {((np_[y1] / np_[y0]) ** 0.2 - 1) * 100:.2f}%")
        return out if len(out) > 1 else []
