"""
Deterministic answers to lookup questions, straight from a fact sheet.

"What is TCS's ROE?" has exactly one right answer and it is already computed.
Sending it to an LLM costs quota, adds latency, and adds the chance of a wrong
digit. This module answers such questions itself.

It is deliberately conservative. It answers only when the question
  * names exactly one metric it knows, and
  * asks for a fact, not a judgement (no "should", "why", "compare", "risk", …).
Everything else returns None and goes to the LLM — with the fact sheet as its
primary context, so even then the numbers come from here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.facts.fact_sheet import METRICS, FactSheet, metric_label
from src.facts.formatting import fmt_value

# Words that make a question a judgement or an explanation, not a lookup.
_JUDGEMENT = re.compile(
    r"\b(why|should|would|could|will|future|forecast|predict|expect|outlook|"
    r"compare|comparison|versus|vs|better|worse|best|worst|buy|sell|hold|invest|"
    r"risk|risky|safe|explain|reason|impact|affect|good|bad|cheap|expensive|"
    r"overvalued|undervalued|worth|recommend|opinion|think)\b"
)

_GROWTH = re.compile(
    r"\b(growth|grow|grew|grown|growing|cagr|increase|increased|rise|rose|risen|"
    r"how fast|growth rate)\b"
)


def possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"

# Synonym → metric key. Longest phrase wins, and a matched span is consumed so
# "earnings per share" is EPS, not also "earnings" → net profit.
_SYNONYMS: dict[str, str] = {
    "return on equity": "roe", "roe": "roe",
    "return on capital employed": "roce", "roce": "roce",
    "operating margin": "operating_margin", "opm": "operating_margin",
    "ebitda margin": "operating_margin", "financing margin": "operating_margin",
    "profit margin": "operating_margin",
    "revenue": "revenue", "sales": "revenue", "turnover": "revenue",
    "top line": "revenue", "topline": "revenue",
    "net profit": "net_profit", "profit": "net_profit", "pat": "net_profit",
    "earnings": "net_profit", "bottom line": "net_profit",
    "earnings per share": "eps", "eps": "eps",
    "debt to equity": "debt_to_equity", "debt/equity": "debt_to_equity",
    "debt-to-equity": "debt_to_equity", "d/e": "debt_to_equity",
    "leverage": "debt_to_equity",
    "debt": "borrowings", "borrowings": "borrowings", "borrowing": "borrowings",
    "free cash flow": "fcf", "fcf": "fcf",
    "operating cash flow": "cfo", "cash from operations": "cfo", "cfo": "cfo",
    "dividend payout": "dividend_payout", "payout ratio": "dividend_payout",
    "interest coverage": "interest_coverage",
    "tax rate": "tax_rate",
    "working capital days": "working_capital_days",
    "book value": "book_value",
    "p/e": "pe", "pe ratio": "pe", "p/e ratio": "pe", "price to earnings": "pe",
    "price-to-earnings": "pe",
    "p/b": "pb", "pb ratio": "pb", "price to book": "pb", "price-to-book": "pb",
    "dividend yield": "dividend_yield",
    "market cap": "market_cap", "market capitalisation": "market_cap",
    "market capitalization": "market_cap",
    "intrinsic value": "intrinsic_value", "fair value": "intrinsic_value",
    "quality score": "score", "composite score": "score", "score": "score",
}
_PATTERNS = sorted(_SYNONYMS, key=len, reverse=True)

_MARKET_KEYS = {
    "pe": ("P/E ratio", "x"), "pb": ("P/B ratio", "x"),
    "dividend_yield": ("dividend yield", "pct"), "market_cap": ("market capitalisation", "cr"),
}


@dataclass
class FactAnswer:
    text: str
    metric: str
    facts: dict[str, Any] = field(default_factory=dict)
    source: str = "facts"


def detect_metrics(question: str) -> list[str]:
    """Metric keys named in the question (whole-word, longest phrase first)."""
    q = question.lower()
    found: list[str] = []
    for phrase in _PATTERNS:
        pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
        if re.search(pattern, q):
            key = _SYNONYMS[phrase]
            if key not in found:
                found.append(key)
            q = re.sub(pattern, " ", q)          # consume the span
    return found


def is_lookup(question: str) -> bool:
    return not _JUDGEMENT.search(question.lower())


def answer_from_facts(question: str, sheet: FactSheet) -> Optional[FactAnswer]:
    """A complete answer from the fact sheet, or None if this needs reasoning."""
    if not is_lookup(question):
        return None
    metrics = detect_metrics(question)
    if len(metrics) != 1:
        return None
    key = metrics[0]
    wants_growth = bool(_GROWTH.search(question.lower()))

    if key in _MARKET_KEYS:
        text = _market_answer(sheet, key)
    elif key == "intrinsic_value":
        text = _valuation_answer(sheet)
    elif key == "score":
        text = _score_answer(sheet)
    else:
        text = _metric_answer(sheet, key, wants_growth)
    if text is None:
        return None

    tail = [f"Source: {sheet.source}, data as of {sheet.data_as_of}."]
    tail += [f"Note: {f}" for f in sheet.flags]
    return FactAnswer(text=text + "\n\n" + "\n".join(tail), metric=key,
                      facts={"symbol": sheet.symbol, "metric": key})


def _peer_sentence(sheet: FactSheet, key: str) -> str:
    p = sheet.peers.get(key)
    if not p or p.get("sector_pct") is None or p.get("sector_n", 0) < 3:
        return ""
    direction = "higher" if p["higher_is_better"] else "lower"
    return (f" Against {p['sector_n']} {sheet.sector} companies in the Nifty 50 the sector "
            f"median is {fmt_value(p['sector_median'], p['unit'])}, placing it at the "
            f"{p['sector_pct']:.0f}th percentile ({direction} is better).")


def _metric_answer(sheet: FactSheet, key: str, wants_growth: bool) -> Optional[str]:
    row = sheet.metrics.get(key)
    if not row or row.get("latest") is None:
        return None
    d     = METRICS[key]
    label = metric_label(key, sheet.is_lender)
    fy    = sheet.years[-1] if sheet.years else "the latest year"
    name  = sheet.name

    if wants_growth and d.is_level:
        parts = []
        if row.get("yoy_growth") is not None:
            parts.append(f"{row['yoy_growth']*100:+.1f}% in {fy}")
        if row.get("cagr_3y") is not None:
            parts.append(f"{row['cagr_3y']*100:.1f}% a year over 3 years")
        if row.get("cagr_5y") is not None:
            parts.append(f"{row['cagr_5y']*100:.1f}% a year over 5 years")
        if not parts:
            return None
        text = (f"{possessive(name)} {label.lower()} grew {', '.join(parts)} "
                f"(compound annual growth), reaching {fmt_value(row['latest'], d.unit)} in {fy}.")
        peer_key = f"{key}_cagr_3y"
        return text + _peer_sentence(sheet, peer_key)

    text = f"{possessive(name)} {label} was {fmt_value(row['latest'], d.unit)} in {fy}"
    if row.get("previous") is not None:
        text += f", compared with {fmt_value(row['previous'], d.unit)} the year before"
    text += "."
    if row.get("mean_5y") is not None and row.get("z") is not None:
        z = row["z"]
        how = ("in line with" if abs(z) < 1 else
               "well above" if z >= 2 else "above" if z > 0 else
               "well below" if z <= -2 else "below")
        text += (f" That is {how} its 5-year average of {fmt_value(row['mean_5y'], d.unit)} "
                 f"({z:+.1f} standard deviations).")
    return text + _peer_sentence(sheet, key)


def _market_answer(sheet: FactSheet, key: str) -> Optional[str]:
    m = sheet.market
    label, unit = _MARKET_KEYS[key]
    value = m.get("market_cap_cr" if key == "market_cap" else key)
    if value is None:
        # The reason is known (no price, or non-positive earnings) and is in the
        # flags appended below; say so rather than hand it to an LLM to guess at.
        if m.get("price") is None:
            return f"{possessive(sheet.name)} {label} is not available: there is no reliable current price."
        if key == "pe":
            return (f"{possessive(sheet.name)} P/E ratio is not meaningful because its "
                    f"earnings per share are zero or negative.")
        return None
    text = f"{possessive(sheet.name)} {label} is {fmt_value(value, unit)}"
    if key == "pe":
        text += f" on {m.get('pe_basis')}, at a price of ₹{m['price']:,.2f}"
    elif key in ("pb", "dividend_yield", "market_cap"):
        text += f" at a price of ₹{m['price']:,.2f}"
    return text + "." + _peer_sentence(sheet, key)


def _valuation_answer(sheet: FactSheet) -> Optional[str]:
    v = sheet.valuation
    if not v.get("intrinsic_value"):
        return None
    text = (f"The {v['model']} model estimates {possessive(sheet.name)} intrinsic value at "
            f"₹{v['intrinsic_value']:,.0f} per share (median of 10,000 simulated scenarios), "
            f"with an 80% range of ₹{v['iv_low']:,.0f}–₹{v['iv_high']:,.0f} "
            f"(discount rate {v['discount_rate']*100:.1f}%).")
    price = sheet.market.get("price")
    if price:
        text += f" The current price is ₹{price:,.2f}"
        if v.get("prob_undervalued") is not None:
            text += (f"; the model value exceeds it in {v['prob_undervalued']*100:.0f}% "
                     f"of scenarios")
        text += "."
    if v.get("implied_status") == "solved" and v.get("implied_growth") is not None:
        text += (f" Working backwards, today's price implies starting cash-flow growth of "
                 f"{v['implied_growth']*100:.1f}% a year")
        if v.get("historical_growth") is not None:
            text += f", against {v['historical_growth']*100:.1f}% estimated from history"
        text += "."
    text += (" A model estimate is sensitive to its growth and discount-rate inputs and "
             "is not a price target.")
    return text


def _score_answer(sheet: FactSheet) -> Optional[str]:
    s = sheet.score
    if s.get("composite") is None:
        return None
    return (f"{sheet.name} scores {s['composite']:.0f}/100 on the composite quality model "
            f"(grade {s['grade']}), ranking {s['rank']} of {s['of']} Nifty 50 companies. "
            f"Components: quality {s['quality']:.0f}, growth {s['growth']:.0f}, earnings "
            f"quality {s['earnings_quality']:.0f}; Piotroski F-score {s['piotroski']}/9.")
