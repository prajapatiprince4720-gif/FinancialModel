"""
Fact sheets: every number the research layer may state about a company,
computed in Python, with its source.

Why this replaces bulk FAQ pre-generation
-----------------------------------------
The old approach had an LLM pre-write ~180 answers per company and stored them
as trusted context. Those answers went stale on every data refresh, cost days of
API quota, and any number the model invented was then retrieved as fact.

A fact sheet inverts that: Python computes the numbers (deterministically, in
milliseconds, always from the current data), and an LLM — when one is needed at
all — only explains them. Simple lookups ("What is TCS's ROE?") are answered
straight from the sheet with no model call (see src/facts/answerer.py).

Statistics attached to each metric
----------------------------------
For a metric series x_1..x_n (one value per fiscal year):

* CAGR over k years      (x_n / x_{n-k})^(1/k) − 1        — positive level series only
* Trend slope            OLS fit x_t = a + b·t over the last 5 years; b is the
                         average change per year, in the metric's own units
* Volatility             sample standard deviation over the last 5 years
* z-score                (x_n − mean_5y) / sd_5y — how unusual this year is
                         relative to the company's own recent history
* Peer percentile        rank of the latest value within the sector group and
                         within the whole universe (lenders ranked among lenders)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from config.nifty50_tickers import NIFTY50_TICKERS
from src.data.company import CORPORATE_ACTIONS, CompanyData, load_company

# How many years of history the per-metric statistics use.
_WINDOW = 5


# ── Metric catalogue ──────────────────────────────────────────────────────────
# (key, label, unit, higher_is_better, is_level)
#   is_level: a quantity that compounds (sales, profit) — CAGR is meaningful.
#             Ratios (margins, returns) get trend/z-score but no CAGR.

@dataclass(frozen=True)
class MetricDef:
    key:    str
    label:  str
    unit:   str            # "cr" (₹ Crore), "rs" (₹/share), "pct", "x" (multiple), "days"
    higher_is_better: Optional[bool]
    is_level: bool


METRICS: dict[str, MetricDef] = {m.key: m for m in [
    MetricDef("revenue",          "Revenue",                "cr",  True,  True),
    MetricDef("net_profit",       "Net profit",             "cr",  True,  True),
    MetricDef("eps",              "EPS",                    "rs",  True,  True),
    MetricDef("operating_margin", "Operating margin",       "pct", True,  False),
    MetricDef("roce",             "ROCE",                   "pct", True,  False),
    MetricDef("roe",              "ROE",                    "pct", True,  False),
    MetricDef("debt_to_equity",   "Debt / equity",          "x",   False, False),
    MetricDef("borrowings",       "Borrowings",             "cr",  False, True),
    MetricDef("book_value",       "Book value (equity)",    "cr",  True,  True),
    MetricDef("cfo",              "Cash from operations",   "cr",  True,  True),
    MetricDef("fcf",              "Free cash flow",         "cr",  True,  True),
    MetricDef("cfo_to_profit",    "Cash conversion (CFO / net profit)", "x", True, False),
    MetricDef("interest_coverage","Interest coverage",      "x",   True,  False),
    MetricDef("dividend_payout",  "Dividend payout",        "pct", None,  False),
    MetricDef("tax_rate",         "Effective tax rate",     "pct", None,  False),
    MetricDef("working_capital_days", "Working capital days", "days", False, False),
]}

# Lenders' rows mean something different under the same key; say so in the label.
_LENDER_LABELS = {
    "operating_margin": "Financing margin (financing profit / revenue)",
    "debt_to_equity":   "Borrowings (excl. deposits) / equity",
    "revenue":          "Revenue (interest + other operating income)",
}


def metric_label(key: str, is_lender: bool) -> str:
    if is_lender and key in _LENDER_LABELS:
        return _LENDER_LABELS[key]
    return METRICS[key].label


# Metrics compared across peers on the fact sheet.
PEER_METRICS = ("roce", "roe", "operating_margin", "revenue_cagr_3y",
                "net_profit_cagr_3y", "debt_to_equity", "pe", "pb")


# ── Small numeric helpers ─────────────────────────────────────────────────────

def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if not _finite(a) or not _finite(b) or b == 0:
        return None
    return a / b


def cagr(values: list[Optional[float]], years: int) -> Optional[float]:
    """Compound annual growth over the last `years` intervals; None if undefined."""
    if len(values) < years + 1:
        return None
    start, end = values[-(years + 1)], values[-1]
    if not _finite(start) or not _finite(end) or start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def trend_stats(values: list[Optional[float]], window: int = _WINDOW) -> dict[str, Optional[float]]:
    """
    OLS slope, mean, standard deviation and z-score of the latest value.

    Slope solves the normal equations for x_t = a + b·t, i.e. b = cov(t, x) / var(t),
    over the finite values among the last `window` years.
    """
    tail = values[-window:]
    pts  = [(t, v) for t, v in enumerate(tail) if _finite(v)]
    out: dict[str, Optional[float]] = {"slope": None, "mean": None, "sd": None, "z": None}
    if len(pts) < 3:
        return out
    t = np.array([p[0] for p in pts], dtype=float)
    x = np.array([p[1] for p in pts], dtype=float)
    tc = t - t.mean()
    out["slope"] = float((tc @ (x - x.mean())) / (tc @ tc))
    out["mean"]  = float(x.mean())
    sd = float(x.std(ddof=1))
    out["sd"] = sd
    if sd > 0 and _finite(tail[-1]):
        out["z"] = float((tail[-1] - x.mean()) / sd)
    return out


def percentile_rank(value: Optional[float], population: list[Optional[float]],
                    higher_is_better: bool = True) -> Optional[float]:
    """
    Percentile of `value` within `population` (0 = worst, 100 = best).

    Mid-rank for ties: (count below + ½·count equal − ½) / (n − 1), so the best of
    n gets 100 and the worst gets 0 regardless of n.
    """
    pop = [p for p in population if _finite(p)]
    if not _finite(value) or len(pop) < 2:
        return None
    below = sum(p < value for p in pop)
    equal = sum(p == value for p in pop)
    pct = (below + 0.5 * equal - 0.5) / (len(pop) - 1) * 100
    pct = max(0.0, min(100.0, pct))
    return pct if higher_is_better else 100.0 - pct


# ── Per-company extraction ────────────────────────────────────────────────────

def _series(cd: CompanyData) -> dict[str, list[Optional[float]]]:
    """Year-aligned series for every catalogued metric (None where undefined)."""
    pl, bs, cf, rat = "profit_loss", "balance_sheet", "cash_flow", "ratios"
    n = len(cd.years(pl))

    def fit(vals: list[Optional[float]]) -> list[Optional[float]]:
        """Right-align a statement row onto the P&L year axis."""
        vals = vals[-n:] if n else []
        return [None] * (n - len(vals)) + vals

    # Lenders usually report "Revenue"/"Financing Margin %", but a lending
    # holding company (BAJAJFINSV) reports "Sales"/"OPM %", so fall back.
    if cd.is_lender:
        revenue = fit(cd.aligned(pl, "Revenue", "Sales"))
        margin  = fit(cd.aligned(pl, "Financing Margin %", "OPM %"))
    else:
        revenue = fit(cd.aligned(pl, "Sales", "Revenue"))
        margin  = fit(cd.aligned(pl, "OPM %", "Financing Margin %"))
    np_      = fit(cd.aligned(pl, "Net Profit"))
    eps      = fit(cd.aligned(pl, "EPS in Rs"))
    interest = fit(cd.aligned(pl, "Interest"))
    pbt      = fit(cd.aligned(pl, "Profit before tax"))
    payout   = fit(cd.aligned(pl, "Dividend Payout %"))
    tax      = fit(cd.aligned(pl, "Tax %"))
    eq_cap   = fit(cd.aligned(bs, "Equity Capital"))
    reserves = fit(cd.aligned(bs, "Reserves"))
    borrow   = fit(cd.aligned(bs, "Borrowings", "Borrowing"))
    cfo      = fit(cd.aligned(cf, "Cash from Operating Activity"))
    fcf      = fit(cd.aligned(cf, "Free Cash Flow"))
    roce     = fit(cd.aligned(rat, "ROCE %"))
    roe_rep  = fit(cd.aligned(rat, "ROE %"))
    wc_days  = fit(cd.aligned(rat, "Working Capital Days"))

    book = [a + b if _finite(a) and _finite(b) else None for a, b in zip(eq_cap, reserves)]

    # ROE on average equity, unless Screener reports it (lenders).
    roe: list[Optional[float]] = []
    for i in range(n):
        if _finite(roe_rep[i]):
            roe.append(roe_rep[i])
            continue
        avg_eq = None
        if i > 0 and _finite(book[i]) and _finite(book[i - 1]):
            avg_eq = (book[i] + book[i - 1]) / 2
        elif _finite(book[i]):
            avg_eq = book[i]
        r = _div(np_[i], avg_eq)
        roe.append(r * 100 if r is not None else None)

    cover = []
    for i in range(n):
        if cd.is_lender:
            cover.append(None)          # interest is the cost of goods for a lender
        else:
            cover.append(_div((pbt[i] or 0) + (interest[i] or 0), interest[i])
                         if _finite(pbt[i]) and _finite(interest[i]) and interest[i] > 0 else None)

    return {
        "revenue": revenue, "net_profit": np_, "eps": eps,
        "operating_margin": margin,
        "roce": [None] * n if cd.is_lender else roce,
        "roe": roe,
        "debt_to_equity": [_div(b, e) for b, e in zip(borrow, book)],
        "borrowings": borrow, "book_value": book,
        "cfo": cfo, "fcf": fcf,
        "cfo_to_profit": [_div(c, p) if _finite(p) and p > 0 else None for c, p in zip(cfo, np_)],
        "interest_coverage": cover,
        "dividend_payout": payout, "tax_rate": tax,
        "working_capital_days": [None] * n if cd.is_lender else wc_days,
    }


@dataclass
class FactSheet:
    symbol: str
    name: str
    sector: str
    is_lender: bool
    years: list[str]
    data_as_of: Optional[str]
    source: str
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    market: dict[str, Any] = field(default_factory=dict)
    valuation: dict[str, Any] = field(default_factory=dict)
    score: dict[str, Any] = field(default_factory=dict)
    peers: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "name": self.name, "sector": self.sector,
            "is_lender": self.is_lender, "years": self.years,
            "data_as_of": self.data_as_of, "source": self.source,
            "metrics": self.metrics, "market": self.market,
            "valuation": self.valuation, "score": self.score,
            "peers": self.peers, "flags": self.flags,
        }

    def to_context(self) -> str:
        """Compact plain-text rendering for an LLM prompt. Every line is a fact."""
        from src.facts.formatting import fmt_value
        fy = self.years[-1] if self.years else "latest FY"
        lines = [
            f"FACT SHEET — {self.name} ({self.symbol}), sector {self.sector}",
            f"Source: {self.source}; latest full year {fy}; data as of {self.data_as_of}.",
        ]
        m = self.market
        if m.get("price"):
            lines.append(
                f"Price ₹{m['price']:,.2f}; market cap {fmt_value(m.get('market_cap_cr'), 'cr')}; "
                f"P/E {fmt_value(m.get('pe'), 'x')} ({m.get('pe_basis')}); "
                f"P/B {fmt_value(m.get('pb'), 'x')}; dividend yield {fmt_value(m.get('dividend_yield'), 'pct')}."
            )
        for key, row in self.metrics.items():
            if row.get("latest") is None:
                continue
            d = METRICS[key]
            parts = [f"{metric_label(key, self.is_lender)} {fy}: {fmt_value(row['latest'], d.unit)}"]
            if row.get("previous") is not None:
                parts.append(f"prior year {fmt_value(row['previous'], d.unit)}")
            if row.get("cagr_3y") is not None:
                parts.append(f"3Y CAGR {row['cagr_3y']*100:.1f}%")
            if row.get("cagr_5y") is not None:
                parts.append(f"5Y CAGR {row['cagr_5y']*100:.1f}%")
            if row.get("z") is not None:
                parts.append(f"z vs own 5Y history {row['z']:+.1f}")
            lines.append("; ".join(parts) + ".")
        v = self.valuation
        if v.get("intrinsic_value"):
            lines.append(
                f"Valuation ({v['model']}, 10,000 simulated scenarios): median intrinsic value "
                f"₹{v['intrinsic_value']:,.0f}; 50% range ₹{v['iv_p25']:,.0f}–₹{v['iv_p75']:,.0f}; "
                f"80% range ₹{v['iv_low']:,.0f}–₹{v['iv_high']:,.0f}; "
                f"discount rate {v['discount_rate']*100:.1f}%; model verdict {v['verdict']}."
            )
            if v.get("prob_undervalued") is not None:
                lines.append(
                    f"Probability the model value exceeds the price: "
                    f"{v['prob_undervalued']*100:.0f}% (model-based, not a forecast of returns)."
                )
            if v.get("implied_status") == "solved" and v.get("implied_growth") is not None:
                hist = v.get("historical_growth")
                lines.append(
                    f"Reverse DCF: the price implies starting cash-flow growth of "
                    f"{v['implied_growth']*100:.1f}% a year fading to the long-run rate"
                    + (f", vs {hist*100:.1f}% estimated from history." if hist is not None else ".")
                )
            sens = v.get("sensitivities") or {}
            names = {"discount_rate": "discount rate", "initial_growth": "starting growth",
                     "terminal_growth": "long-run growth", "long_run_roe": "long-run ROE"}
            parts = [f"{names[k]} {s:+.1f}%" for k, s in sens.items() if k in names and s is not None]
            if parts:
                lines.append("Value change per +1 percentage point: " + "; ".join(parts) + ".")
        s = self.score
        if s.get("composite") is not None:
            lines.append(
                f"Composite quality score {s['composite']:.0f}/100 (grade {s['grade']}), "
                f"rank {s['rank']} of {s['of']} in Nifty 50; Piotroski {s['piotroski']}/9."
            )
        for key, p in self.peers.items():
            if p.get("sector_pct") is None:
                continue
            unit = p.get("unit", "")
            direction = "higher is better" if p["higher_is_better"] else "lower is better"
            lines.append(
                f"Peer standing — {p['label']}: {fmt_value(p['value'], unit)} vs sector median "
                f"{fmt_value(p['sector_median'], unit)} across {p['sector_n']} {self.sector} "
                f"companies; sector percentile {p['sector_pct']:.0f} ({direction})."
            )
        for f in self.flags:
            lines.append(f"Caveat: {f}")
        return "\n".join(lines)


def _metric_rows(series: dict[str, list[Optional[float]]], years: list[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key, vals in series.items():
        d = METRICS[key]
        if not any(_finite(v) for v in vals):
            continue
        latest = vals[-1] if vals and _finite(vals[-1]) else None
        prev   = vals[-2] if len(vals) >= 2 and _finite(vals[-2]) else None
        stats  = trend_stats(vals)
        row: dict[str, Any] = {
            "history": {y: (round(v, 4) if _finite(v) else None) for y, v in zip(years, vals)},
            "latest": latest, "previous": prev,
            "yoy_change": (latest - prev) if latest is not None and prev is not None else None,
            "trend_slope": stats["slope"], "mean_5y": stats["mean"],
            "sd_5y": stats["sd"], "z": stats["z"],
        }
        if d.is_level:
            row["yoy_growth"] = _div(latest - prev, abs(prev)) if latest is not None and prev else None
            row["cagr_3y"] = cagr(vals, 3)
            row["cagr_5y"] = cagr(vals, 5)
        rows[key] = row
    return rows


def _market(cd: CompanyData, series: dict[str, list[Optional[float]]],
            price: Optional[float]) -> dict[str, Any]:
    eps_fy  = series["eps"][-1] if series["eps"] else None
    np_fy   = series["net_profit"][-1] if series["net_profit"] else None
    shares  = _div(np_fy, eps_fy)                      # crore shares
    eps_ttm = cd.ttm("profit_loss", "EPS in Rs")
    eps_for_pe, basis = (eps_ttm, "TTM EPS") if _finite(eps_ttm) else (eps_fy, "FY EPS")
    book    = series["book_value"][-1] if series["book_value"] else None
    payout  = series["dividend_payout"][-1] if series["dividend_payout"] else None

    out: dict[str, Any] = {"price": price, "shares_cr": shares, "pe_basis": basis}
    if not _finite(price):
        return out
    out["market_cap_cr"] = price * shares if _finite(shares) else None
    out["pe"] = _div(price, eps_for_pe) if _finite(eps_for_pe) and eps_for_pe > 0 else None
    out["earnings_yield"] = _div(eps_for_pe, price) * 100 if _finite(eps_for_pe) else None
    bvps = _div(book, shares)
    out["book_value_per_share"] = bvps
    out["pb"] = _div(price, bvps) if _finite(bvps) and bvps > 0 else None
    if _finite(payout) and _finite(eps_fy) and eps_fy > 0:
        out["dividend_per_share"] = eps_fy * payout / 100
        out["dividend_yield"] = out["dividend_per_share"] / price * 100
    return out


def _quality_flags(cd: CompanyData, market: dict[str, Any], metrics: dict[str, Any],
                   data_as_of: Optional[str], age_days: Optional[float]) -> list[str]:
    flags = []
    if cd.symbol in CORPORATE_ACTIONS:
        flags.append(CORPORATE_ACTIONS[cd.symbol])
    if not _finite(market.get("price")):
        flags.append("No live price available — price-based figures (P/E, P/B, margin of safety) omitted.")
    pe = market.get("pe")
    if pe is not None and pe < 2:
        flags.append(f"P/E of {pe:.1f}x is implausibly low — check for a stock split or a bad price.")
    elif pe is not None and pe > 200:
        flags.append(f"P/E of {pe:.0f}x: earnings are tiny relative to price, so P/E is "
                     f"not a meaningful valuation measure here.")
    elif pe is None and _finite(market.get("price")):
        flags.append("P/E not meaningful: earnings are zero or negative.")
    if age_days is not None and age_days > 90:
        flags.append(f"Fundamentals are {age_days:.0f} days old.")
    if not metrics.get("revenue"):
        flags.append("Revenue history missing.")
    return flags


def valuation_summary(v: Any) -> dict[str, Any]:
    """The fact-sheet view of a DCFResult; {} when no valuation was possible."""
    # A zero value with a verdict is a result (debt exceeds value), not a gap.
    if v is None or getattr(v, "verdict", "") in ("INSUFFICIENT_DATA", ""):
        return {}
    return {
        "model": v.model,
        "intrinsic_value": v.intrinsic_value,       # median of the simulation
        "iv_low": v.iv_low, "iv_high": v.iv_high,   # P10 / P90
        "iv_p25": v.iv_p25, "iv_p75": v.iv_p75,
        "iv_base": v.iv_base,
        "prob_undervalued": v.prob_undervalued,
        "margin_of_safety": v.margin_of_safety,
        "discount_rate": v.wacc,
        "historical_growth": v.fcf_cagr if v.model == "DCF" else None,
        "implied_growth": v.implied_growth,
        "implied_status": v.implied_status,
        "sensitivities": dict(v.sensitivities or {}),
        "terminal_share": v.terminal_share,
        "verdict": v.verdict,
        "notes": list(v.notes or []),
    }


# ── Builder ───────────────────────────────────────────────────────────────────

def build_fact_sheets(
    symbols: Optional[list[str]] = None,
    prices: Optional[dict[str, Optional[float]]] = None,
    valuations: Optional[dict[str, Any]] = None,
    scores: Optional[dict[str, Any]] = None,
    sectors: Optional[dict[str, str]] = None,
    freshness: Optional[dict[str, dict]] = None,
) -> dict[str, FactSheet]:
    """
    Build fact sheets for many companies at once (peer ranks need the universe).

    All external inputs are injectable so this stays pure and testable; the
    research service supplies live prices, valuations and scores.
    """
    from src.data.sectors import SECTOR

    symbols    = symbols or list(NIFTY50_TICKERS)
    prices     = prices or {}
    valuations = valuations or {}
    scores     = scores or {}
    sectors    = sectors or SECTOR
    freshness  = freshness or {}

    sheets: dict[str, FactSheet] = {}
    for sym in symbols:
        cd = load_company(sym)
        if cd is None:
            continue
        years  = cd.years("profit_loss")
        series = _series(cd)
        fresh  = freshness.get(sym, {})
        sheet  = FactSheet(
            symbol=sym, name=NIFTY50_TICKERS.get(sym, sym),
            sector=sectors.get(sym, "DEFAULT"), is_lender=cd.is_lender,
            years=years, data_as_of=fresh.get("last_modified"),
            source=("Screener.in " + ("consolidated" if cd.raw.get("consolidated") else "standalone")
                    + " annual statements"),
        )
        sheet.metrics = _metric_rows(series, years)
        sheet.market  = _market(cd, series, prices.get(sym))
        sheet.flags   = _quality_flags(cd, sheet.market, sheet.metrics,
                                       sheet.data_as_of, fresh.get("age_days"))

        v = valuations.get(sym)
        if v is not None:
            sheet.valuation = valuation_summary(v)
        sheets[sym] = sheet

    _attach_scores(sheets, scores)
    _attach_peers(sheets)
    return sheets


def _attach_scores(sheets: dict[str, FactSheet], scores: dict[str, Any]) -> None:
    ranked = sorted((s for s in scores.values() if s.symbol in sheets),
                    key=lambda r: r.composite, reverse=True)
    for rank, r in enumerate(ranked, 1):
        sheets[r.symbol].score = {
            "composite": r.composite, "grade": r.grade, "verdict": r.verdict,
            "quality": r.quality, "growth": r.growth,
            "earnings_quality": r.earnings_quality, "piotroski": r.piotroski,
            "rank": rank, "of": len(ranked),
        }


def _peer_value(sheet: FactSheet, key: str) -> Optional[float]:
    if key in ("pe", "pb"):
        return sheet.market.get(key)
    if key.endswith("_cagr_3y"):
        base = key[: -len("_cagr_3y")]
        return (sheet.metrics.get(base) or {}).get("cagr_3y")
    return (sheet.metrics.get(key) or {}).get("latest")


_PEER_META = {
    "roce": ("ROCE", "pct", True), "roe": ("ROE", "pct", True),
    "operating_margin": ("Operating margin", "pct", True),
    "revenue_cagr_3y": ("Revenue 3Y CAGR", "ratio_pct", True),
    "net_profit_cagr_3y": ("Net profit 3Y CAGR", "ratio_pct", True),
    "debt_to_equity": ("Debt / equity", "x", False),
    "pe": ("P/E", "x", False), "pb": ("P/B", "x", False),
}


def _attach_peers(sheets: dict[str, FactSheet]) -> None:
    """
    Sector and universe percentiles for headline metrics.

    Lender-specific measures (margin, leverage) are only compared among lenders,
    for the same reason the composite scorer ranks them separately.
    """
    lender_sensitive = {"operating_margin", "debt_to_equity", "roce"}
    for sym, sheet in sheets.items():
        for key in PEER_METRICS:
            value = _peer_value(sheet, key)
            if value is None:
                continue
            label, unit, hib = _PEER_META[key]
            same_kind = [s for s in sheets.values()
                         if key not in lender_sensitive or s.is_lender == sheet.is_lender]
            sector = [s for s in same_kind if s.sector == sheet.sector]
            sector_vals = [_peer_value(s, key) for s in sector]
            universe_vals = [_peer_value(s, key) for s in same_kind]
            finite_sector = [x for x in sector_vals if _finite(x)]
            sheet.peers[key] = {
                "label": metric_label(key, sheet.is_lender) if key in METRICS else label,
                "unit": unit, "value": value, "higher_is_better": hib,
                "sector_pct": percentile_rank(value, sector_vals, hib),
                "sector_n": len(finite_sector),
                "sector_median": float(np.median(finite_sector)) if finite_sector else None,
                "universe_pct": percentile_rank(value, universe_vals, hib),
                "universe_n": sum(_finite(x) for x in universe_vals),
            }
