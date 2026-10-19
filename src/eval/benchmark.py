"""
EquityLens benchmark — question/answer pairs with exact gold answers derived from
Screener.in financials (data/raw/screener/*.json, values in ₹ Crore).

Categories:
  lookup        a single reported figure for one fiscal year
  derived       a figure that must be calculated (growth, margin, D/E, CAGR)
  compare       which of two companies is higher on a metric
  unanswerable  a fiscal year with no data — the correct answer is to abstain

Generation is deterministic (fixed seed), so every run scores the same questions.

Usage:
    python main.py eval build
"""

import json
import os
import random
import re
from typing import Any

from config.nifty50_tickers import NIFTY50_TICKERS

SCREENER_DIR = "data/raw/screener"
BENCHMARK_PATH = "data/eval/benchmark_v1.jsonl"
SEED = 42

# (label, statement, candidate row names, unit) — first row name present wins
LOOKUP_METRICS = [
    ("revenue", "profit_loss", ["Sales", "Revenue"], "crore"),
    ("net profit", "profit_loss", ["Net Profit"], "crore"),
    ("EPS", "profit_loss", ["EPS in Rs"], "rupees"),
    ("total borrowings", "balance_sheet", ["Borrowings", "Borrowing"], "crore"),
    ("reserves", "balance_sheet", ["Reserves"], "crore"),
    ("cash flow from operating activities", "cash_flow", ["Cash from Operating Activity"], "crore"),
    ("ROCE", "ratios", ["ROCE %"], "percent"),
]


def _load(symbol: str) -> dict[str, Any] | None:
    path = os.path.join(SCREENER_DIR, f"{symbol}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _series(data: dict[str, Any], statement: str, rows: list[str]) -> dict[int, float]:
    """Return {fiscal_year: value} for the first available row, skipping TTM and blanks."""
    block = data.get(statement) or {}
    years = block.get("years") or []
    for row in rows:
        values = (block.get("rows") or {}).get(row)
        if not values:
            continue
        out: dict[int, float] = {}
        for label, value in zip(years, values):
            m = re.fullmatch(r"Mar (\d{4})", label)
            if value is None or not m:
                continue
            out[int(m.group(1))] = float(value)
        if out:
            return out
    return {}


def _fy(year: int) -> str:
    return f"FY{year} (year ended March {year})"


def _q(qid: str, category: str, symbol: str, question: str, answer: Any, unit: str, **extra) -> dict[str, Any]:
    return {
        "id": qid, "category": category, "ticker": f"{symbol}.NS",
        "question": question, "answer": answer, "unit": unit, **extra,
    }


def build(seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    symbols = sorted(s for s in NIFTY50_TICKERS if _load(s))
    data = {s: _load(s) for s in symbols}
    questions: list[dict[str, Any]] = []

    for symbol in symbols:
        d, name = data[symbol], NIFTY50_TICKERS[symbol]

        # ── lookup: one random available metric × recent year ────────────────
        options = [(m, _series(d, st, rows), unit) for m, st, rows, unit in LOOKUP_METRICS]
        options = [(m, s, u) for m, s, u in options if s]
        metric, series, unit = rng.choice(options)
        year = rng.choice(sorted(series)[-4:])
        questions.append(_q(
            f"lookup-{symbol}", "lookup", symbol,
            f"What was {name}'s {metric} in {_fy(year)}?",
            series[year], unit, metric=metric, year=year,
        ))

        # ── derived: one calculation per company, rotating through types ─────
        revenue = _series(d, "profit_loss", ["Sales", "Revenue"])
        profit = _series(d, "profit_loss", ["Net Profit"])
        equity = _series(d, "balance_sheet", ["Equity Capital"])
        reserves = _series(d, "balance_sheet", ["Reserves"])
        debt = _series(d, "balance_sheet", ["Borrowings", "Borrowing"])
        kind = ["growth", "margin", "de", "cagr"][len(questions) % 4]
        years = sorted(set(revenue) & set(profit))
        if kind == "growth" and len(years) >= 2:
            y = rng.choice(years[-3:])
            if y - 1 in profit and profit[y - 1] > 0:
                g = (profit[y] / profit[y - 1] - 1) * 100
                questions.append(_q(
                    f"derived-{symbol}", "derived", symbol,
                    f"By what percentage did {name}'s net profit grow in {_fy(y)} compared with the previous year?",
                    round(g, 2), "percent", calc="yoy_net_profit_growth", year=y,
                ))
                continue
        if kind == "de":
            ys = sorted(set(equity) & set(reserves) & set(debt))
            if ys:
                y = ys[-1]
                de = debt[y] / (equity[y] + reserves[y])
                questions.append(_q(
                    f"derived-{symbol}", "derived", symbol,
                    f"What was {name}'s debt-to-equity ratio (borrowings ÷ shareholders' equity) at the end of {_fy(y)}?",
                    round(de, 3), "ratio", calc="debt_to_equity", year=y,
                ))
                continue
        if kind == "cagr" and len(years) >= 6:
            y1, y0 = years[-1], years[-6]
            if revenue[y0] > 0:
                cagr = ((revenue[y1] / revenue[y0]) ** (1 / 5) - 1) * 100
                questions.append(_q(
                    f"derived-{symbol}", "derived", symbol,
                    f"What was {name}'s 5-year revenue CAGR from FY{y0} to FY{y1}?",
                    round(cagr, 2), "percent", calc="revenue_cagr_5y", year=y1,
                ))
                continue
        # margin is the fallback for every kind that could not be computed
        y = years[-1]
        if revenue[y] > 0:
            questions.append(_q(
                f"derived-{symbol}", "derived", symbol,
                f"What was {name}'s net profit margin (net profit ÷ revenue) in {_fy(y)}?",
                round(profit[y] / revenue[y] * 100, 2), "percent", calc="net_margin", year=y,
            ))

    # ── compare: 20 random pairs on net profit or revenue ────────────────────
    for i in range(20):
        a, b = rng.sample(symbols, 2)
        metric, rows = rng.choice([("net profit", ["Net Profit"]), ("revenue", ["Sales", "Revenue"])])
        sa, sb = _series(data[a], "profit_loss", rows), _series(data[b], "profit_loss", rows)
        common = sorted(set(sa) & set(sb))
        if not common:
            continue
        y = common[-1]
        if sa[y] == sb[y]:
            continue
        winner = a if sa[y] > sb[y] else b
        questions.append(_q(
            f"compare-{i:02d}", "compare", a,
            f"Which company had higher {metric} in {_fy(y)}: {NIFTY50_TICKERS[a]} or {NIFTY50_TICKERS[b]}?",
            winner, "ticker", other_ticker=f"{b}.NS",
            names={a: NIFTY50_TICKERS[a], b: NIFTY50_TICKERS[b]}, year=y,
        ))

    # ── unanswerable: future fiscal years — the right answer is to abstain ───
    for i, symbol in enumerate(rng.sample(symbols, 10)):
        year = rng.choice([2031, 2032, 2033])
        metric = rng.choice(["net profit", "revenue", "EPS"])
        questions.append(_q(
            f"unanswerable-{i:02d}", "unanswerable", symbol,
            f"What was {NIFTY50_TICKERS[symbol]}'s reported {metric} in {_fy(year)}?",
            None, "abstain", year=year,
        ))

    return questions


def save(questions: list[dict[str, Any]], path: str = BENCHMARK_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def load(path: str = BENCHMARK_PATH) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
