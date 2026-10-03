"""
The model's backtested track record, saved by `main.py backtest` and stated in
every report that gives a valuation or quality view — so a reader sees how far
the signal has been worth trusting, not just the signal.

Sentences are generated from the numbers with fixed thresholds, never by an
LLM: a signal is called reliable only with a mean IC above zero and t ≥ 2.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from src.backtest.engine import BacktestResult

TRACK_RECORD_PATH = Path("data/processed/track_record.json")


def save_track_record(result: BacktestResult, path: Optional[Path] = None) -> Optional[Path]:
    path = path or TRACK_RECORD_PATH
    s = result.summary()
    if not s["periods"]:
        return None
    years = [p.when.year for p in result.periods]
    score_m, score_t = s["score_ic"]
    dcf_m, dcf_t     = s["dcf_ic"]
    hit, n_calls     = s["undervalued_hit_rate"]
    record = {
        "generated": date.today().isoformat(),
        "first_year": min(years), "last_year": max(years), "periods": s["periods"],
        "horizon_days": result.horizon_days,
        "universe_growth":     s["universe_growth"],
        "score": {"mean_ic": score_m, "t_stat": score_t,
                  "positive": s["score_ic_positive"][0], "top_quintile_growth": s["top_quintile_growth"]},
        "dcf":   {"mean_ic": dcf_m, "t_stat": dcf_t, "positive": s["dcf_ic_positive"][0],
                  "undervalued_growth": s["undervalued_growth"], "hit_rate": hit, "calls": n_calls},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return path


def load_track_record(path: Optional[Path] = None) -> Optional[dict]:
    try:
        return json.loads((path or TRACK_RECORD_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _reliability(mean_ic: Optional[float], t: Optional[float]) -> str:
    if mean_ic is None:
        return "could not be measured"
    if mean_ic > 0 and t is not None and t >= 2:
        return "has predicted returns with statistical significance"
    if mean_ic > 0:
        return "has leaned the right way, but not enough to rule out luck"
    return "has not predicted returns"


def track_record_lines(record: Optional[dict]) -> list[str]:
    """Plain-English statements of how each signal has performed."""
    if not record:
        return ["This model has not yet been backtested, so its valuation and quality "
                "views are unvalidated."]
    span = f"{record['first_year']}–{record['last_year']} ({record['periods']} yearly periods)"
    u, sc, dc = record["universe_growth"], record["score"], record["dcf"]
    lines = [
        f"Quality score — backtested {span}: it {_reliability(sc['mean_ic'], sc['t_stat'])}. "
        f"The highest-scoring fifth of stocks turned Rs 1 into Rs {sc['top_quintile_growth']:.2f}, "
        f"against Rs {u:.2f} for holding every stock equally. Read the score as a description "
        f"of the business, not as a buy signal.",
    ]
    if dc["calls"]:
        lines.append(
            f"Valuation (DCF) — same period: it {_reliability(dc['mean_ic'], dc['t_stat'])}. "
            f"Stocks it rated likely undervalued beat the equal-weight average in "
            f"{dc['hit_rate'] * 100:.0f}% of {dc['calls']} calls.")
    lines.append("Backtest universe is today's Nifty 50, which flatters past returns "
                 "(survivorship bias); figures are as later restated.")
    return lines
