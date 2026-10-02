"""Terminal report for a BacktestResult."""
from __future__ import annotations

from typing import Optional

from src.backtest.engine import BacktestResult


def _pct(x: Optional[float], signed: bool = True) -> str:
    if x is None:
        return "—"
    return f"{x*100:+.1f}%" if signed else f"{x*100:.1f}%"


def _num(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:+.2f}"


_FACTOR_LABELS = {
    "quality":          "Quality (35%)",
    "growth":           "Growth (30%)",
    "earnings_quality": "Earnings quality (20%)",
    "piotroski":        "Piotroski (15%)",
    "roce":             "  ROCE / ROE",
    "margin":           "  Op. margin / NIM",
    "low_debt":         "  Low debt",
    "revenue_growth":   "  Revenue CAGR 3y",
    "eps_growth":       "  EPS CAGR 3y",
    "fcf_growth":       "  FCF CAGR 3y",
    "earnings_yield":   "Earnings yield (ref.)",
}


def _print_factor_table(result: BacktestResult, s: dict) -> None:
    years = [p.when.year for p in result.periods]
    print()
    print("\033[1m  WHAT DRIVES THE COMPOSITE SCORE — rank IC per factor\033[0m")
    print("  " + "─" * (24 + 7 * len(years) + 24))
    print(f"  {'Factor':<24}" + "".join(f"{y:>7}" for y in years) + f"{'Mean':>8}{'t':>8}{'+ve':>8}")

    m, t = s["score_ic"]
    pos, n = s["score_ic_positive"]
    print(f"  {'Composite':<24}" + "".join(f"{_num(p.score_ic):>7}" for p in result.periods)
          + f"{_num(m):>8}{_num(t):>8}{f'{pos}/{n}':>8}")
    for name, label in _FACTOR_LABELS.items():
        m, t, pos, n = s["factor_ic"].get(name, (None, None, 0, 0))
        cells = "".join(f"{_num(p.factor_ic.get(name)):>7}" for p in result.periods)
        print(f"  {label:<24}{cells}{_num(m):>8}{_num(t):>8}{f'{pos}/{n}':>8}")
    print("  Weights are the pillar's share of the composite. Each factor is oriented the way the")
    print("  score uses it, so a positive IC means it helped; earnings yield is not in the score.")


def print_backtest(result: BacktestResult) -> None:
    print()
    print(f"\033[1m  WALK-FORWARD BACKTEST — {result.horizon_days}-day forward returns\033[0m")
    print("  " + "─" * 96)
    print(f"  {'Date':<11}{'N':>4}{'Universe':>10}{'Nifty50':>10}{'Top 20%':>10}"
          f"{'Score IC':>10}{'UV n':>6}{'UV ret':>10}{'UV hit':>9}{'DCF IC':>9}")
    for p in result.periods:
        print(f"  {p.when.isoformat():<11}{p.n:>4}{_pct(p.universe_return):>10}"
              f"{_pct(p.benchmark_return):>10}{_pct(p.top_quintile_return):>10}"
              f"{_num(p.score_ic):>10}{p.undervalued_n:>6}{_pct(p.undervalued_return):>10}"
              f"{_pct(p.undervalued_hit_rate, signed=False):>9}{_num(p.dcf_ic):>9}")

    s = result.summary()
    if not s["periods"]:
        print("\n  No periods could be evaluated.")
    else:
        def growth(key):
            g = s[key]
            return "—" if g is None else f"₹1 → ₹{g:.2f}"

        print("  " + "─" * 96)
        for label, key in (("Composite score", "score_ic"), ("DCF P(undervalued)", "dcf_ic")):
            m, t = s[key]
            pos, n = s[f"{key}_positive"]
            print(f"  {label:<20} mean IC {_num(m)}   t-stat {_num(t)}   positive in {pos}/{n} periods")
        print(f"  Equal-weight universe  {growth('universe_growth')}")
        print(f"  Nifty 50 (price only)  {growth('benchmark_growth')}")
        print(f"  Top-score quintile     {growth('top_quintile_growth')}")
        print(f"  DCF undervalued        {growth('undervalued_growth')}")
        rate, n_uv = s["undervalued_hit_rate"]
        print(f"  Undervalued hit rate   {_pct(rate, signed=False)} of {n_uv} calls beat the universe")

    if s["periods"]:
        _print_factor_table(result, s)

    print()
    print("  UV = names with P(value > price) > 50%.  IC = Spearman rank correlation with forward return.")
    print("  With few periods a t-stat below ~2 is not evidence either way.")
    for line in result.skipped:
        print(f"  \033[93mSkipped {line}\033[0m")
    print("\n  Caveats:")
    for c in result.caveats:
        print(f"   • {c}")
    print()
