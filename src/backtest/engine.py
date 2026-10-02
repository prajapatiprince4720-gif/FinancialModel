"""
Walk-forward backtest of the composite score and the probabilistic DCF.

On each rebalance date every company is seen as an investor could have seen it
(CompanyData.as_of: only fiscal years already published, the price of that
day), scored and valued, and then compared with its dividend-inclusive return
over the following horizon. Annual dates with a one-year horizon keep the
periods non-overlapping, so each year is an independent observation.

What is measured, per date:
  score IC          Spearman rank correlation of composite score vs forward return
  top quintile      mean forward return of the highest-scoring fifth
  undervalued       names with P(value > price) > 50%: mean return, and the share
                    of them that beat the equal-weight universe (hit rate)
  DCF IC            Spearman correlation of P(undervalued) vs forward return
  factor IC         the same for each pillar and sub-factor of the composite,
                    plus earnings yield as a cheapness reference, also split into
                    between sectors (did high-signal sectors win?) and within
                    sectors (did the high-signal company beat its sector?)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import mean, stdev
from typing import Callable, Optional

import pandas as pd

from src.backtest.prices import BENCHMARK, PriceHistory
from src.data.company import CompanyData, load_company
from src.data.sectors import sector_of
from src.scoring.composite_score import CompositeScorer, ScoreResult
from src.utils.macro_rates import RFContext
from src.valuation.dcf_engine import DCFEngine

# The demerger split one price history across two listings, so returns across
# it describe neither the old company nor the new ones.
EXCLUDED = {"TATAMOTORS": "2025 demerger breaks the price history"}

# India 10Y G-Sec yield around the start of June. APPROXIMATE — recalled
# levels, not pulled from a data source; verify against RBI/CCIL before
# relying on DCF results. Each rebalance date uses the latest year ≤ its own.
HISTORICAL_RF: dict[int, float] = {
    2016: 0.0750, 2017: 0.0665, 2018: 0.0785, 2019: 0.0700, 2020: 0.0580,
    2021: 0.0600, 2022: 0.0745, 2023: 0.0697, 2024: 0.0698, 2025: 0.0625,
}

CAVEATS = [
    "Survivorship bias: the universe is today's Nifty 50, so past winners are "
    "over-represented and absolute returns are flattered.",
    "Screener shows figures as later restated, not as first reported.",
    "Historical risk-free rates are approximate June levels, not exact yields.",
    "The Nifty 50 benchmark is a price index (no dividends); stock returns include dividends.",
]


# The composite's pillars and main sub-factors, each oriented so a higher value
# is what the score rewards: a positive IC means the factor did its job.
SCORE_FACTORS: dict[str, Callable[[ScoreResult], float]] = {
    "quality":          lambda r: r.quality,
    "growth":           lambda r: r.growth,
    "earnings_quality": lambda r: r.earnings_quality,
    "piotroski":        lambda r: r.piotroski_pct,
    "roce":             lambda r: r.roce_pct,          # peer-ranked; banks → ROE
    "margin":           lambda r: r.opm_pct,           # peer-ranked; banks → NIM
    "low_debt":         lambda r: -r.de_ratio,
    "revenue_growth":   lambda r: r.revenue_cagr_3y,
    "eps_growth":       lambda r: r.eps_cagr_3y,
    "fcf_growth":       lambda r: r.fcf_cagr_3y,
}
# Not part of the score: a cheapness reference, to test whether the score's
# misses are explained by paying up for quality.
REFERENCE_FACTORS = ("earnings_yield",)
SIGNALS = ("composite", *SCORE_FACTORS, *REFERENCE_FACTORS)


@dataclass
class Observation:
    when:             date
    symbol:           str
    price:            float
    score:            float
    prob_undervalued: Optional[float]
    margin_of_safety: Optional[float]
    fwd_return:       float
    factors:          dict[str, float] = field(default_factory=dict)
    sector:           str = "DEFAULT"


@dataclass
class PeriodStats:
    when:                date
    n:                   int
    universe_return:     float            # equal-weight, dividends included
    benchmark_return:    Optional[float]  # Nifty 50 price index
    score_ic:            Optional[float]
    top_quintile_return: Optional[float]
    undervalued_n:       int
    undervalued_hits:    int              # undervalued names that beat the universe
    undervalued_return:  Optional[float]
    dcf_ic:              Optional[float]
    # Per signal ("composite" and each factor): IC across all stocks, across
    # sector averages, and within sectors (see _between_sector_ic / _within_sector_ic).
    factor_ic:           dict[str, Optional[float]] = field(default_factory=dict)
    factor_ic_between:   dict[str, Optional[float]] = field(default_factory=dict)
    factor_ic_within:    dict[str, Optional[float]] = field(default_factory=dict)

    @property
    def undervalued_hit_rate(self) -> Optional[float]:
        return self.undervalued_hits / self.undervalued_n if self.undervalued_n else None


@dataclass
class BacktestResult:
    horizon_days: int
    observations: list[Observation]
    periods:      list[PeriodStats]
    skipped:      list[str] = field(default_factory=list)
    caveats:      list[str] = field(default_factory=lambda: list(CAVEATS))

    def summary(self) -> dict:
        """Aggregates across periods: mean IC with t-stat, compounded returns, hit rate."""
        ps = self.periods
        out: dict = {"periods": len(ps)}
        for name in ("score_ic", "dcf_ic"):
            vals = [getattr(p, name) for p in ps if getattr(p, name) is not None]
            out[name] = _mean_t(vals)
            out[f"{name}_positive"] = (sum(v > 0 for v in vals), len(vals))
        out["universe_growth"]     = _compound(p.universe_return for p in ps)
        out["benchmark_growth"]    = _compound(p.benchmark_return for p in ps)
        out["top_quintile_growth"] = _compound(p.top_quintile_return for p in ps)
        out["undervalued_growth"]  = _compound(
            p.undervalued_return if p.undervalued_return is not None else p.universe_return
            for p in ps)     # no undervalued names that year → hold the universe
        hits = sum(p.undervalued_hits for p in ps)
        n_uv = sum(p.undervalued_n for p in ps)
        out["undervalued_hit_rate"] = (hits / n_uv, n_uv) if n_uv else (None, 0)

        # key → name → (mean IC, t-stat, periods positive, periods measured)
        for key in ("factor_ic", "factor_ic_between", "factor_ic_within"):
            out[key] = {}
            for name in SIGNALS:
                vals = [v for p in ps if (v := getattr(p, key).get(name)) is not None]
                m, t = _mean_t(vals)
                out[key][name] = (m, t, sum(v > 0 for v in vals), len(vals))
        return out


def rebalance_dates(start_year: int, end_year: int, month: int = 6, day: int = 1) -> list[date]:
    """One date a year; June 1 is just after March fiscal-year results are due."""
    return [date(y, month, day) for y in range(start_year, end_year + 1)]


def historical_rf(when: date) -> RFContext:
    years = [y for y in HISTORICAL_RF if y <= when.year]
    year  = max(years) if years else min(HISTORICAL_RF)
    return RFContext(rate=HISTORICAL_RF[year], source="manual", as_of=f"{year}-06-01",
                     is_live=False, detail="approximate historical 10Y G-Sec (backtest)")


def run_backtest(dates: list[date], symbols: Optional[list[str]] = None,
                 horizon_days: int = 365, min_years: int = 4,
                 prices: Optional[PriceHistory] = None,
                 cache_dir: str = "data/raw/screener",
                 refresh_prices: bool = False) -> BacktestResult:
    """
    Score and value every symbol as of each date, then measure the next
    `horizon_days`. A symbol is left out of a date when it has fewer than
    `min_years` published fiscal years (3-year CAGRs need four), no price that
    day, or no price at the end of the horizon.
    """
    if symbols is None:
        from config.nifty50_tickers import NIFTY50_TICKERS
        symbols = list(NIFTY50_TICKERS)
    symbols = [s for s in symbols if s not in EXCLUDED]

    companies = {s: cd for s in symbols if (cd := load_company(s, cache_dir=cache_dir)) is not None}
    if prices is None:
        end = min(max(dates) + timedelta(days=horizon_days + 10), date.today())
        prices = PriceHistory.load(list(companies), min(dates) - timedelta(days=10), end,
                                   refresh=refresh_prices)

    scorer = CompositeScorer(cache_dir=cache_dir)
    observations: list[Observation] = []
    periods: list[PeriodStats] = []
    skipped: list[str] = []

    for when in dates:
        until = when + timedelta(days=horizon_days)
        if prices.last_date is None or until > prices.last_date:
            skipped.append(f"{when}: horizon ends {until}, after the last price")
            continue

        views:   dict[str, CompanyData] = {}
        returns: dict[str, float] = {}
        for sym, cd in companies.items():
            price = prices.close_on(sym, when)
            if price is None:
                continue
            view = cd.as_of(when, price=price)
            if len(view.years("profit_loss")) < min_years:
                continue
            fwd = prices.total_return(sym, when, until)
            if fwd is None:
                continue
            views[sym], returns[sym] = view, fwd
        if len(views) < 5:
            skipped.append(f"{when}: only {len(views)} companies with enough history")
            continue

        scores = {r.symbol: r for r in scorer.score_all(list(views), data=views)}
        engine = DCFEngine(cache_dir=cache_dir, rf_context=historical_rf(when), quiet=True)
        period_obs = []
        for sym, view in views.items():
            if sym not in scores:
                continue
            r = engine.value(sym, data=view)
            ok = r.intrinsic_value > 0
            period_obs.append(Observation(
                when=when, symbol=sym, price=view.current_price, score=scores[sym].composite,
                prob_undervalued=r.prob_undervalued if ok else None,
                margin_of_safety=r.margin_of_safety if ok else None,
                fwd_return=returns[sym],
                factors=_factors(scores[sym], view),
                sector=sector_of(sym),
            ))
        observations.extend(period_obs)
        periods.append(_period_stats(when, period_obs,
                                     prices.total_return(BENCHMARK, when, until)))

    return BacktestResult(horizon_days, observations, periods, skipped)


def _factors(score: ScoreResult, view: CompanyData) -> dict[str, float]:
    out = {name: float(f(score)) for name, f in SCORE_FACTORS.items()}
    eps = view.series("profit_loss", "EPS in Rs")
    if eps and view.current_price:
        out["earnings_yield"] = eps[-1] / view.current_price
    return out


# ── Statistics ────────────────────────────────────────────────────────────────

def _period_stats(when: date, obs: list[Observation], benchmark: Optional[float]) -> PeriodStats:
    df = pd.DataFrame([{k: v for k, v in o.__dict__.items() if k != "factors"} for o in obs])
    universe = float(df["fwd_return"].mean())
    signals  = pd.DataFrame([o.factors for o in obs], index=df.index).assign(composite=df["score"])
    ret, sector = df["fwd_return"], df["sector"]

    k   = max(1, len(df) // 5)
    top = df.nlargest(k, "score")

    valued = df.dropna(subset=["prob_undervalued"])
    uv     = valued[valued["prob_undervalued"] > 0.5]

    return PeriodStats(
        when=when, n=len(df), universe_return=universe, benchmark_return=benchmark,
        score_ic=_spearman(df["score"], df["fwd_return"]),
        top_quintile_return=float(top["fwd_return"].mean()),
        undervalued_n=len(uv),
        undervalued_hits=int((uv["fwd_return"] > universe).sum()),
        undervalued_return=float(uv["fwd_return"].mean()) if len(uv) else None,
        dcf_ic=_spearman(valued["prob_undervalued"], valued["fwd_return"]),
        factor_ic={name: _spearman(col.dropna(), ret[col.dropna().index])
                   for name, col in signals.items()},
        factor_ic_between={name: _between_sector_ic(col, ret, sector) for name, col in signals.items()},
        factor_ic_within={name: _within_sector_ic(col, ret, sector) for name, col in signals.items()},
    )


def _between_sector_ic(x: pd.Series, ret: pd.Series, sector: pd.Series) -> Optional[float]:
    """
    Did sectors that score high on the signal outperform? Rank IC across sectors
    of the average percentile rank of the signal vs the equal-weight sector return.
    """
    d = pd.DataFrame({"x": x.rank(pct=True), "r": ret, "s": sector}).dropna()
    g = d.groupby("s")[["x", "r"]].mean()
    return _spearman(g["x"], g["r"])


def _within_sector_ic(x: pd.Series, ret: pd.Series, sector: pd.Series) -> Optional[float]:
    """
    Did the higher-scoring company beat its own sector? The signal is ranked within
    each sector and centred, the return is taken relative to the sector's mean, and
    the two are rank-correlated. One-company sectors carry no such information.
    """
    d = pd.DataFrame({"x": x, "r": ret, "s": sector}).dropna()
    d = d[d.groupby("s")["x"].transform("count") >= 2]
    if d.empty:
        return None
    rank = d.groupby("s")["x"].rank(pct=True)
    xw   = rank - rank.groupby(d["s"]).transform("mean")
    rw   = d["r"] - d.groupby("s")["r"].transform("mean")
    return _spearman(xw, rw)


def _spearman(x: pd.Series, y: pd.Series) -> Optional[float]:
    if len(x) < 5 or x.nunique() < 2:
        return None
    rho = x.corr(y, method="spearman")
    return None if pd.isna(rho) else float(rho)


def _mean_t(vals: list[float]) -> tuple[Optional[float], Optional[float]]:
    """Mean and its t-statistic (mean / standard error)."""
    if not vals:
        return None, None
    m = mean(vals)
    if len(vals) < 2 or stdev(vals) == 0:
        return m, None
    return m, m / (stdev(vals) / len(vals) ** 0.5)


def _compound(returns) -> Optional[float]:
    vals = [r for r in returns if r is not None]
    if not vals:
        return None
    g = 1.0
    for r in vals:
        g *= 1 + r
    return g
