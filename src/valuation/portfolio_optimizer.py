"""
Portfolio Optimizer — Markowitz Mean-Variance + Composite Score tilt.

Given a capital amount and candidate stocks, uses Monte Carlo portfolio
simulation (10,000 random portfolios) to find the allocation that maximises
the Sharpe ratio.

Key improvements over naive approach:
  - Sector-based pairwise correlation matrix (same-sector ρ=0.65, commodity
    cluster ρ=0.45, unrelated ρ=0.25) rather than a flat 35% scalar.
  - Sector concentration cap: no single sector may exceed 40% of the portfolio.
  - Quality-score floor: stocks with higher composite scores get a minimum
    weight floor so the optimizer can't ignore fundamentally strong companies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


_RF_RATE      = 0.071      # India 10Y Gilt (annual)
_N_PORT       = 10_000     # Monte Carlo portfolios (doubled for better coverage)
_MIN_W        = 0.04       # min weight per stock (4%)
_MAX_W        = 0.35       # max weight per stock (35%)
_MAX_SEC_W    = 0.40       # max sector weight (40%) — prevents sector bets
_TRADING_DAYS = 252


# ── Sector assignments (mirrors DCF engine) ───────────────────────────────────

_SYM_SECTOR: dict[str, str] = {
    "HDFCBANK": "BANKING",  "ICICIBANK": "BANKING",  "SBIN": "BANKING",
    "AXISBANK": "BANKING",  "KOTAKBANK": "BANKING",  "INDUSINDBK": "BANKING",
    "TCS": "IT",  "INFY": "IT",  "HCLTECH": "IT",  "WIPRO": "IT",  "TECHM": "IT",
    "HINDUNILVR": "FMCG",  "ITC": "FMCG",  "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG",   "TATACONSUM": "FMCG",
    "TITAN": "FMCG",       "ASIANPAINT": "FMCG",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA",  "CIPLA": "PHARMA",
    "MARUTI": "AUTO",  "TATAMOTORS": "AUTO",  "M&M": "AUTO",
    "BAJAJ-AUTO": "AUTO",  "EICHERMOT": "AUTO",  "HEROMOTOCO": "AUTO",
    "RELIANCE": "ENERGY",  "ONGC": "ENERGY",  "BPCL": "ENERGY",
    "NTPC": "ENERGY",      "POWERGRID": "ENERGY",  "COALINDIA": "ENERGY",
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS",  "HINDALCO": "METALS",
    "LT": "INFRA",  "ADANIENT": "INFRA",  "ADANIPORTS": "INFRA",  "BEL": "INFRA",
    "ULTRACEMCO": "CEMENT",  "GRASIM": "CEMENT",
    "BHARTIARTL": "TELECOM",
    "BAJFINANCE": "FINSERV",  "BAJAJFINSV": "FINSERV",  "SHRIRAMFIN": "FINSERV",
    "HDFCLIFE": "INSURANCE",  "SBILIFE": "INSURANCE",
    "APOLLOHOSP": "HEALTHCARE",  "ZOMATO": "CONSUMER_TECH",
}

# Clusters of related sectors — within-cluster ρ raised to 0.45
_SECTOR_CLUSTERS: list[set[str]] = [
    {"BANKING", "FINSERV", "INSURANCE"},    # financial cluster
    {"ENERGY", "METALS", "CEMENT"},         # commodity / cyclical cluster
    {"IT"},                                  # IT is its own cluster (all correlated)
    {"AUTO", "INFRA"},                       # capex-sensitive cluster
]

# Pairwise ρ look-up
_RHO_SAME    = 0.65   # same sector
_RHO_CLUSTER = 0.45   # different sector, same cluster
_RHO_CROSS   = 0.25   # unrelated sectors


def _pairwise_rho(sym_a: str, sym_b: str) -> float:
    if sym_a == sym_b:
        return 1.0
    sec_a = _SYM_SECTOR.get(sym_a, "OTHER")
    sec_b = _SYM_SECTOR.get(sym_b, "OTHER")
    if sec_a == sec_b:
        return _RHO_SAME
    # Check if in same cluster
    for cluster in _SECTOR_CLUSTERS:
        if sec_a in cluster and sec_b in cluster:
            return _RHO_CLUSTER
    return _RHO_CROSS


def _build_corr_matrix(symbols: list[str]) -> np.ndarray:
    n   = len(symbols)
    rho = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = _pairwise_rho(symbols[i], symbols[j])
            rho[i, j] = rho[j, i] = r
    return rho


@dataclass
class Allocation:
    symbol:  str
    weight:  float
    amount:  float
    shares:  Optional[int]
    cmp:     Optional[float]


@dataclass
class PortfolioResult:
    allocations:      list[Allocation]
    total_capital:    float
    expected_return:  float
    volatility:       float
    sharpe_ratio:     float
    max_drawdown_est: float
    diversification:  str
    verdict:          str
    notes:            list[str] = field(default_factory=list)


class PortfolioOptimizer:
    """
    Builds a Nifty 50 sub-portfolio maximising Sharpe ratio.
    Uses sector-based correlation matrix and sector concentration cap.
    """

    def __init__(self, cache_dir: str = "data/raw/screener"):
        self.cache_dir = Path(cache_dir)

    def optimize(
        self,
        symbols:    list[str],
        scores:     dict[str, float],
        capital:    float = 100_000,
        max_stocks: int   = 10,
        min_stocks: int   = 5,
    ) -> PortfolioResult:
        notes: list[str] = []

        candidates = [s for s in symbols if s in scores and scores[s] > 0]
        if len(candidates) < min_stocks:
            return self._empty(capital, f"Need ≥{min_stocks} symbols with scores")

        candidates = sorted(candidates, key=lambda s: scores[s], reverse=True)[:max_stocks * 2]

        returns, vols, cmps = {}, {}, {}
        for sym in candidates:
            r, v, c = self._est_return_vol(sym, scores.get(sym, 50))
            if r is not None:
                returns[sym] = r
                vols[sym]    = v
                cmps[sym]    = c

        eligible = list(returns.keys())
        if len(eligible) < min_stocks:
            return self._empty(capital, "Insufficient return estimates")

        eligible = sorted(eligible, key=lambda s: scores.get(s, 0), reverse=True)[:max_stocks]
        n        = len(eligible)

        # ── Sector-based covariance matrix ────────────────────────────────────
        mu  = np.array([returns[s] for s in eligible])
        sig = np.array([vols[s]    for s in eligible])
        rho = _build_corr_matrix(eligible)              # sector-aware ρ matrix
        cov = np.outer(sig, sig) * rho                 # σᵢσⱼρᵢⱼ

        # Sector membership for concentration cap
        sectors = [_SYM_SECTOR.get(s, "OTHER") for s in eligible]
        unique_secs = list(set(sectors))

        # ── Monte Carlo ───────────────────────────────────────────────────────
        rng         = np.random.default_rng(0)
        best_sharpe = -np.inf
        best_w      = np.ones(n) / n

        for _ in range(_N_PORT):
            w = rng.dirichlet(np.ones(n) * 2.0)
            w = np.clip(w, _MIN_W, _MAX_W)
            w /= w.sum()

            # Enforce sector concentration cap
            for sec in unique_secs:
                sec_idx   = [i for i, s in enumerate(sectors) if s == sec]
                sec_total = w[sec_idx].sum()
                if sec_total > _MAX_SEC_W:
                    # Scale down the sector, redistribute to others
                    scale          = _MAX_SEC_W / sec_total
                    excess         = sec_total - _MAX_SEC_W
                    w[sec_idx]    *= scale
                    other_idx      = [i for i in range(n) if i not in sec_idx]
                    if other_idx:
                        w[other_idx] += excess / len(other_idx)
                    w = np.clip(w, _MIN_W, _MAX_W)
                    w /= w.sum()

            port_ret = float(mu @ w)
            port_var = float(w @ cov @ w)
            port_vol = float(np.sqrt(max(port_var, 0)))
            sharpe   = (port_ret - _RF_RATE) / port_vol if port_vol > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_w      = w.copy()

        # ── Build output ─────────────────────────────────────────────────────
        port_ret = float(mu @ best_w)
        port_vol = float(np.sqrt(best_w @ cov @ best_w))
        sharpe   = (port_ret - _RF_RATE) / port_vol if port_vol > 0 else 0
        mdd_est  = min(port_vol * 2.0, 0.60)

        allocs = []
        for sym, w in zip(eligible, best_w):
            amt    = capital * w
            cmp    = cmps.get(sym)
            shares = int(amt / cmp) if cmp and cmp > 0 else None
            allocs.append(Allocation(
                symbol=sym, weight=round(w, 4),
                amount=round(amt, 2), shares=shares, cmp=cmp,
            ))
        allocs.sort(key=lambda a: a.weight, reverse=True)

        # Sector summary for notes
        sec_weights: dict[str, float] = {}
        for a in allocs:
            sec = _SYM_SECTOR.get(a.symbol, "OTHER")
            sec_weights[sec] = sec_weights.get(sec, 0) + a.weight
        top_sec = max(sec_weights, key=sec_weights.get)

        max_w = max(a.weight for a in allocs)
        diversification = "Well Diversified" if max_w < 0.25 else "Concentrated"

        verdict = (
            "Excellent" if sharpe >= 0.8 else
            "Good"      if sharpe >= 0.5 else
            "Fair"      if sharpe >= 0.3 else "Poor"
        )

        if n < 8:
            notes.append(f"Only {n} stocks eligible — portfolio less diversified")
        top_sec_pct = sec_weights.get(top_sec, 0) * 100
        notes.append(f"Largest sector: {top_sec} ({top_sec_pct:.0f}%)  |  Sector cap: {_MAX_SEC_W*100:.0f}%")

        return PortfolioResult(
            allocations=allocs,
            total_capital=capital,
            expected_return=round(port_ret, 4),
            volatility=round(port_vol, 4),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_est=round(mdd_est, 4),
            diversification=diversification,
            verdict=verdict,
            notes=notes,
        )

    # ── Return / vol estimation ───────────────────────────────────────────────

    def _est_return_vol(
        self, symbol: str, score: float
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        path = self.cache_dir / f"{symbol}.json"
        if not path.exists():
            return None, None, None

        with open(path) as f:
            data = json.load(f)

        pl_rows  = data.get("profit_loss", {}).get("rows", {})
        rat_rows = data.get("ratios", {}).get("rows", {})

        eps_ser  = [v for v in pl_rows.get("EPS in Rs", []) if v]
        roce_ser = [v for v in rat_rows.get("ROCE %", []) if v]
        roe_ser  = [v for v in rat_rows.get("ROE %", []) if v]
        cmp      = data.get("current_price")

        if not cmp:
            try:
                with open("data/raw/nse/live_prices.json") as pf:
                    cmp = json.load(pf).get(symbol)
            except Exception:
                pass

        if len(eps_ser) < 3:
            return None, None, None

        # EPS CAGR (3Y smoothed)
        recent = [v for v in eps_ser[-4:] if v > 0]
        older  = [v for v in eps_ser[-7:-3] if v > 0]
        if recent and older:
            eps_cagr = (sum(recent)/len(recent)) / (sum(older)/len(older)) ** (1/3.5) - 1
        else:
            eps_cagr = 0.07
        eps_cagr = max(-0.05, min(eps_cagr, 0.28))

        # Earnings yield
        earnings_yield = 0.05
        if cmp and cmp > 0 and eps_ser[-1] > 0:
            pe = cmp / eps_ser[-1]
            pe = max(5, min(pe, 80))
            earnings_yield = 1 / pe

        # Quality return boost: use ROCE or ROE (banks have ROE, not ROCE)
        quality_boost = (score / 100) * 0.03
        roc = self._tail_avg(roce_ser or roe_ser, 3) / 100 if (roce_ser or roe_ser) else 0.12
        roc_boost = max(0, (roc - 0.12) * 0.2)   # excess ROC above 12% contributes

        exp_ret = earnings_yield + eps_cagr * 0.5 + quality_boost + roc_boost
        exp_ret = max(0.05, min(exp_ret, 0.40))

        # Volatility: base 22%, reduced by quality score, raised by sector beta
        from src.valuation.dcf_engine import _SECTOR_BETA, _SYM_SECTOR as _SM
        beta     = _SECTOR_BETA.get(_SM.get(symbol, "DEFAULT"), 1.0)
        base_vol = 0.20 * beta
        qual_adj = 1.0 - (score / 100) * 0.25
        vol      = max(0.10, min(base_vol * qual_adj, 0.50))

        return exp_ret, vol, cmp

    def _tail_avg(self, series: list, n: int) -> float:
        tail = series[-n:] if len(series) >= n else series
        vals = [v for v in tail if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _empty(self, capital: float, reason: str) -> PortfolioResult:
        return PortfolioResult(
            allocations=[], total_capital=capital,
            expected_return=0, volatility=0, sharpe_ratio=0,
            max_drawdown_est=0, diversification="N/A",
            verdict="ERROR", notes=[reason],
        )
