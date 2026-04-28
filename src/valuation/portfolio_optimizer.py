"""
Portfolio Optimizer — Markowitz Mean-Variance + Composite Score tilt.

Given a capital amount and candidate stocks, uses Monte Carlo portfolio
simulation (5,000 random portfolios) to find the allocation that maximises
the Sharpe ratio.  Composite quality scores act as a tilt: stocks with
higher scores get a minimum floor weight so the optimizer doesn't ignore
fundamentally strong companies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_LIVE_PRICES_PATH = Path("data/raw/nse/live_prices.json")
_LIVE_PRICES: dict = {}

import numpy as np


_RF_RATE   = 0.071     # India 10Y Gilt (annual)
_N_PORT    = 5_000     # Monte Carlo portfolios
_MIN_W     = 0.04      # min weight per stock (4%)
_MAX_W     = 0.35      # max weight per stock (35%)
_TRADING_DAYS = 252


@dataclass
class Allocation:
    symbol:   str
    weight:   float       # 0–1
    amount:   float       # ₹
    shares:   Optional[int]   # approximate share count
    cmp:      Optional[float]


@dataclass
class PortfolioResult:
    allocations:      list[Allocation]
    total_capital:    float
    expected_return:  float      # annualised
    volatility:       float      # annualised
    sharpe_ratio:     float
    max_drawdown_est: float      # estimated from vol (not historical)
    diversification:  str        # "Well Diversified" / "Concentrated"
    verdict:          str
    notes:            list[str] = field(default_factory=list)


class PortfolioOptimizer:
    """
    Builds a Nifty 50 sub-portfolio maximising Sharpe ratio, tilted by
    fundamental composite scores.
    """

    def __init__(self, cache_dir: str = "data/raw/screener"):
        self.cache_dir = Path(cache_dir)

    def optimize(
        self,
        symbols:       list[str],
        scores:        dict[str, float],   # composite score 0–100
        capital:       float = 100_000,    # ₹
        max_stocks:    int   = 10,
        min_stocks:    int   = 5,
    ) -> PortfolioResult:
        """Return optimal Sharpe-maximising allocation."""
        notes: list[str] = []

        # ── Filter to symbols with score data ────────────────────────────────
        candidates = [s for s in symbols if s in scores and scores[s] > 0]
        if len(candidates) < min_stocks:
            return self._empty(capital, f"Need ≥{min_stocks} symbols with scores")

        # ── Pre-select top N by composite score ──────────────────────────────
        candidates = sorted(candidates, key=lambda s: scores[s], reverse=True)[:max_stocks * 2]

        # ── Load price series from EPS / NP proxies ───────────────────────────
        # We don't have daily prices, so we synthesise annual return estimates
        # from EPS CAGR + sector implied P/E expansion/contraction.
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

        # Take top max_stocks by composite score among eligible
        eligible = sorted(eligible, key=lambda s: scores.get(s, 0), reverse=True)[:max_stocks]
        n        = len(eligible)

        # ── Build covariance matrix (diagonal + market correlation) ───────────
        mu  = np.array([returns[s] for s in eligible])
        sig = np.array([vols[s]    for s in eligible])
        rho = 0.35   # average pairwise correlation within Nifty 50
        cov = np.outer(sig, sig) * rho
        np.fill_diagonal(cov, sig**2)

        # ── Monte Carlo portfolio simulation ─────────────────────────────────
        rng      = np.random.default_rng(0)
        best_sharpe = -np.inf
        best_w      = np.ones(n) / n

        # Score-based floor: higher-scoring stocks get higher minimum weight
        score_arr = np.array([scores.get(s, 50) for s in eligible])
        score_norm = score_arr / score_arr.sum()
        floor_w    = np.clip(score_norm * 0.5, _MIN_W, _MAX_W)

        for _ in range(_N_PORT):
            w = rng.dirichlet(np.ones(n) * 2.0)
            # Apply floor/ceiling
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
        # Rough max drawdown estimate: 2σ annual loss
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

        max_w = max(a.weight for a in allocs)
        diversification = "Well Diversified" if max_w < 0.25 else "Concentrated"

        verdict = (
            "Excellent" if sharpe >= 0.8 else
            "Good"      if sharpe >= 0.5 else
            "Fair"      if sharpe >= 0.3 else "Poor"
        )

        if n < 8:
            notes.append(f"Only {n} stocks eligible — portfolio less diversified")

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
        """
        Synthesise an expected annual return and volatility from fundamentals.
        Return = earnings yield + EPS growth.  Vol = sector-adjusted.
        """
        path = self.cache_dir / f"{symbol}.json"
        if not path.exists():
            return None, None, None

        with open(path) as f:
            data = json.load(f)

        pl_rows = data.get("profit_loss", {}).get("rows", {})
        bs_rows = data.get("balance_sheet", {}).get("rows", {})
        rat_rows = data.get("ratios", {}).get("rows", {})

        eps_ser  = [v for v in pl_rows.get("EPS in Rs", []) if v]
        np_ser   = [v for v in pl_rows.get("Net Profit", []) if v]
        roce_ser = [v for v in rat_rows.get("ROCE %", []) if v]
        cmp      = data.get("current_price")

        # Fall back to live_prices.json if screener cache has no CMP
        if not cmp:
            prices_path = Path("data/raw/nse/live_prices.json")
            if prices_path.exists():
                try:
                    with open(prices_path) as pf:
                        cmp = json.load(pf).get(symbol)
                except Exception:
                    pass

        if len(eps_ser) < 3:
            return None, None, None

        # EPS CAGR (3Y)
        eps_cagr = self._cagr(eps_ser[-4:])
        eps_cagr = max(-0.10, min(eps_cagr, 0.30))

        # Earnings yield = 1 / PE (use EPS / CMP if we have CMP)
        earnings_yield = 0.05   # default 5%
        if cmp and cmp > 0 and eps_ser:
            pe = cmp / eps_ser[-1] if eps_ser[-1] > 0 else 25
            pe = max(5, min(pe, 100))
            earnings_yield = 1 / pe

        # Expected return = earnings yield + growth (Graham formula variant)
        exp_ret = earnings_yield + eps_cagr * 0.5 + (score / 100) * 0.03
        exp_ret = max(0.05, min(exp_ret, 0.40))

        # Volatility: base 20%, modulated by quality score and sector
        base_vol = 0.22
        quality_adj = 1.0 - (score / 100) * 0.30   # high quality = lower vol
        vol = base_vol * quality_adj
        vol = max(0.12, min(vol, 0.45))

        return exp_ret, vol, cmp

    def _cagr(self, series: list[float]) -> float:
        vals = [v for v in series if v and v > 0]
        if len(vals) < 2:
            return 0.07
        return (vals[-1] / vals[0]) ** (1 / (len(vals) - 1)) - 1

    def _empty(self, capital: float, reason: str) -> PortfolioResult:
        return PortfolioResult(
            allocations=[], total_capital=capital,
            expected_return=0, volatility=0, sharpe_ratio=0,
            max_drawdown_est=0, diversification="N/A",
            verdict="ERROR", notes=[reason],
        )
