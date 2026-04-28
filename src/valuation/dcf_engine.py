"""
DCF Valuation Engine with Monte Carlo simulation.

Uses 10 years of actual Screener.in financial data to compute intrinsic value
per share, margin of safety, and a confidence interval via 10,000 simulations.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_PRICE_CACHE: dict[str, Optional[float]] = {}
_PRICE_CACHE_FILE = Path("data/raw/nse/live_prices.json")


def _fetch_cmp(symbol: str) -> Optional[float]:
    """Get live price from yfinance (cached in memory + file for the session)."""
    if symbol in _PRICE_CACHE:
        return _PRICE_CACHE[symbol]

    # Try file cache first (refreshed if >4 hours old)
    if _PRICE_CACHE_FILE.exists():
        age = time.time() - _PRICE_CACHE_FILE.stat().st_mtime
        if age < 3600 * 4:
            try:
                with open(_PRICE_CACHE_FILE) as f:
                    file_cache = json.load(f)
                if symbol in file_cache:
                    _PRICE_CACHE[symbol] = file_cache[symbol]
                    return file_cache[symbol]
            except Exception:
                pass

    try:
        import yfinance as yf
        t = yf.Ticker(f"{symbol}.NS")
        info = t.fast_info
        price = float(info.last_price) if hasattr(info, "last_price") and info.last_price else None
        if price is None or price == 0:
            hist = t.history(period="1d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        price = None

    _PRICE_CACHE[symbol] = price

    # Persist to file
    try:
        _PRICE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if _PRICE_CACHE_FILE.exists():
            with open(_PRICE_CACHE_FILE) as f:
                existing = json.load(f)
        existing[symbol] = price
        with open(_PRICE_CACHE_FILE, "w") as f:
            json.dump(existing, f)
    except Exception:
        pass

    return price

# ── India macro constants ──────────────────────────────────────────────────────
_RF_RATE = 0.071          # India 10Y Gilt yield (risk-free rate)
_ERP     = 0.055          # India Equity Risk Premium
_TERM_G  = 0.055          # Terminal growth rate (India long-run nominal GDP)
_PROJ_YR = 10             # Projection horizon (years)
_MC_SIMS = 10_000         # Monte Carlo simulations

# Sector betas (systematic risk — sourced from NSE historical data)
_SECTOR_BETA: dict[str, float] = {
    "BANKING":    1.05,
    "IT":         0.80,
    "FMCG":       0.65,
    "PHARMA":     0.75,
    "AUTO":       1.20,
    "ENERGY":     1.10,
    "METALS":     1.40,
    "INFRA":      1.25,
    "TELECOM":    0.90,
    "INSURANCE":  0.85,
    "FINSERV":    1.10,
    "CEMENT":     1.15,
    "DEFAULT":    1.00,
}

_SYM_SECTOR: dict[str, str] = {
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING",
    "AXISBANK": "BANKING", "KOTAKBANK": "BANKING", "INDUSINDBK": "BANKING",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO",
    "BAJAJ-AUTO": "AUTO", "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO",
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY", "NTPC": "ENERGY",
    "POWERGRID": "ENERGY", "COALINDIA": "ENERGY",
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "HINDALCO": "METALS",
    "LT": "INFRA", "ULTRACEMCO": "CEMENT", "GRASIM": "CEMENT",
    "BHARTIARTL": "TELECOM",
    "BAJFINANCE": "FINSERV", "BAJAJFINSV": "FINSERV", "SHRIRAMFIN": "FINSERV",
    "HDFCLIFE": "INSURANCE", "SBILIFE": "INSURANCE",
    "ADANIENT": "INFRA", "ADANIPORTS": "INFRA",
    "TITAN": "FMCG", "ASIANPAINT": "FMCG", "BEL": "INFRA",
    "APOLLOHOSP": "DEFAULT", "ZOMATO": "DEFAULT",
}


@dataclass
class DCFResult:
    symbol:          str
    cmp:             Optional[float]       # Current Market Price (₹)
    intrinsic_value: float                 # Base-case DCF intrinsic value (₹/share)
    iv_low:          float                 # 10th-percentile MC intrinsic value
    iv_high:         float                 # 90th-percentile MC intrinsic value
    margin_of_safety: Optional[float]      # (IV - CMP) / CMP  — positive = undervalued
    wacc:            float
    fcf_cagr:        float                 # Historical FCF CAGR used
    terminal_value:  float                 # PV of terminal value (₹ Cr)
    pv_fcf:          float                 # PV of projected FCFs (₹ Cr)
    shares_cr:       float                 # Shares outstanding in crore
    verdict:         str                   # BUY / HOLD / SELL / INSUFFICIENT_DATA
    notes:           list[str] = field(default_factory=list)


class DCFEngine:
    """Computes DCF intrinsic value from cached Screener.in data."""

    def __init__(self, cache_dir: str = "data/raw/screener"):
        self.cache_dir = Path(cache_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def value(self, symbol: str) -> DCFResult:
        data = self._load(symbol)
        if data is None:
            return self._empty(symbol, "No cached data")

        pl   = data.get("profit_loss", {})
        bs   = data.get("balance_sheet", {})
        cf   = data.get("cash_flow", {})
        cmp  = data.get("current_price") or _fetch_cmp(symbol)

        years   = pl.get("years", [])
        pl_rows = pl.get("rows", {})
        bs_rows = bs.get("rows", {})
        cf_rows = cf.get("rows", {})

        # ── Extract key series (exclude TTM — last element may be partial) ──
        n = len(years)
        cut = n - 1 if years and "TTM" in str(years[-1]) else n

        def _series(rows: dict, key: str) -> list[float]:
            vals = rows.get(key, [])
            return [v for v in vals[:cut] if v is not None]

        fcf_series = _series(cf_rows, "Free Cash Flow")
        cfo_series = _series(cf_rows, "Cash from Operating Activity")
        np_series  = _series(pl_rows, "Net Profit")
        eps_series = _series(pl_rows, "EPS in Rs")
        sales_ser  = _series(pl_rows, "Sales")
        interest   = _series(pl_rows, "Interest")
        borr_ser   = _series(bs_rows, "Borrowings")
        eq_cap     = _series(bs_rows, "Equity Capital")
        reserves   = _series(bs_rows, "Reserves")
        tax_series = _series(pl_rows, "Tax %")

        notes: list[str] = []

        # ── Shares outstanding ────────────────────────────────────────────────
        shares_cr = self._shares(np_series, eps_series)
        if shares_cr <= 0:
            return self._empty(symbol, "Cannot determine share count")

        # ── WACC ─────────────────────────────────────────────────────────────
        beta        = _SECTOR_BETA.get(_SYM_SECTOR.get(symbol, "DEFAULT"), 1.0)
        cost_equity = _RF_RATE + beta * _ERP

        # Effective cost of debt from actuals
        avg_borr = self._tail_avg(borr_ser, 3)
        avg_int  = self._tail_avg(interest, 3)
        if avg_borr > 0 and avg_int > 0:
            cost_debt = avg_int / avg_borr
            cost_debt = max(0.05, min(cost_debt, 0.20))   # cap to sane range
        else:
            cost_debt = 0.085   # default: prime + spread

        avg_tax = self._tail_avg(tax_series, 3) / 100 if tax_series else 0.25
        avg_tax = max(0.0, min(avg_tax, 0.40))

        # Market value weights (use book equity as proxy)
        total_equity_cr = (eq_cap[-1] + reserves[-1]) if eq_cap and reserves else 0
        total_debt_cr   = borr_ser[-1] if borr_ser else 0
        total_v         = total_equity_cr + total_debt_cr
        if total_v <= 0:
            we, wd = 0.7, 0.3
        else:
            we = total_equity_cr / total_v
            wd = total_debt_cr  / total_v

        wacc = we * cost_equity + wd * cost_debt * (1 - avg_tax)
        wacc = max(0.08, min(wacc, 0.20))   # floor / ceiling

        # ── FCF selection ─────────────────────────────────────────────────────
        # Prefer FCF; fall back to CFO if FCF is consistently negative
        use_series = fcf_series if fcf_series else cfo_series
        tail       = use_series[-5:] if len(use_series) >= 5 else use_series

        pos_count = sum(1 for v in tail if v > 0)
        if pos_count < 2:
            # Use CFO instead — FCF too erratic (heavy capex cycle)
            use_series = cfo_series if cfo_series else []
            notes.append("FCF erratic — using CFO as proxy")

        if len(use_series) < 2:
            return self._empty(symbol, "Insufficient FCF/CFO history")

        # ── Historical FCF CAGR ───────────────────────────────────────────────
        # Use last 3 years of smoothed values for CAGR to reduce noise from
        # capex cycles. Compare 3Y-avg → last year to get stable growth rate.
        recent_pos = [v for v in use_series[-5:] if v and v > 0]
        older_pos  = [v for v in use_series[-8:-3] if v and v > 0]
        if recent_pos and older_pos:
            avg_recent = sum(recent_pos) / len(recent_pos)
            avg_older  = sum(older_pos)  / len(older_pos)
            n_periods  = 4   # midpoint difference ~4 years
            if avg_older > 0 and avg_recent > 0:
                fcf_cagr = (avg_recent / avg_older) ** (1 / n_periods) - 1
            else:
                fcf_cagr = 0.07
        else:
            fcf_cagr = self._cagr(use_series[-6:])
        # Floor / cap to realistic range
        fcf_cagr = max(-0.03, min(fcf_cagr, 0.25))

        # Base FCF: average of last 3 positive years to smooth capex cycles
        pos_vals = [v for v in use_series[-5:] if v and v > 0]
        base_fcf = sum(pos_vals) / len(pos_vals) if pos_vals else 0.0
        if base_fcf <= 0:
            return self._empty(symbol, "No positive FCF/CFO baseline")

        # ── Net debt (enterprise → equity bridge) ─────────────────────────────
        # Use Borrowings - liquid buffer (approx 15% of borrowings held as cash)
        # This is conservative and avoids using FCF as a cash proxy.
        net_debt_cr = max(0.0, total_debt_cr * 0.85)

        # ── Base-case DCF ─────────────────────────────────────────────────────
        pv_fcf, pv_tv = self._dcf_calc(base_fcf, fcf_cagr, wacc, _TERM_G, _PROJ_YR)
        total_iv_cr   = pv_fcf + pv_tv
        equity_iv_cr  = total_iv_cr - net_debt_cr

        # For very capital-heavy companies (negative equity after debt bridge),
        # show enterprise value per share and flag it.
        ev_mode = False
        if equity_iv_cr <= 0:
            equity_iv_cr = total_iv_cr   # show EV basis
            ev_mode = True
            notes.append("EV basis — total debt exceeds DCF value (capex cycle)")

        iv_per_share = equity_iv_cr / shares_cr   # ₹ Cr / Cr shares = ₹/share

        # ── Monte Carlo ───────────────────────────────────────────────────────
        rng        = np.random.default_rng(42)
        g_samples  = rng.normal(fcf_cagr, 0.04, _MC_SIMS).clip(-0.06, 0.30)
        w_samples  = rng.normal(wacc,     0.015, _MC_SIMS).clip(0.07, 0.22)

        mc_ivs = []
        for g, w in zip(g_samples, w_samples):
            if w <= _TERM_G:
                continue
            pf, pt   = self._dcf_calc(base_fcf, g, w, _TERM_G, _PROJ_YR)
            eq_iv    = (pf + pt) - net_debt_cr
            if ev_mode or eq_iv <= 0:
                eq_iv = pf + pt
            mc_ivs.append(eq_iv / shares_cr)

        mc_ivs = np.array(mc_ivs)
        iv_low  = float(np.percentile(mc_ivs, 10))
        iv_high = float(np.percentile(mc_ivs, 90))

        # ── Margin of Safety ─────────────────────────────────────────────────
        mos = None
        if cmp and cmp > 0 and iv_per_share > 0:
            mos = (iv_per_share - cmp) / cmp

        verdict = self._verdict(mos, iv_per_share)

        return DCFResult(
            symbol=symbol,
            cmp=cmp,
            intrinsic_value=iv_per_share,
            iv_low=iv_low,
            iv_high=iv_high,
            margin_of_safety=mos,
            wacc=wacc,
            fcf_cagr=fcf_cagr,
            terminal_value=pv_tv,
            pv_fcf=pv_fcf,
            shares_cr=shares_cr,
            verdict=verdict,
            notes=notes,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dcf_calc(
        self, base: float, g: float, wacc: float, term_g: float, yrs: int
    ) -> tuple[float, float]:
        """Return (PV_of_FCFs, PV_of_terminal_value) in ₹ Crore."""
        pv_fcfs = 0.0
        fcf = base
        for t in range(1, yrs + 1):
            fcf *= (1 + g)
            pv_fcfs += fcf / (1 + wacc) ** t
        terminal_fcf = fcf * (1 + term_g)
        if wacc <= term_g:
            return pv_fcfs, 0.0
        tv     = terminal_fcf / (wacc - term_g)
        pv_tv  = tv / (1 + wacc) ** yrs
        return pv_fcfs, pv_tv

    def _cagr(self, series: list[float]) -> float:
        """Compound Annual Growth Rate from first non-zero to last value."""
        vals = [v for v in series if v and v != 0]
        if len(vals) < 2:
            return 0.07
        start, end, n = vals[0], vals[-1], len(vals) - 1
        if start <= 0 or end <= 0:
            # One side negative — use arithmetic mean growth
            changes = [(vals[i+1] - vals[i]) / abs(vals[i]) for i in range(len(vals)-1) if vals[i] != 0]
            return sum(changes) / len(changes) if changes else 0.07
        return (end / start) ** (1 / n) - 1

    def _shares(self, np_series: list[float], eps_series: list[float]) -> float:
        """Estimate shares outstanding in crore from NP and EPS."""
        if not np_series or not eps_series:
            return 0.0
        np_cr  = np_series[-1]
        eps_rs = eps_series[-1]
        if not eps_rs or eps_rs == 0:
            return 0.0
        # NP (crore) * 1e7 Rs/crore = EPS (Rs/share) * N shares
        # → shares = NP_crore / EPS_rs  (result is in crore)
        return np_cr / eps_rs

    def _tail_avg(self, series: list[float], n: int) -> float:
        tail = series[-n:] if len(series) >= n else series
        vals = [v for v in tail if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _load(self, symbol: str) -> Optional[dict]:
        path = self.cache_dir / f"{symbol}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _empty(self, symbol: str, reason: str) -> DCFResult:
        return DCFResult(
            symbol=symbol, cmp=None, intrinsic_value=0.0,
            iv_low=0.0, iv_high=0.0, margin_of_safety=None,
            wacc=0.0, fcf_cagr=0.0, terminal_value=0.0,
            pv_fcf=0.0, shares_cr=0.0, verdict="INSUFFICIENT_DATA",
            notes=[reason],
        )

    def _verdict(self, mos: Optional[float], iv: float) -> str:
        if iv <= 0:
            return "INSUFFICIENT_DATA"
        if mos is None:
            return "NO PRICE DATA"
        if mos >= 0.25:
            return "STRONG BUY"
        if mos >= 0.10:
            return "BUY"
        if mos >= -0.10:
            return "HOLD"
        if mos >= -0.25:
            return "SELL"
        return "STRONG SELL"
