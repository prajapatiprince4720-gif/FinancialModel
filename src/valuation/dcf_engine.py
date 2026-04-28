"""
DCF Valuation Engine with Monte Carlo simulation.

Non-financial companies: Free-Cash-Flow DCF with 10-yr projection + Gordon-growth terminal value.
Banks / NBFCs: Residual Income Model (RIM) — IV = BV + PV(excess returns) + TV.
Both use 10,000 Monte Carlo draws for P10–P90 confidence bands.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_PRICE_CACHE: dict[str, Optional[float]] = {}
_PRICE_CACHE_FILE = Path("data/raw/nse/live_prices.json")


def _fetch_cmp(symbol: str) -> Optional[float]:
    """Live price from yfinance — cached per session + to disk for 4 h."""
    if symbol in _PRICE_CACHE:
        return _PRICE_CACHE[symbol]

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
_RF_RATE  = 0.071   # India 10Y Gilt (risk-free rate)
_ERP      = 0.055   # India Equity Risk Premium
_TERM_G   = 0.055   # Terminal growth rate (long-run nominal GDP)
_PROJ_YR  = 10      # Projection horizon
_MC_SIMS  = 10_000  # Monte Carlo draws

# Sector betas (systematic risk)
_SECTOR_BETA: dict[str, float] = {
    "BANKING":   1.05,
    "IT":        0.80,
    "FMCG":      0.65,
    "PHARMA":    0.75,
    "AUTO":      1.20,
    "ENERGY":    1.10,
    "METALS":    1.40,
    "INFRA":     1.25,
    "TELECOM":   0.90,
    "INSURANCE": 0.85,
    "FINSERV":   1.10,
    "CEMENT":    1.15,
    "DEFAULT":   1.00,
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

# Symbols routed to Residual Income Model instead of FCF-DCF
_BANK_SYMBOLS: set[str] = {
    "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "INDUSINDBK",
}
_NBFC_SYMBOLS: set[str] = {
    "BAJFINANCE", "SHRIRAMFIN", "BAJAJFINSV",
}
_FIN_SECTOR: set[str] = _BANK_SYMBOLS | _NBFC_SYMBOLS

# Projection horizon for RIM (banks are slow-moving; 15Y better than 10Y)
_BANK_PROJ_YR = 15

# Long-run sustainable ROE — calibrated to structural franchise advantages:
# Premium private banks: durable CASA + cross-sell moat supports above-CoE ROE.
# NBFCs: niche specialisation (CV, consumer) sustains higher spread economics.
_ROE_TERMINAL: dict[str, float] = {
    "HDFCBANK":   0.17,   # best CASA franchise; structural low-cost funding
    "ICICIBANK":  0.18,   # fastest-growing large bank; expanding retail engine
    "KOTAKBANK":  0.16,   # premium valuation bank; conservative book growth
    "AXISBANK":   0.15,   # mid-tier private; improving credit quality
    "SBIN":       0.14,   # PSU with improving RoA; scale moat
    "INDUSINDBK": 0.13,   # smaller private; vehicle/microfinance mix
    "BAJFINANCE": 0.23,   # superior NBFC; broad consumer franchise
    "BAJAJFINSV": 0.18,   # holding co; discount to subsidiary BajFin
    "SHRIRAMFIN": 0.20,   # CV specialist NBFC; durable niche ROE
}


@dataclass
class DCFResult:
    symbol:           str
    cmp:              Optional[float]
    intrinsic_value:  float
    iv_low:           float
    iv_high:          float
    margin_of_safety: Optional[float]
    wacc:             float           # WACC for DCF; CoE for RIM
    fcf_cagr:         float           # FCF CAGR (DCF) or current ROE (RIM)
    terminal_value:   float
    pv_fcf:           float           # PV of FCFs (DCF) or PV of residual income (RIM)
    shares_cr:        float
    verdict:          str
    model:            str = "DCF"     # "DCF" or "RIM"
    notes:            list[str] = field(default_factory=list)


class DCFEngine:
    """Computes intrinsic value per share from cached Screener.in data."""

    def __init__(self, cache_dir: str = "data/raw/screener"):
        self.cache_dir = Path(cache_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def value(self, symbol: str) -> DCFResult:
        data = self._load(symbol)
        if data is None:
            return self._empty(symbol, "No cached data")

        if symbol in _FIN_SECTOR:
            return self._bank_rim_value(symbol, data)
        return self._dcf_value(symbol, data)

    # ── FCF-DCF for non-financial companies ───────────────────────────────────

    def _dcf_value(self, symbol: str, data: dict) -> DCFResult:
        pl   = data.get("profit_loss", {})
        bs   = data.get("balance_sheet", {})
        cf   = data.get("cash_flow", {})
        cmp  = data.get("current_price") or _fetch_cmp(symbol)

        years   = pl.get("years", [])
        pl_rows = pl.get("rows", {})
        bs_rows = bs.get("rows", {})
        cf_rows = cf.get("rows", {})

        n   = len(years)
        cut = n - 1 if years and "TTM" in str(years[-1]) else n

        def _s(rows: dict, key: str) -> list[float]:
            return [v for v in rows.get(key, [])[:cut] if v is not None]

        fcf_series = _s(cf_rows, "Free Cash Flow")
        cfo_series = _s(cf_rows, "Cash from Operating Activity")
        np_series  = _s(pl_rows, "Net Profit")
        eps_series = _s(pl_rows, "EPS in Rs")
        sales_ser  = _s(pl_rows, "Sales")
        interest   = _s(pl_rows, "Interest")
        tax_series = _s(pl_rows, "Tax %")
        # Fix: banks use "Borrowing" singular; non-banks use "Borrowings"
        borr_ser   = _s(bs_rows, "Borrowings") or _s(bs_rows, "Borrowing")
        eq_cap     = _s(bs_rows, "Equity Capital")
        reserves   = _s(bs_rows, "Reserves")

        notes: list[str] = []

        shares_cr = self._shares(np_series, eps_series)
        if shares_cr <= 0:
            return self._empty(symbol, "Cannot determine share count")

        beta        = _SECTOR_BETA.get(_SYM_SECTOR.get(symbol, "DEFAULT"), 1.0)
        cost_equity = _RF_RATE + beta * _ERP

        avg_borr = self._tail_avg(borr_ser, 3)
        avg_int  = self._tail_avg(interest, 3)
        if avg_borr > 0 and avg_int > 0:
            cost_debt = max(0.05, min(avg_int / avg_borr, 0.20))
        else:
            cost_debt = 0.085

        avg_tax = self._tail_avg(tax_series, 3) / 100 if tax_series else 0.25
        avg_tax = max(0.0, min(avg_tax, 0.40))

        # Market-cap equity weights (academically correct; book equity understates
        # equity weight for high P/B companies, making WACC artificially low).
        total_debt_cr = borr_ser[-1] if borr_ser else 0
        if cmp and cmp > 0 and shares_cr > 0:
            market_equity_cr = shares_cr * cmp      # Cr shares × ₹/share = ₹ Cr
        else:
            # Fallback to book equity when CMP unavailable
            market_equity_cr = (eq_cap[-1] + reserves[-1]) if eq_cap and reserves else 0
            notes.append("WACC uses book equity (no CMP available)")
        total_v = market_equity_cr + total_debt_cr
        we = market_equity_cr / total_v if total_v > 0 else 0.70
        wd = total_debt_cr   / total_v if total_v > 0 else 0.30

        wacc = we * cost_equity + wd * cost_debt * (1 - avg_tax)
        wacc = max(0.08, min(wacc, 0.20))

        use_series = fcf_series if fcf_series else cfo_series
        tail       = use_series[-5:] if len(use_series) >= 5 else use_series
        if sum(1 for v in tail if v > 0) < 2:
            use_series = cfo_series if cfo_series else []
            notes.append("FCF erratic — using CFO as proxy")

        if len(use_series) < 2:
            return self._empty(symbol, "Insufficient FCF/CFO history")

        recent_pos = [v for v in use_series[-5:] if v and v > 0]
        older_pos  = [v for v in use_series[-8:-3] if v and v > 0]
        if recent_pos and older_pos:
            avg_recent = sum(recent_pos) / len(recent_pos)
            avg_older  = sum(older_pos)  / len(older_pos)
            if avg_older > 0 and avg_recent > 0:
                fcf_cagr = (avg_recent / avg_older) ** (1 / 4) - 1
            else:
                fcf_cagr = 0.07
        else:
            fcf_cagr = self._cagr(use_series[-6:])
        fcf_cagr = max(-0.03, min(fcf_cagr, 0.25))

        pos_vals = [v for v in use_series[-5:] if v and v > 0]
        base_fcf = sum(pos_vals) / len(pos_vals) if pos_vals else 0.0
        if base_fcf <= 0:
            return self._empty(symbol, "No positive FCF/CFO baseline")

        net_debt_cr = max(0.0, total_debt_cr * 0.85)

        pv_fcf, pv_tv = self._dcf_calc(base_fcf, fcf_cagr, wacc, _TERM_G, _PROJ_YR)
        total_iv_cr   = pv_fcf + pv_tv
        equity_iv_cr  = total_iv_cr - net_debt_cr

        ev_mode = False
        if equity_iv_cr <= 0:
            equity_iv_cr = total_iv_cr
            ev_mode = True
            notes.append("EV basis — total debt exceeds DCF value (capex-heavy cycle)")

        iv_per_share = equity_iv_cr / shares_cr

        rng       = np.random.default_rng(42)
        g_samples = rng.normal(fcf_cagr, 0.04, _MC_SIMS).clip(-0.06, 0.30)
        w_samples = rng.normal(wacc,     0.015, _MC_SIMS).clip(0.07, 0.22)

        mc_ivs = []
        for g, w in zip(g_samples, w_samples):
            if w <= _TERM_G:
                continue
            pf, pt = self._dcf_calc(base_fcf, g, w, _TERM_G, _PROJ_YR)
            eq_iv  = (pf + pt) - net_debt_cr
            if ev_mode or eq_iv <= 0:
                eq_iv = pf + pt
            mc_ivs.append(eq_iv / shares_cr)

        mc_arr  = np.array(mc_ivs)
        iv_low  = float(np.percentile(mc_arr, 10))
        iv_high = float(np.percentile(mc_arr, 90))

        mos = (iv_per_share - cmp) / cmp if cmp and cmp > 0 and iv_per_share > 0 else None

        return DCFResult(
            symbol=symbol, cmp=cmp,
            intrinsic_value=round(iv_per_share, 2),
            iv_low=round(iv_low, 2), iv_high=round(iv_high, 2),
            margin_of_safety=mos, wacc=wacc, fcf_cagr=fcf_cagr,
            terminal_value=round(pv_tv, 2), pv_fcf=round(pv_fcf, 2),
            shares_cr=shares_cr, verdict=self._verdict(mos, iv_per_share),
            model="DCF", notes=notes,
        )

    # ── Residual Income Model for banks & NBFCs ───────────────────────────────

    def _bank_rim_value(self, symbol: str, data: dict) -> DCFResult:
        """
        IV = BV₀ + Σ [(ROE_t − CoE) × BV_{t-1}] / (1+CoE)^t  +  TV / (1+CoE)^10

        ROE declines linearly from current to long-run competitive equilibrium.
        BV grows each year by retained earnings: BV_t = BV_{t-1} × (1 + ROE_t × retention).
        Monte Carlo: draws on ROE₀ and CoE to produce P10/P90 bands.
        """
        pl   = data.get("profit_loss", {})
        bs   = data.get("balance_sheet", {})
        rat  = data.get("ratios", {})
        cmp  = data.get("current_price") or _fetch_cmp(symbol)

        years   = pl.get("years", [])
        pl_rows = pl.get("rows", {})
        bs_rows = bs.get("rows", {})
        rat_rows = rat.get("rows", {})

        n   = len(years)
        cut = n - 1 if years and "TTM" in str(years[-1]) else n

        def _s(rows: dict, key: str) -> list[float]:
            return [v for v in rows.get(key, [])[:cut] if v is not None]

        np_series  = _s(pl_rows, "Net Profit")
        eps_series = _s(pl_rows, "EPS in Rs")
        eq_cap     = _s(bs_rows, "Equity Capital")
        reserves   = _s(bs_rows, "Reserves")
        roe_series = _s(rat_rows, "ROE %")
        div_payout = _s(pl_rows, "Dividend Payout %")

        notes: list[str] = []

        shares_cr = self._shares(np_series, eps_series)
        if shares_cr <= 0:
            return self._empty(symbol, "Cannot determine share count")

        if not eq_cap or not reserves:
            return self._empty(symbol, "No equity data for Residual Income Model")

        bv0 = (eq_cap[-1] + reserves[-1]) / shares_cr   # BV per share (₹)

        # Current ROE: prefer direct ratio series; fallback to NP/BV
        if roe_series:
            roe0 = self._tail_avg(roe_series, 3) / 100
        elif np_series and eq_cap and reserves:
            bv_total = eq_cap[-1] + reserves[-1]
            roe0 = np_series[-1] / bv_total if bv_total > 0 else 0.12
        else:
            return self._empty(symbol, "No ROE data for Residual Income Model")

        roe0 = max(0.05, min(roe0, 0.35))

        # Cost of Equity
        beta = _SECTOR_BETA.get(_SYM_SECTOR.get(symbol, "DEFAULT"), 1.0)
        coe  = _RF_RATE + beta * _ERP

        # Retention ratio (1 - payout)
        avg_payout = self._tail_avg(div_payout, 3) / 100 if div_payout else 0.25
        retention  = 1.0 - max(0.10, min(avg_payout, 0.60))

        # Long-run ROE equilibrium (sector-specific)
        roe_term = _ROE_TERMINAL.get(symbol, 0.13)

        proj_yr = _BANK_PROJ_YR  # 15Y horizon — banks are slow-mean-reverting

        # ── Base-case RIM ─────────────────────────────────────────────────────
        pv_ri, bv_n = self._rim_calc(bv0, roe0, roe_term, coe, retention, proj_yr)

        ri_term = (roe_term - coe) * bv_n
        pv_tv   = 0.0
        if coe > _TERM_G and ri_term > 0:
            pv_tv = (ri_term / (coe - _TERM_G)) / (1 + coe) ** proj_yr

        iv_per_share = bv0 + pv_ri + pv_tv

        # ── Vectorised Monte Carlo ────────────────────────────────────────────
        rng         = np.random.default_rng(42)
        roe_samples = rng.normal(roe0, 0.02,  _MC_SIMS).clip(0.04, 0.32)
        coe_samples = rng.normal(coe,  0.015, _MC_SIMS).clip(0.07, 0.18)

        t_arr    = np.arange(1, proj_yr + 1, dtype=float)
        roe_traj = (roe_samples[:, None]
                    + (roe_term - roe_samples[:, None]) * (t_arr / proj_yr))

        bv_prev       = np.empty((_MC_SIMS, proj_yr))
        bv_prev[:, 0] = bv0
        for ti in range(1, proj_yr):
            bv_prev[:, ti] = bv_prev[:, ti - 1] * (1 + roe_traj[:, ti - 1] * retention)

        disc     = (1 + coe_samples[:, None]) ** t_arr
        ri_mat   = (roe_traj - coe_samples[:, None]) * bv_prev
        pv_ri_mc = np.sum(ri_mat / disc, axis=1)

        bv_end   = bv_prev[:, -1] * (1 + roe_traj[:, -1] * retention)
        ri_t_mc  = (roe_term - coe_samples) * bv_end
        valid    = (coe_samples > _TERM_G) & (ri_t_mc > 0)
        pv_tv_mc = np.where(
            valid,
            ri_t_mc / (coe_samples - _TERM_G) / (1 + coe_samples) ** proj_yr,
            0.0,
        )

        mc_arr  = bv0 + pv_ri_mc + pv_tv_mc
        iv_low  = float(np.percentile(mc_arr, 10))
        iv_high = float(np.percentile(mc_arr, 90))

        mos = (iv_per_share - cmp) / cmp if cmp and cmp > 0 and iv_per_share > 0 else None

        notes.append(
            f"Residual Income Model  |  ROE={roe0:.1%}  CoE={coe:.1%}  "
            f"ROE→{roe_term:.0%} ({proj_yr}Y)  Retention={retention:.0%}"
        )

        return DCFResult(
            symbol=symbol, cmp=cmp,
            intrinsic_value=round(iv_per_share, 2),
            iv_low=round(iv_low, 2), iv_high=round(iv_high, 2),
            margin_of_safety=mos, wacc=coe,
            fcf_cagr=roe0,
            terminal_value=round(pv_tv, 2),
            pv_fcf=round(pv_ri, 2),
            shares_cr=shares_cr,
            verdict=self._verdict(mos, iv_per_share),
            model="RIM", notes=notes,
        )

    # ── Shared calculation helpers ────────────────────────────────────────────

    def _rim_calc(
        self,
        bv0: float, roe0: float, roe_term: float,
        coe: float, retention: float, yrs: int,
    ) -> tuple[float, float]:
        """Return (PV of residual income, BV at year yrs) — both per share."""
        bv    = bv0
        pv_ri = 0.0
        for t in range(1, yrs + 1):
            roe_t  = roe0 + (roe_term - roe0) * (t / yrs)
            ri_t   = (roe_t - coe) * bv
            pv_ri += ri_t / (1 + coe) ** t
            bv    *= (1 + roe_t * retention)
        return pv_ri, bv

    def _dcf_calc(
        self, base: float, g: float, wacc: float, term_g: float, yrs: int
    ) -> tuple[float, float]:
        """Return (PV_FCFs, PV_terminal_value) in ₹ Crore."""
        pv_fcfs = 0.0
        fcf = base
        for t in range(1, yrs + 1):
            fcf    *= (1 + g)
            pv_fcfs += fcf / (1 + wacc) ** t
        terminal_fcf = fcf * (1 + term_g)
        if wacc <= term_g:
            return pv_fcfs, 0.0
        tv    = terminal_fcf / (wacc - term_g)
        pv_tv = tv / (1 + wacc) ** yrs
        return pv_fcfs, pv_tv

    def _cagr(self, series: list[float]) -> float:
        vals = [v for v in series if v and v != 0]
        if len(vals) < 2:
            return 0.07
        start, end, n = vals[0], vals[-1], len(vals) - 1
        if start <= 0 or end <= 0:
            changes = [
                (vals[i + 1] - vals[i]) / abs(vals[i])
                for i in range(len(vals) - 1) if vals[i] != 0
            ]
            return sum(changes) / len(changes) if changes else 0.07
        return (end / start) ** (1 / n) - 1

    def _shares(self, np_series: list[float], eps_series: list[float]) -> float:
        if not np_series or not eps_series:
            return 0.0
        np_cr, eps_rs = np_series[-1], eps_series[-1]
        if not eps_rs or eps_rs == 0:
            return 0.0
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
            model="N/A", notes=[reason],
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
