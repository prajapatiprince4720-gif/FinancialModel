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
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.company import BANKS, INSURERS, LENDERS, NBFCS, CompanyData, load_company
from src.utils.stats import log_linear_growth, robust_sd
from src.valuation.probabilistic import ValueDriverInputs, simulate, verdict_from_probability

_PRICE_CACHE: dict[str, Optional[float]] = {}
_PRICE_CACHE_FILE = Path("data/raw/nse/live_prices.json")


# Some NSE symbols need alternate yfinance tickers
_TICKER_OVERRIDES: dict[str, list[str]] = {
    "ZOMATO":     ["ETERNAL.NS"],
    # TATAMOTORS deliberately has no fallback: after the 2025 demerger the
    # traded TMPV price covers passenger vehicles only, while Screener still
    # reports the combined company — pairing them gave a 1.3x P/E. No price is
    # better than a wrong one. See CORPORATE_ACTIONS in src/data/company.py.
}


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
                if symbol in file_cache and file_cache[symbol]:
                    _PRICE_CACHE[symbol] = file_cache[symbol]
                    return file_cache[symbol]
            except Exception:
                pass

    # Build list of tickers to try: primary NSE + any overrides
    tickers_to_try = [f"{symbol}.NS"] + _TICKER_OVERRIDES.get(symbol, [])
    price = None
    try:
        import yfinance as yf
        for ticker in tickers_to_try:
            try:
                t     = yf.Ticker(ticker)
                info  = t.fast_info
                price = float(info.last_price) if hasattr(info, "last_price") and info.last_price else None
                if price is None or price == 0:
                    hist  = t.history(period="1d")
                    price = float(hist["Close"].iloc[-1]) if not hist.empty else None
                if price and price > 0:
                    break   # found a good price — stop trying
            except Exception:
                continue
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


def _price_for(cd: CompanyData) -> Optional[float]:
    """
    The price to value `cd` against. A historical view uses only the price it
    carries — falling back to the live quote would pair old fundamentals with
    today's price — so without one, price-based outputs are left empty.
    """
    if cd.is_historical:
        return cd.current_price
    return cd.current_price or _fetch_cmp(cd.symbol)


# ── India macro constants ──────────────────────────────────────────────────────
# RF_RATE is fetched live from NSE/yfinance (24h cache) — never hardcoded.
from src.utils.macro_rates import RFContext, get_rf_context
_ERP      = 0.055            # India Equity Risk Premium (Damodaran estimate)
_TERM_G   = 0.055            # Terminal growth rate (long-run nominal GDP)
_PROJ_YR  = 10               # Projection horizon
_MC_SIMS  = 10_000           # Monte Carlo draws

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
    "DEFENSE":   0.90,   # PSU defense; steady order books, low cyclicality
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
    "TITAN": "FMCG", "ASIANPAINT": "FMCG", "BEL": "DEFENSE",
    "APOLLOHOSP": "DEFAULT", "ZOMATO": "DEFAULT",
}

# Symbols routed to Residual Income Model instead of FCF-DCF
_BANK_SYMBOLS = BANKS
_NBFC_SYMBOLS = NBFCS
_FIN_SECTOR   = LENDERS

# Insurance: P/E relative model — CFO includes policyholder premiums (inflated),
# so DCF gives nonsensical STRONG BUY (+150%). Sector trades at ~62x trailing P/E.
_INSURANCE_SYMBOLS = INSURERS
_INSURANCE_SECTOR_PE = 62.0   # Indian listed life insurers trailing P/E anchor
_INSURANCE_PE_SIGMA  = 12.0   # ±1σ P/E band (~50–74x historically observed)

# Conglomerates whose consolidated balance sheet carries a lending subsidiary
# (L&T Finance, M&M Financial, Aditya Birla Capital). Their borrowings are a lender's funding, not
# corporate leverage; deducting them understates equity value.
_HAS_LENDING_ARM: set[str] = {"LT", "M&M", "GRASIM"}   # + Aditya Birla Capital

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
    rf_rate:          float = 0.0     # risk-free rate actually used
    rf_source:        str   = ""      # where that rate came from
    rf_is_live:       bool  = False   # False => valuation rests on a stale rate
    rf_label:         str   = ""      # one-line provenance for report footers
    # ── Probabilistic outputs (see src/valuation/probabilistic.py) ──
    # intrinsic_value is the MEDIAN of the simulated distribution; iv_low/iv_high
    # are P10/P90. prob_undervalued = P(value > price).
    prob_undervalued: Optional[float] = None
    iv_p25:           float = 0.0
    iv_p75:           float = 0.0
    iv_base:          float = 0.0     # value at central assumptions (no simulation)
    implied_growth:   Optional[float] = None   # reverse DCF: starting growth priced in
    implied_status:   str   = ""
    sensitivities:    dict  = field(default_factory=dict)  # % value change per +1pp
    terminal_share:   float = 0.0     # share of value from the terminal period


class DCFEngine:
    """
    Computes intrinsic value per share from cached Screener.in data.

    The risk-free rate is resolved once per engine instance rather than at
    import time, so a long-running process (the web API) picks up a new rate
    when the 24 h macro cache rolls over, and importing this module never
    triggers a network call.
    """

    def __init__(self, cache_dir: str = "data/raw/screener",
                 rf_context: RFContext | None = None, quiet: bool = False):
        self.cache_dir = Path(cache_dir)
        self.rf = rf_context or get_rf_context(quiet=quiet)

    @property
    def _rf_rate(self) -> float:
        return self.rf.rate

    def _rf_fields(self) -> dict:
        """Provenance kwargs stamped onto every DCFResult this engine produces."""
        return {
            "rf_rate":    self.rf.rate,
            "rf_source":  self.rf.source,
            "rf_is_live": self.rf.is_live,
            "rf_label":   self.rf.label,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def value(self, symbol: str, as_of: datetime | str | None = None,
              data: Optional[CompanyData] = None) -> DCFResult:
        """
        Value `symbol` from current data, from the snapshot observed at `as_of`,
        or from an explicit CompanyData (the backtest passes one built with
        CompanyData.as_of(), carrying the historical price).

        `as_of` pins the fundamentals only; the risk-free rate is whatever this
        engine was given.
        """
        cd = data if data is not None else load_company(symbol, as_of=as_of, cache_dir=self.cache_dir)
        if cd is None:
            return self._empty(symbol, "No cached data")

        if symbol in _INSURANCE_SYMBOLS:
            return self._insurance_pe_value(symbol, cd)
        if symbol in _FIN_SECTOR:
            return self._bank_rim_value(symbol, cd)
        return self._dcf_value(symbol, cd)

    # ── Value-driver DCF for non-financial companies ──────────────────────────

    def _dcf_value(self, symbol: str, cd: CompanyData) -> DCFResult:
        """
        Probabilistic value-driver DCF (see src/valuation/probabilistic.py).

        Inputs come from the statements, each estimated robustly:
          NOPAT₀  latest revenue × median 5Y EBIT margin × (1 − tax) — normalised,
                  so one boom or bust year does not set the base
          g₀      OLS slope of ln(revenue) over 6 years — uses every year, not
                  just the endpoints, so a depressed FY21 base cannot inflate it
          ROIC₀   median 5Y NOPAT / average capital employed; its spread is the
                  MAD-based robust standard deviation of the same years

        EBIT (= PBT + interest) includes other income earned on cash and
        investments, and capital employed (= equity + borrowings) includes that
        cash. Equity value is therefore EV minus *gross* borrowings — cash is
        already valued through its income, so it is not added again.
        """
        cmp = _price_for(cd)
        pl, bs = "profit_loss", "balance_sheet"
        n = len(cd.years(pl))

        def fit(vals: list) -> list:
            vals = vals[-n:] if n else []
            return [None] * (n - len(vals)) + vals

        sales    = fit(cd.aligned(pl, "Sales", "Revenue"))
        pbt      = fit(cd.aligned(pl, "Profit before tax"))
        interest = fit(cd.aligned(pl, "Interest"))
        tax_pct  = fit(cd.aligned(pl, "Tax %"))
        eq_cap   = fit(cd.aligned(bs, "Equity Capital"))
        reserves = fit(cd.aligned(bs, "Reserves"))
        borrow   = fit(cd.aligned(bs, "Borrowings", "Borrowing"))
        np_series  = cd.series(pl, "Net Profit")
        eps_series = cd.series(pl, "EPS in Rs")

        notes: list[str] = []
        shares_cr = self._shares(np_series, eps_series)
        if shares_cr <= 0:
            return self._empty(symbol, "Cannot determine share count")

        def ok(*xs) -> bool:
            return all(x is not None and np.isfinite(x) for x in xs)

        def tax_rate(pct) -> float:
            return min(max((pct if ok(pct) else 25.17) / 100, 0.0), 0.35)

        ebit = [p + (i or 0) if ok(p) else None for p, i in zip(pbt, interest)]
        cap  = [e + r + (b or 0) if ok(e, r) else None
                for e, r, b in zip(eq_cap, reserves, borrow)]

        roic_hist = [
            ebit[t] * (1 - tax_rate(tax_pct[t])) / ((cap[t] + cap[t - 1]) / 2)
            for t in range(1, n)
            if ok(ebit[t], cap[t], cap[t - 1]) and cap[t] + cap[t - 1] > 0
        ][-5:]
        margins = [e / s for e, s in zip(ebit[-5:], sales[-5:]) if ok(e, s) and s > 0]
        if len(roic_hist) < 3 or len(margins) < 3 or not ok(sales[-1]):
            return self._empty(symbol, "Insufficient operating history for value-driver model")

        tax0   = tax_rate(float(np.median([t for t in tax_pct[-3:] if ok(t)] or [25.17])))
        nopat0 = sales[-1] * float(np.median(margins)) * (1 - tax0)
        if nopat0 <= 0:
            return self._empty(
                symbol, "Not consistently profitable — normalised operating profit is negative, "
                        "so an earnings-based valuation is not meaningful")

        roic0   = float(np.median(roic_hist))
        roic_sd = max(robust_sd(roic_hist) or 0.0, 0.01)
        g0      = log_linear_growth(sales)
        if g0 is None:
            g0 = 0.07
            notes.append("Revenue history too short for a growth fit — using 7%")
        g0 = max(-0.05, min(g0, 0.25))

        # ── Discount rate: market-weight WACC ────────────────────────────────
        beta        = _SECTOR_BETA.get(_SYM_SECTOR.get(symbol, "DEFAULT"), 1.0)
        debt_cr     = borrow[-1] if ok(borrow[-1]) else 0.0
        int_3y      = [i for i in interest[-3:] if ok(i)]
        borr_3y     = [b for b in borrow[-3:] if ok(b)]
        if borr_3y and int_3y and np.mean(borr_3y) > 0 and np.mean(int_3y) > 0:
            cost_debt = min(max(np.mean(int_3y) / np.mean(borr_3y), 0.05), 0.20)
        else:
            cost_debt = 0.085
        if cmp and cmp > 0:
            mkt_eq = shares_cr * cmp
        else:
            mkt_eq = (eq_cap[-1] or 0) + (reserves[-1] or 0)
            notes.append("WACC uses book equity (no CMP available)")
        total = mkt_eq + debt_cr
        we, wd = (mkt_eq / total, debt_cr / total) if total > 0 else (0.7, 0.3)

        if symbol in _HAS_LENDING_ARM:
            notes.append(
                "Consolidated statements include a lending subsidiary whose borrowings are "
                "deducted as corporate debt — a sum-of-the-parts valuation is needed")

        inputs = ValueDriverInputs(
            nopat0=nopat0, g0=g0, roic0=roic0, roic_sd=roic_sd,
            rf=self._rf_rate, beta=beta, erp=_ERP, shares=shares_cr, debt=debt_cr,
            w_equity=we, w_debt=wd, kd_after_tax=cost_debt * (1 - tax0),
            g_inf=_TERM_G, horizon=_PROJ_YR,
        )
        pv = simulate(inputs, cmp)
        notes.append(
            f"Value-driver model | NOPAT₀ ₹{nopat0:,.0f} Cr  ROIC {roic0:.1%} (±{roic_sd:.1%})  "
            f"g₀ {g0:.1%} → {_TERM_G:.1%}  WACC {pv.discount_rate:.1%}"
        )
        if pv.zero_equity_share > 0.10:
            notes.append(f"Debt exceeds enterprise value in {pv.zero_equity_share:.0%} of scenarios")

        return DCFResult(
            symbol=symbol, cmp=cmp,
            intrinsic_value=round(pv.median, 2),
            iv_low=round(pv.p10, 2), iv_high=round(pv.p90, 2),
            margin_of_safety=(pv.median - cmp) / cmp if cmp and cmp > 0 else None,
            wacc=pv.discount_rate, fcf_cagr=g0,
            terminal_value=round(pv.pv_terminal, 2), pv_fcf=round(pv.pv_explicit, 2),
            shares_cr=shares_cr,
            # A zero median here is a result (debt exceeds value in most
            # scenarios), not missing data, so the verdict comes from P alone.
            verdict=verdict_from_probability(pv.prob_undervalued),
            model="DCF", notes=notes, **self._rf_fields(),
            prob_undervalued=pv.prob_undervalued,
            iv_p25=round(pv.p25, 2), iv_p75=round(pv.p75, 2), iv_base=round(pv.base_case, 2),
            implied_growth=pv.implied.g0 if pv.implied else None,
            implied_status=pv.implied.status if pv.implied else "",
            sensitivities=pv.sensitivities, terminal_share=pv.terminal_share,
        )

    # ── P/E relative model for listed life insurers ───────────────────────────

    def _insurance_pe_value(self, symbol: str, cd: CompanyData) -> DCFResult:
        """
        IV = 3Y-avg EPS × sector P/E anchor (62x for Indian life insurers).

        Why not DCF/RIM? Insurance CFO includes policyholder premiums (inflows)
        that are not free cash to equity — DCF overstates IV by 150%+.
        P/E vs sector peers gives a grounded relative valuation.
        """
        cmp  = _price_for(cd)

        np_series  = cd.series("profit_loss", "Net Profit")
        eps_series = cd.series("profit_loss", "EPS in Rs")
        notes: list[str] = []

        shares_cr = self._shares(np_series, eps_series)
        if shares_cr <= 0:
            return self._empty(symbol, "Cannot determine share count")

        eps_tail = [v for v in eps_series[-3:] if v and v > 0]
        if not eps_tail:
            return self._empty(symbol, "No positive EPS for P/E model")

        avg_eps      = sum(eps_tail) / len(eps_tail)
        iv_per_share = avg_eps * _INSURANCE_SECTOR_PE

        rng       = np.random.default_rng(42)
        pe_draws  = rng.normal(_INSURANCE_SECTOR_PE, _INSURANCE_PE_SIGMA, _MC_SIMS).clip(30, 100)
        eps_draws = rng.normal(avg_eps, avg_eps * 0.10, _MC_SIMS).clip(avg_eps * 0.5, avg_eps * 2.0)
        mc_arr    = pe_draws * eps_draws

        dist = self._distribution(mc_arr, cmp)
        beta = _SECTOR_BETA.get("INSURANCE", 0.85)
        coe  = self._rf_rate + beta * _ERP

        notes.append(
            f"P/E Relative model  |  3Y-avg EPS=₹{avg_eps:.1f}  "
            f"Sector P/E={_INSURANCE_SECTOR_PE:.0f}x  [Range: ~50–74x]"
        )

        return DCFResult(
            symbol=symbol, cmp=cmp,
            intrinsic_value=round(dist["median"], 2),
            iv_low=round(dist["p10"], 2), iv_high=round(dist["p90"], 2),
            margin_of_safety=(dist["median"] - cmp) / cmp if cmp and cmp > 0 else None,
            wacc=coe,
            fcf_cagr=avg_eps,
            terminal_value=0.0,
            pv_fcf=round(iv_per_share, 2),
            shares_cr=shares_cr,
            verdict=self._prob_verdict(dist["prob"], dist["median"]),
            model="PE", notes=notes, **self._rf_fields(),
            prob_undervalued=dist["prob"],
            iv_p25=round(dist["p25"], 2), iv_p75=round(dist["p75"], 2),
            iv_base=round(iv_per_share, 2),
            # Value is linear in the multiple: +1x P/E → +1/62 of value.
            sensitivities={"sector_pe_plus_1x": 100 / _INSURANCE_SECTOR_PE},
        )

    # ── Residual Income Model for banks & NBFCs ───────────────────────────────

    def _bank_rim_value(self, symbol: str, cd: CompanyData) -> DCFResult:
        """
        IV = BV₀ + Σ [(ROE_t − CoE) × BV_{t-1}] / (1+CoE)^t  +  TV / (1+CoE)^10

        ROE declines linearly from current to long-run competitive equilibrium.
        BV grows each year by retained earnings: BV_t = BV_{t-1} × (1 + ROE_t × retention).
        Monte Carlo: draws on ROE₀ and CoE to produce P10/P90 bands.
        """
        cmp  = _price_for(cd)

        np_series  = cd.series("profit_loss", "Net Profit")
        eps_series = cd.series("profit_loss", "EPS in Rs")
        eq_cap     = cd.series("balance_sheet", "Equity Capital")
        reserves   = cd.series("balance_sheet", "Reserves")
        roe_series = cd.series("ratios", "ROE %")
        div_payout = cd.series("profit_loss", "Dividend Payout %")

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
        coe  = self._rf_rate + beta * _ERP

        # Retention ratio (1 - payout)
        avg_payout = self._tail_avg(div_payout, 3) / 100 if div_payout else 0.25
        retention  = 1.0 - max(0.10, min(avg_payout, 0.60))

        # Long-run ROE equilibrium (sector-specific)
        roe_term = _ROE_TERMINAL.get(symbol, 0.13)

        proj_yr = _BANK_PROJ_YR  # 15Y horizon — banks are slow-mean-reverting

        # ── Base-case RIM ─────────────────────────────────────────────────────
        def rim(coe_: float, roe_term_: float) -> tuple[float, float, float]:
            pv_ri_, bv_n_ = self._rim_calc(bv0, roe0, roe_term_, coe_, retention, proj_yr)
            ri_term_ = (roe_term_ - coe_) * bv_n_
            pv_tv_ = 0.0
            if coe_ > _TERM_G and ri_term_ > 0:
                pv_tv_ = (ri_term_ / (coe_ - _TERM_G)) / (1 + coe_) ** proj_yr
            return bv0 + pv_ri_ + pv_tv_, pv_ri_, pv_tv_

        iv_per_share, pv_ri, pv_tv = rim(coe, roe_term)

        # Sensitivities by central difference: % value change per +1pp.
        h = 1e-4
        sens = {}
        if iv_per_share > 0:
            d_coe = (rim(coe + h, roe_term)[0] - rim(coe - h, roe_term)[0]) / (2 * h)
            d_roe = (rim(coe, roe_term + h)[0] - rim(coe, roe_term - h)[0]) / (2 * h)
            sens = {"discount_rate":  d_coe * 0.01 / iv_per_share * 100,
                    "long_run_roe":   d_roe * 0.01 / iv_per_share * 100}

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

        mc_arr = bv0 + pv_ri_mc + pv_tv_mc
        dist   = self._distribution(mc_arr, cmp)

        notes.append(
            f"Residual Income Model  |  ROE={roe0:.1%}  CoE={coe:.1%}  "
            f"ROE→{roe_term:.0%} ({proj_yr}Y)  Retention={retention:.0%}"
        )

        return DCFResult(
            symbol=symbol, cmp=cmp,
            intrinsic_value=round(dist["median"], 2),
            iv_low=round(dist["p10"], 2), iv_high=round(dist["p90"], 2),
            margin_of_safety=(dist["median"] - cmp) / cmp if cmp and cmp > 0 else None,
            wacc=coe,
            fcf_cagr=roe0,
            terminal_value=round(pv_tv, 2),
            pv_fcf=round(pv_ri, 2),
            shares_cr=shares_cr,
            verdict=self._prob_verdict(dist["prob"], dist["median"]),
            model="RIM", notes=notes, **self._rf_fields(),
            prob_undervalued=dist["prob"],
            iv_p25=round(dist["p25"], 2), iv_p75=round(dist["p75"], 2),
            iv_base=round(iv_per_share, 2), sensitivities=sens,
            terminal_share=pv_tv / iv_per_share if iv_per_share > 0 else 0.0,
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
        """
        Current share count (Cr) implied by the latest year's NP / EPS.

        Only the latest year is used, so pre-IPO share counts (ZOMATO) never
        enter. A former "pre-IPO guard" here trimmed earlier years on any >20x EPS
        jump — which an ordinary loss year also triggers (BHARTIARTL 0.74 → −55) —
        and, because it shortened the list it was iterating, raised IndexError.
        """
        if not np_series or not eps_series:
            return 0.0
        np_cr, eps_rs = np_series[-1], eps_series[-1]
        if not eps_rs:
            return 0.0
        return np_cr / eps_rs

    def _tail_avg(self, series: list[float], n: int) -> float:
        tail = series[-n:] if len(series) >= n else series
        vals = [v for v in tail if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _empty(self, symbol: str, reason: str) -> DCFResult:
        return DCFResult(
            symbol=symbol, cmp=None, intrinsic_value=0.0,
            iv_low=0.0, iv_high=0.0, margin_of_safety=None,
            wacc=0.0, fcf_cagr=0.0, terminal_value=0.0,
            pv_fcf=0.0, shares_cr=0.0, verdict="INSUFFICIENT_DATA",
            model="N/A", notes=[reason], **self._rf_fields(),
        )

    @staticmethod
    def _prob_verdict(prob: Optional[float], iv: float) -> str:
        if iv <= 0:
            return "INSUFFICIENT_DATA"
        return verdict_from_probability(prob)

    @staticmethod
    def _distribution(mc_arr: np.ndarray, cmp: Optional[float]) -> dict:
        """Median, quartiles, P10/P90 and P(value > price) from simulated values."""
        p10, p25, p50, p75, p90 = np.percentile(mc_arr, [10, 25, 50, 75, 90])
        prob = float(np.mean(mc_arr > cmp)) if cmp and cmp > 0 else None
        return {"median": float(p50), "p10": float(p10), "p25": float(p25),
                "p75": float(p75), "p90": float(p90), "prob": prob}

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
