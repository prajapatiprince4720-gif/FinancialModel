"""
Multi-Factor Composite Scoring Engine.

Scores each Nifty 50 company 0–100 across four pillars:
  Quality  (35%) — ROCE, Operating Margin, Debt load, Interest Coverage
  Growth   (30%) — Revenue CAGR, EPS CAGR, FCF CAGR (3-year)
  Earnings Quality (20%) — CFO/NP, Accruals ratio
  Piotroski F-Score (15%)

Scores are percentile-ranked within the universe so they are relative, not
absolute — a 75 means better than 75% of Nifty 50 on that factor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ScoreResult:
    symbol:             str
    composite:          float          # 0–100
    quality:            float
    growth:             float
    earnings_quality:   float
    piotroski:          int            # 0–9
    piotroski_pct:      float          # percentile of piotroski within universe
    roce_pct:           float
    opm_pct:            float
    revenue_cagr_3y:    float
    eps_cagr_3y:        float
    fcf_cagr_3y:        float
    de_ratio:           float          # raw Debt/Equity
    interest_coverage:  float          # raw EBIT / Interest
    grade:              str            # A+ / A / B+ / B / C / D
    verdict:            str            # TOP PICK / BUY / HOLD / AVOID


class CompositeScorer:
    """Compute composite scores for a list of symbols, percentile-ranked."""

    def __init__(self, cache_dir: str = "data/raw/screener"):
        self.cache_dir = Path(cache_dir)

    def score_all(self, symbols: list[str]) -> list[ScoreResult]:
        raw = {sym: self._extract(sym) for sym in symbols}
        raw = {k: v for k, v in raw.items() if v is not None}
        if not raw:
            return []
        results = self._rank_and_score(raw, symbols)
        return sorted(results, key=lambda r: r.composite, reverse=True)

    # ── Per-company raw metric extraction ────────────────────────────────────

    def _extract(self, symbol: str) -> Optional[dict]:
        path = self.cache_dir / f"{symbol}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)

        pl   = data.get("profit_loss", {})
        bs   = data.get("balance_sheet", {})
        cf   = data.get("cash_flow", {})
        rat  = data.get("ratios", {})

        years    = pl.get("years", [])
        pl_rows  = pl.get("rows", {})
        bs_rows  = bs.get("rows", {})
        cf_rows  = cf.get("rows", {})
        rat_rows = rat.get("rows", {})

        # Drop TTM column
        n   = len(years)
        cut = n - 1 if years and "TTM" in str(years[-1]) else n

        def _s(rows, key):
            return [v for v in rows.get(key, [])[:cut] if v is not None]

        sales_ser   = _s(pl_rows, "Sales")
        np_ser      = _s(pl_rows, "Net Profit")
        eps_ser     = _s(pl_rows, "EPS in Rs")
        opm_ser     = _s(pl_rows, "OPM %")
        int_ser     = _s(pl_rows, "Interest")
        dep_ser     = _s(pl_rows, "Depreciation")
        pbt_ser     = _s(pl_rows, "Profit before tax")
        cfo_ser     = _s(cf_rows, "Cash from Operating Activity")
        fcf_ser     = _s(cf_rows, "Free Cash Flow")
        borr_ser    = _s(bs_rows, "Borrowings")
        eq_cap      = _s(bs_rows, "Equity Capital")
        reserves    = _s(bs_rows, "Reserves")
        tot_assets  = _s(bs_rows, "Total Assets")
        roce_ser    = _s(rat_rows, "ROCE %")

        # ── Quality metrics ───────────────────────────────────────────────────
        roce_3y = self._tail_avg(roce_ser, 3)

        opm_3y = self._tail_avg(opm_ser, 3)

        # D/E = Borrowings / (Equity Capital + Reserves)
        eq_book = (eq_cap[-1] + reserves[-1]) if eq_cap and reserves else 1
        de_ratio = borr_ser[-1] / eq_book if borr_ser and eq_book > 0 else 0.0

        # Interest Coverage = (PBT + Interest + Depreciation) / Interest = EBITDA/Interest
        ebitda = None
        if pbt_ser and int_ser and dep_ser:
            ebitda   = pbt_ser[-1] + int_ser[-1] + dep_ser[-1]
            int_cov  = ebitda / int_ser[-1] if int_ser[-1] > 0 else 99.0
        else:
            int_cov = 99.0

        # ── Growth metrics ────────────────────────────────────────────────────
        rev_cagr = self._cagr(sales_ser[-4:]) if len(sales_ser) >= 2 else 0.0
        eps_cagr = self._cagr(eps_ser[-4:])   if len(eps_ser)   >= 2 else 0.0
        # FCF: use positive-only series if FCF is erratic
        pos_fcf = [v for v in fcf_ser if v and v > 0]
        fcf_cagr = self._cagr(fcf_ser[-4:]) if len(fcf_ser) >= 4 else (
                   self._cagr(cfo_ser[-4:]) if len(cfo_ser) >= 4 else 0.0)

        # ── Earnings quality ──────────────────────────────────────────────────
        # CFO / Net Profit (>1 = high quality — cash earnings > accounting earnings)
        cfo_np_ratio = 0.0
        if cfo_ser and np_ser and np_ser[-1] and np_ser[-1] > 0:
            cfo_np_ratio = cfo_ser[-1] / np_ser[-1]

        # Accrual ratio: (NP - CFO) / Avg Total Assets — lower is better
        accrual_ratio = 0.0
        if np_ser and cfo_ser and tot_assets and len(tot_assets) >= 2:
            avg_ta = (tot_assets[-1] + tot_assets[-2]) / 2
            accrual_ratio = (np_ser[-1] - cfo_ser[-1]) / avg_ta if avg_ta > 0 else 0.0

        # ── Piotroski F-Score (0–9) ───────────────────────────────────────────
        score = 0
        if np_ser and tot_assets and tot_assets[-1]:
            roa_now  = np_ser[-1] / tot_assets[-1]
            score   += 1 if roa_now > 0 else 0
            if len(np_ser) >= 2 and len(tot_assets) >= 2:
                roa_prev = np_ser[-2] / tot_assets[-2]
                score   += 1 if roa_now > roa_prev else 0
        if cfo_ser:
            score += 1 if cfo_ser[-1] > 0 else 0
        # Accruals: CFO/Assets > ROA
        if np_ser and cfo_ser and tot_assets:
            cfo_a   = cfo_ser[-1] / tot_assets[-1]
            roa_now = np_ser[-1]  / tot_assets[-1]
            score  += 1 if cfo_a > roa_now else 0
        # Leverage decreasing
        if borr_ser and tot_assets and len(borr_ser) >= 2 and len(tot_assets) >= 2:
            lev_now  = borr_ser[-1] / tot_assets[-1]
            lev_prev = borr_ser[-2] / tot_assets[-2]
            score   += 1 if lev_now < lev_prev else 0
        # OPM improving
        if opm_ser and len(opm_ser) >= 2:
            score += 1 if opm_ser[-1] > opm_ser[-2] else 0
        # Asset Turnover improving
        if sales_ser and tot_assets and len(sales_ser) >= 2 and len(tot_assets) >= 2:
            at_now  = sales_ser[-1] / tot_assets[-1]
            at_prev = sales_ser[-2] / tot_assets[-2]
            score  += 1 if at_now > at_prev else 0
        # No share dilution: EPS grows at least as fast as NP
        if eps_ser and np_ser and len(eps_ser) >= 2 and len(np_ser) >= 2:
            eps_g = (eps_ser[-1] - eps_ser[-2]) / abs(eps_ser[-2]) if eps_ser[-2] else 0
            np_g  = (np_ser[-1]  - np_ser[-2])  / abs(np_ser[-2])  if np_ser[-2] else 0
            score += 1 if eps_g >= np_g - 0.03 else 0   # 3% tolerance
        # CFO positive trend (bonus)
        if cfo_ser and len(cfo_ser) >= 3:
            score += 1 if cfo_ser[-1] >= cfo_ser[-3] else 0

        return {
            "roce_3y":       roce_3y,
            "opm_3y":        opm_3y,
            "de_ratio":      de_ratio,
            "int_cov":       int_cov,
            "rev_cagr":      rev_cagr,
            "eps_cagr":      eps_cagr,
            "fcf_cagr":      fcf_cagr,
            "cfo_np_ratio":  cfo_np_ratio,
            "accrual_ratio": accrual_ratio,
            "piotroski":     score,
        }

    # ── Cross-sectional percentile ranking ────────────────────────────────────

    def _rank_and_score(self, raw: dict[str, dict], all_syms: list[str]) -> list[ScoreResult]:
        syms  = list(raw.keys())
        n     = len(syms)

        def _pct(vals: list[float], higher_better: bool = True) -> list[float]:
            """Percentile rank: 0–100, higher = better rank (always)."""
            arr = np.array(vals, dtype=float)
            ranks = np.array([np.sum(arr < v) for v in arr], dtype=float)
            pcts  = ranks / max(n - 1, 1) * 100
            return pcts if higher_better else (100 - pcts)

        roce_vals  = [raw[s]["roce_3y"]       for s in syms]
        opm_vals   = [raw[s]["opm_3y"]        for s in syms]
        de_vals    = [raw[s]["de_ratio"]       for s in syms]
        ic_vals    = [raw[s]["int_cov"]        for s in syms]
        rev_vals   = [raw[s]["rev_cagr"]       for s in syms]
        eps_vals   = [raw[s]["eps_cagr"]       for s in syms]
        fcf_vals   = [raw[s]["fcf_cagr"]       for s in syms]
        cfo_vals   = [raw[s]["cfo_np_ratio"]   for s in syms]
        acc_vals   = [raw[s]["accrual_ratio"]  for s in syms]
        pio_vals   = [raw[s]["piotroski"]      for s in syms]

        roce_pct = _pct(roce_vals)
        opm_pct  = _pct(opm_vals)
        de_pct   = _pct(de_vals,   higher_better=False)   # lower D/E is better
        ic_pct   = _pct(ic_vals)
        rev_pct  = _pct(rev_vals)
        eps_pct  = _pct(eps_vals)
        fcf_pct  = _pct(fcf_vals)
        cfo_pct  = _pct(cfo_vals)
        acc_pct  = _pct(acc_vals,  higher_better=False)   # lower accruals better
        pio_pct  = _pct(pio_vals)

        results = []
        for i, sym in enumerate(syms):
            q  = 0.40 * roce_pct[i] + 0.30 * opm_pct[i] + 0.20 * de_pct[i] + 0.10 * ic_pct[i]
            g  = 0.35 * rev_pct[i]  + 0.40 * eps_pct[i]  + 0.25 * fcf_pct[i]
            eq = 0.60 * cfo_pct[i]  + 0.40 * acc_pct[i]
            p  = pio_pct[i]

            composite = 0.35 * q + 0.30 * g + 0.20 * eq + 0.15 * p
            composite = max(0, min(100, composite))

            r = raw[sym]
            results.append(ScoreResult(
                symbol=sym,
                composite=round(composite, 1),
                quality=round(q, 1),
                growth=round(g, 1),
                earnings_quality=round(eq, 1),
                piotroski=r["piotroski"],
                piotroski_pct=round(pio_pct[i], 1),
                roce_pct=round(roce_pct[i], 1),
                opm_pct=round(opm_pct[i], 1),
                revenue_cagr_3y=round(r["rev_cagr"] * 100, 1),
                eps_cagr_3y=round(r["eps_cagr"] * 100, 1),
                fcf_cagr_3y=round(r["fcf_cagr"] * 100, 1),
                de_ratio=round(r["de_ratio"], 2),
                interest_coverage=round(min(r["int_cov"], 99.9), 1),
                grade=self._grade(composite),
                verdict=self._verdict(composite),
            ))

        # Fill missing symbols with zero scores
        scored_syms = {r.symbol for r in results}
        for sym in all_syms:
            if sym not in scored_syms:
                results.append(ScoreResult(
                    symbol=sym, composite=0, quality=0, growth=0,
                    earnings_quality=0, piotroski=0, piotroski_pct=0,
                    roce_pct=0, opm_pct=0, revenue_cagr_3y=0,
                    eps_cagr_3y=0, fcf_cagr_3y=0, de_ratio=0,
                    interest_coverage=0, grade="N/A", verdict="NO DATA",
                ))
        return results

    # ── Utils ─────────────────────────────────────────────────────────────────

    def _cagr(self, series: list[float]) -> float:
        vals = [v for v in series if v is not None and v != 0]
        if len(vals) < 2:
            return 0.0
        start, end, n = vals[0], vals[-1], len(vals) - 1
        if start <= 0 or end <= 0:
            changes = [(vals[i+1] - vals[i]) / abs(vals[i]) for i in range(len(vals)-1) if vals[i] != 0]
            return sum(changes) / len(changes) if changes else 0.0
        return (end / start) ** (1 / n) - 1

    def _tail_avg(self, series: list, n: int) -> float:
        tail = series[-n:] if len(series) >= n else series
        vals = [v for v in tail if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _grade(self, score: float) -> str:
        if score >= 80: return "A+"
        if score >= 65: return "A"
        if score >= 50: return "B+"
        if score >= 35: return "B"
        if score >= 20: return "C"
        return "D"

    def _verdict(self, score: float) -> str:
        if score >= 75: return "TOP PICK"
        if score >= 55: return "BUY"
        if score >= 35: return "HOLD"
        return "AVOID"
