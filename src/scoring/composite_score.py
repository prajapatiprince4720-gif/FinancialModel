"""
Multi-Factor Composite Scoring Engine — Nifty 50.

Non-financial companies  →  Quality (ROCE, OPM, D/E, IntCov) / Growth / EQ / Piotroski
Banks & NBFCs           →  Quality (ROE, NIM, ROA, Borrowing-D/E) / Growth / EQ / Piotroski

All scores are percentile-ranked within the universe (0 = worst, 100 = best).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.company import LENDERS, CompanyData, load_company

# Symbols that need bank-specific quality metrics
_BANK_SYMS = LENDERS


@dataclass
class ScoreResult:
    symbol:           str
    composite:        float
    quality:          float
    growth:           float
    earnings_quality: float
    piotroski:        int          # 0–9
    piotroski_pct:    float
    roce_pct:         float        # ROCE percentile (banks → ROE percentile)
    opm_pct:          float        # OPM percentile  (banks → NIM percentile)
    revenue_cagr_3y:  float
    eps_cagr_3y:      float
    fcf_cagr_3y:      float
    de_ratio:         float
    interest_coverage: float       # EBITDA/Interest (banks → ROA %)
    grade:            str          # A+/A/B+/B/C/D
    verdict:          str


class CompositeScorer:
    """Compute composite scores for a list of symbols, percentile-ranked."""

    def __init__(self, cache_dir: str = "data/raw/screener"):
        self.cache_dir = Path(cache_dir)

    def score_all(self, symbols: list[str],
                  data: Optional[dict[str, CompanyData]] = None) -> list[ScoreResult]:
        """
        Score `symbols` from the current cache, or from `data` (symbol →
        CompanyData) — the backtest passes historical as-of views.
        """
        if data is None:
            raw = {sym: self._extract(sym) for sym in symbols}
        else:
            # Historical mode: only the supplied views. Falling back to the cache
            # for a missing symbol would leak today's data into the past.
            raw = {sym: self._extract(sym, data[sym]) for sym in symbols if sym in data}
        raw = {k: v for k, v in raw.items() if v is not None}
        if not raw:
            return []
        results = self._rank_and_score(raw, symbols)
        return sorted(results, key=lambda r: r.composite, reverse=True)

    # ── Per-company raw metric extraction ─────────────────────────────────────

    def _extract(self, symbol: str, cd: Optional[CompanyData] = None) -> Optional[dict]:
        cd = cd or load_company(symbol, cache_dir=self.cache_dir)
        if cd is None:
            return None

        np_ser  = cd.series("profit_loss", "Net Profit")
        eps_ser = cd.series("profit_loss", "EPS in Rs")
        int_ser = cd.series("profit_loss", "Interest")
        dep_ser = cd.series("profit_loss", "Depreciation")
        pbt_ser = cd.series("profit_loss", "Profit before tax")
        cfo_ser = cd.series("cash_flow", "Cash from Operating Activity")
        fcf_ser = cd.series("cash_flow", "Free Cash Flow")
        # Balance sheet — banks store "Borrowing" (singular)
        borr_ser   = cd.series("balance_sheet", "Borrowings", "Borrowing")
        eq_cap     = cd.series("balance_sheet", "Equity Capital")
        reserves   = cd.series("balance_sheet", "Reserves")
        tot_assets = cd.series("balance_sheet", "Total Assets")
        eq_book    = (eq_cap[-1] + reserves[-1]) if eq_cap and reserves else 1
        de_ratio   = borr_ser[-1] / eq_book if borr_ser and eq_book > 0 else 0.0

        if cd.is_lender:
            return self._extract_bank(
                cd, np_ser, eps_ser, int_ser, cfo_ser, fcf_ser,
                borr_ser, eq_cap, reserves, tot_assets, de_ratio,
            )

        # ── Non-financial extraction ──────────────────────────────────────────
        sales_ser = cd.series("profit_loss", "Sales")
        opm_ser   = cd.series("profit_loss", "OPM %")
        roce_ser  = cd.series("ratios", "ROCE %")

        roce_3y = self._tail_avg(roce_ser, 3)
        opm_3y  = self._tail_avg(opm_ser, 3)

        if pbt_ser and int_ser and dep_ser:
            ebitda  = pbt_ser[-1] + int_ser[-1] + dep_ser[-1]
            int_cov = ebitda / int_ser[-1] if int_ser[-1] > 0 else 99.0
        else:
            int_cov = 99.0

        rev_cagr = self._cagr(sales_ser[-4:]) if len(sales_ser) >= 2 else 0.0
        eps_cagr = self._cagr(eps_ser[-4:])   if len(eps_ser)   >= 2 else 0.0
        fcf_cagr = (self._cagr(fcf_ser[-4:]) if len(fcf_ser) >= 4
                    else self._cagr(cfo_ser[-4:]) if len(cfo_ser) >= 4 else 0.0)

        cfo_np_ratio  = (cfo_ser[-1] / np_ser[-1]
                         if cfo_ser and np_ser and np_ser[-1] and np_ser[-1] > 0 else 0.0)
        accrual_ratio = 0.0
        if np_ser and cfo_ser and tot_assets and len(tot_assets) >= 2:
            avg_ta = (tot_assets[-1] + tot_assets[-2]) / 2
            accrual_ratio = (np_ser[-1] - cfo_ser[-1]) / avg_ta if avg_ta > 0 else 0.0

        piotroski = self._piotroski_std(
            np_ser, cfo_ser, borr_ser, tot_assets, opm_ser, sales_ser, eps_ser,
        )

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
            "piotroski":     piotroski,
            "is_bank":       False,
        }

    def _extract_bank(
        self, cd: CompanyData,
        np_ser, eps_ser, int_ser, cfo_ser, fcf_ser,
        borr_ser, eq_cap, reserves, tot_assets, de_ratio,
    ) -> dict:
        """Bank-specific metrics: ROE → roce_3y, NIM → opm_3y, ROA → int_cov."""
        roe_ser = cd.series("ratios", "ROE %")
        rev_ser = cd.series("profit_loss", "Revenue")    # banks: total interest income

        # ROE quality metric (3Y avg)
        roe_3y = self._tail_avg(roe_ser, 3)
        if not roe_3y and np_ser and eq_cap and reserves:
            bv = eq_cap[-1] + reserves[-1]
            roe_3y = np_ser[-1] / bv * 100 if bv > 0 else 0.0

        # NIM = Net Interest Income / Total Assets (3Y avg)
        nim_3y = 0.0
        if rev_ser and int_ser and tot_assets:
            nii_ser = [r - i for r, i in zip(rev_ser, int_ser)]
            nim_raw = [
                nii / a * 100
                for nii, a in zip(nii_ser, tot_assets)
                if a and a > 0
            ]
            nim_3y = self._tail_avg(nim_raw, 3)

        # ROA = Net Profit / Total Assets (3Y avg, %)
        roa_3y = 0.0
        if np_ser and tot_assets:
            roa_raw = [
                np_ / a * 100
                for np_, a in zip(np_ser, tot_assets)
                if a and a > 0
            ]
            roa_3y = self._tail_avg(roa_raw, 3)

        # Growth: use Revenue (not Sales) for banks
        rev_cagr = self._cagr(rev_ser[-4:]) if len(rev_ser) >= 2 else 0.0
        eps_cagr = self._cagr(eps_ser[-4:]) if len(eps_ser) >= 2 else 0.0
        # NBFCs & banks: FCF is meaningless; use NP growth as FCF proxy
        fcf_cagr = self._cagr(np_ser[-4:]) if len(np_ser) >= 2 else 0.0

        # Earnings quality: CFO/NP ratio and accruals
        cfo_np_ratio  = 0.0
        accrual_ratio = 0.0
        # For banks, CFO is inflated by deposit inflows — use NP/Revenue as quality signal
        if np_ser and rev_ser and rev_ser[-1] > 0:
            cfo_np_ratio = np_ser[-1] / rev_ser[-1]  # net profit margin (quality proxy)
        if np_ser and tot_assets and len(tot_assets) >= 2:
            avg_ta = (tot_assets[-1] + tot_assets[-2]) / 2
            # Accrual ratio for banks: NP growth vs asset growth (lower divergence = better)
            if len(np_ser) >= 2 and avg_ta > 0:
                np_growth   = (np_ser[-1] - np_ser[-2]) / abs(np_ser[-2]) if np_ser[-2] else 0
                asset_growth = (tot_assets[-1] - tot_assets[-2]) / tot_assets[-2]
                accrual_ratio = np_growth - asset_growth   # positive = earnings outpace assets (bad)

        piotroski = self._piotroski_bank(np_ser, cfo_ser, borr_ser, tot_assets, roe_ser, rev_ser)

        return {
            "roce_3y":       roe_3y,     # re-uses field; holds ROE for banks
            "opm_3y":        nim_3y,     # re-uses field; holds NIM for banks
            "de_ratio":      de_ratio,
            "int_cov":       roa_3y,     # re-uses field; holds ROA for banks
            "rev_cagr":      rev_cagr,
            "eps_cagr":      eps_cagr,
            "fcf_cagr":      fcf_cagr,
            "cfo_np_ratio":  cfo_np_ratio,
            "accrual_ratio": accrual_ratio,
            "piotroski":     piotroski,
            "is_bank":       True,
        }

    # ── Piotroski variants ────────────────────────────────────────────────────

    def _piotroski_std(self, np_ser, cfo_ser, borr_ser, tot_assets, opm_ser, sales_ser, eps_ser):
        score = 0
        if np_ser and tot_assets and tot_assets[-1]:
            roa_now = np_ser[-1] / tot_assets[-1]
            score  += 1 if roa_now > 0 else 0
            if len(np_ser) >= 2 and len(tot_assets) >= 2:
                score += 1 if roa_now > np_ser[-2] / tot_assets[-2] else 0
        if cfo_ser:
            score += 1 if cfo_ser[-1] > 0 else 0
        if np_ser and cfo_ser and tot_assets:
            score += 1 if cfo_ser[-1] / tot_assets[-1] > np_ser[-1] / tot_assets[-1] else 0
        if borr_ser and tot_assets and len(borr_ser) >= 2 and len(tot_assets) >= 2:
            score += 1 if borr_ser[-1] / tot_assets[-1] < borr_ser[-2] / tot_assets[-2] else 0
        if opm_ser and len(opm_ser) >= 2:
            score += 1 if opm_ser[-1] > opm_ser[-2] else 0
        if sales_ser and tot_assets and len(sales_ser) >= 2 and len(tot_assets) >= 2:
            score += 1 if (sales_ser[-1] / tot_assets[-1] > sales_ser[-2] / tot_assets[-2]) else 0
        if eps_ser and np_ser and len(eps_ser) >= 2 and len(np_ser) >= 2:
            eps_g = (eps_ser[-1] - eps_ser[-2]) / abs(eps_ser[-2]) if eps_ser[-2] else 0
            np_g  = (np_ser[-1]  - np_ser[-2])  / abs(np_ser[-2])  if np_ser[-2] else 0
            score += 1 if eps_g >= np_g - 0.03 else 0
        if cfo_ser and len(cfo_ser) >= 3:
            score += 1 if cfo_ser[-1] >= cfo_ser[-3] else 0
        return score

    def _piotroski_bank(self, np_ser, cfo_ser, borr_ser, tot_assets, roe_ser, rev_ser):
        """Adapted Piotroski for banks: replaces CFO/OPM checks with ROE/NIM checks."""
        score = 0
        # 1. Positive ROA
        if np_ser and tot_assets and tot_assets[-1]:
            score += 1 if np_ser[-1] / tot_assets[-1] > 0 else 0
        # 2. ROA improving
        if np_ser and tot_assets and len(np_ser) >= 2 and len(tot_assets) >= 2:
            roa_now  = np_ser[-1] / tot_assets[-1]
            roa_prev = np_ser[-2] / tot_assets[-2]
            score   += 1 if roa_now > roa_prev else 0
        # 3. Positive NP
        if np_ser:
            score += 1 if np_ser[-1] > 0 else 0
        # 4. NP growth accelerating (proxy for earnings quality)
        if np_ser and len(np_ser) >= 3:
            g1 = np_ser[-1] - np_ser[-2]
            g2 = np_ser[-2] - np_ser[-3]
            score += 1 if g1 > g2 else 0
        # 5. Leverage (Borrowings/Assets) not increasing
        if borr_ser and tot_assets and len(borr_ser) >= 2 and len(tot_assets) >= 2:
            score += 1 if (borr_ser[-1] / tot_assets[-1] <= borr_ser[-2] / tot_assets[-2]) else 0
        # 6. ROE improving or stable (≥ 12%)
        if roe_ser and len(roe_ser) >= 2:
            score += 1 if roe_ser[-1] >= 12 else 0
        # 7. Revenue (interest income) growing
        if rev_ser and len(rev_ser) >= 2:
            score += 1 if rev_ser[-1] > rev_ser[-2] else 0
        # 8. No share dilution
        # 9. NP CAGR > Revenue CAGR (operating leverage improving)
        if np_ser and rev_ser and len(np_ser) >= 3 and len(rev_ser) >= 3:
            np_g  = (np_ser[-1]  / np_ser[-3])  if np_ser[-3]  > 0 else 1
            rev_g = (rev_ser[-1] / rev_ser[-3]) if rev_ser[-3] > 0 else 1
            score += 1 if np_g >= rev_g else 0
            score += 1 if np_ser[-1] / rev_ser[-1] >= np_ser[-3] / rev_ser[-3] else 0
        return min(score, 9)

    # ── Cross-sectional percentile ranking ────────────────────────────────────

    def _rank_and_score(self, raw: dict[str, dict], all_syms: list[str]) -> list[ScoreResult]:
        syms = list(raw.keys())
        # Peer group per symbol. Lenders store ROE/NIM/ROA/margin/bank-Piotroski in
        # the same fields non-lenders use for ROCE/OPM/IntCov/CFO-NP/Piotroski, and
        # a bank's 4% NIM ranked against a manufacturer's 20% OPM always lands at
        # the bottom. Those metrics are therefore ranked within the peer group.
        groups = np.array([raw[s]["is_bank"] for s in syms])

        def _rank(arr: np.ndarray) -> np.ndarray:
            ranks = np.array([np.sum(arr < v) for v in arr], dtype=float)
            return ranks / max(len(arr) - 1, 1) * 100

        def _pct(vals: list[float], higher_better: bool = True,
                 by_peer: bool = False) -> np.ndarray:
            arr  = np.array(vals, dtype=float)
            pcts = np.empty_like(arr)
            if by_peer:
                for g in (True, False):
                    mask = groups == g
                    if mask.any():
                        pcts[mask] = _rank(arr[mask])
            else:
                pcts = _rank(arr)
            return pcts if higher_better else (100 - pcts)

        roce_pct = _pct([raw[s]["roce_3y"]       for s in syms], by_peer=True)
        opm_pct  = _pct([raw[s]["opm_3y"]        for s in syms], by_peer=True)
        de_pct   = _pct([raw[s]["de_ratio"]       for s in syms], higher_better=False, by_peer=True)
        ic_pct   = _pct([raw[s]["int_cov"]        for s in syms], by_peer=True)
        rev_pct  = _pct([raw[s]["rev_cagr"]       for s in syms])
        eps_pct  = _pct([raw[s]["eps_cagr"]       for s in syms])
        fcf_pct  = _pct([raw[s]["fcf_cagr"]       for s in syms])
        cfo_pct  = _pct([raw[s]["cfo_np_ratio"]   for s in syms], by_peer=True)
        acc_pct  = _pct([raw[s]["accrual_ratio"]  for s in syms], higher_better=False, by_peer=True)
        pio_pct  = _pct([raw[s]["piotroski"]      for s in syms], by_peer=True)

        results = []
        for i, sym in enumerate(syms):
            q  = 0.40 * roce_pct[i] + 0.30 * opm_pct[i] + 0.20 * de_pct[i] + 0.10 * ic_pct[i]
            g  = 0.35 * rev_pct[i]  + 0.40 * eps_pct[i]  + 0.25 * fcf_pct[i]
            eq = 0.60 * cfo_pct[i]  + 0.40 * acc_pct[i]
            p  = pio_pct[i]

            composite = max(0, min(100, 0.35 * q + 0.30 * g + 0.20 * eq + 0.15 * p))

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

        scored = {r.symbol for r in results}
        for sym in all_syms:
            if sym not in scored:
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
            changes = [
                (vals[i + 1] - vals[i]) / abs(vals[i])
                for i in range(len(vals) - 1) if vals[i] != 0
            ]
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
