"""
Dataset report generator — reads from cached raw data (no API calls).

Outputs:
  1. Valuation table    — Mkt Cap, PE, P/B, EV/EBITDA, ROE, ROCE, D/E, Div Yield
  2. P&L Income table   — Revenue, EBITDA, EBITDA%, Net Profit, EPS, Div Payout
  3. P&L Cost table     — Depreciation, Interest, Other Income, PBT, Tax%
  4. Cash Flow table    — CFO, CFI, CFF, Net CF, Borrowings, Total Assets
  5. CSV dataset        — all columns combined, saved to reports/

Banks use a different P&L structure (Revenue / Financing Profit) — handled automatically.
All ANSI colour codes are handled with visual-width padding so columns align correctly.
"""

import csv
import json
import os
import re as _re
from datetime import datetime
from typing import Any

from config.nifty50_tickers import NIFTY50_TICKERS
from src.utils.helpers import safe_float
from src.utils.logger import get_logger

logger = get_logger(__name__)

CR = 1e7  # 1 Crore

# Companies that use bank P&L structure on Screener (Revenue / Financing Profit keys)
# Insurance (HDFCLIFE, SBILIFE) and Bajaj Finserv use corporate keys (Sales / Operating Profit)
BANK_SYMBOLS = {
    "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN",
    "INDUSINDBK", "BAJFINANCE", "SHRIRAMFIN",
}


# ── ANSI-safe alignment helpers ───────────────────────────────────────────────
_ANSI_RE = _re.compile(r'\x1B\[[0-9;]*[a-zA-Z]')

def _vlen(s: str) -> int:
    """Visible character length — strips ANSI escape codes before measuring."""
    return len(_ANSI_RE.sub('', s))

def _rj(s: str, w: int) -> str:
    """Right-justify string to visual width w (ANSI-safe)."""
    return ' ' * max(0, w - _vlen(s)) + s

def _lj(s: str, w: int) -> str:
    """Left-justify string to visual width w (ANSI-safe)."""
    return s + ' ' * max(0, w - _vlen(s))

def _tr(s: str, n: int) -> str:
    """Truncate to n chars, appending '…' if cut."""
    return (s[:n - 1] + '…') if len(s) > n else s


# ── Colour helpers ────────────────────────────────────────────────────────────
def _g(s): return f"\033[92m{s}\033[0m"
def _r(s): return f"\033[91m{s}\033[0m"
def _y(s): return f"\033[93m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"
def _c(s): return f"\033[96m{s}\033[0m"
def _d(s): return f"\033[2m{s}\033[0m"


# ── Value formatters (return raw colored string, NO internal padding) ─────────
def _mc_fmt(mc):
    """Market cap → '₹18,433K Cr' or '₹999 Cr'."""
    if mc is None: return _d('—')
    return f'₹{mc / 100:,.0f}K Cr' if mc >= 100 else f'₹{mc:,.0f} Cr'

def _pe_fmt(v):
    if v is None or v == 0: return _d('—')
    s = f'{v:.1f}x'
    return _g(s) if v < 20 else (_y(s) if v < 35 else _r(s))

def _x_fmt(v, dec=1):
    """Plain multiplier (e.g. P/B, D/E, EV/EBITDA)."""
    if v is None or v == 0: return _d('—')
    return f'{v:.{dec}f}x'

def _pct_fmt(v, good=15, warn=8):
    if v is None: return _d('—')
    s = f'{v:.1f}%'
    return _g(s) if v >= good else (_y(s) if v >= warn else _r(s))

def _margin_fmt(v, good=20, warn=12):
    """EBITDA margin colour: green ≥20%, yellow 12–20%, red <12%."""
    if v is None: return _d('—')
    s = f'{v:.1f}%'
    return _g(s) if v >= good else (_y(s) if v >= warn else _r(s))

def _n_fmt(v, dec=0):
    """Plain comma number for crore values — no ₹/Cr prefix (unit shown in header)."""
    if v is None: return _d('—')
    return f'{v:,.{dec}f}'

def _eps_fmt(v):
    if v is None or v == 0: return _d('—')
    return f'{v:.2f}'

def _p_fmt(v, suf='%', dec=1):
    """Generic percentage/ratio formatter — dash if None or zero."""
    if v is None or v == 0: return _d('—')
    return f'{v:.{dec}f}{suf}'

def _tax_fmt(v):
    if v is None: return _d('—')
    s = f'{v:.0f}%'
    return _r(s) if v < 0 else s


# ── Data loading ──────────────────────────────────────────────────────────────
def _yf(sym):
    p = f"data/raw/financials/{sym}_NS.json"
    return json.load(open(p)) if os.path.exists(p) else {}

def _sc(sym):
    p = f"data/raw/screener/{sym}.json"
    return json.load(open(p)) if os.path.exists(p) else {}

def _sc_val(section: dict, year: str, row: str) -> float | None:
    years = section.get("years", [])
    rows  = section.get("rows", {})
    # Exact match first
    if year in years:
        idx = years.index(year)
    else:
        # Fallback: match any label that starts with the target year (e.g. "Mar 2025 10m")
        idx = next((i for i, y in enumerate(years) if y.startswith(year)), None)
        if idx is None:
            return None
    vals = rows.get(row, [])
    return safe_float(vals[idx]) if idx < len(vals) else None

def _sc_latest(section: dict, row: str) -> float | None:
    """Return the most recent non-null value for a row."""
    years = section.get("years", [])
    rows  = section.get("rows", {})
    vals  = rows.get(row, [])
    for i in range(len(years) - 1, -1, -1):
        v = safe_float(vals[i]) if i < len(vals) else None
        if v is not None:
            return v
    return None


# ── Build dataset ─────────────────────────────────────────────────────────────
def build_dataset(pl_years: list[str] | None = None) -> list[dict[str, Any]]:
    if pl_years is None:
        pl_years = ["Mar 2021", "Mar 2022", "Mar 2023", "Mar 2024", "Mar 2025"]

    rows = []
    for sym, company in NIFTY50_TICKERS.items():
        yf_d  = _yf(sym)
        sc_d  = _sc(sym)
        rat   = yf_d.get("key_ratios", {})
        info  = yf_d.get("info", {})
        pl    = sc_d.get("profit_loss", {})
        bs    = sc_d.get("balance_sheet", {})
        cf    = sc_d.get("cash_flow", {})
        sc_r  = sc_d.get("ratios", {})
        is_bank = sym in BANK_SYMBOLS

        # ── Revenue row name differs for banks ──
        rev_key   = "Revenue"          if is_bank else "Sales"
        ebitda_key= "Financing Profit" if is_bank else "Operating Profit"
        margin_key= "Financing Margin %" if is_bank else "OPM %"

        row: dict[str, Any] = {
            "Symbol":       sym,
            "Company":      company,
            "Sector":       info.get("sector", "—"),
            "Type":         "Bank/NBFC" if is_bank else "Corp",
            # ── Valuation ──
            "PE":           safe_float(rat.get("pe_ratio"))   or None,
            "Fwd PE":       safe_float(rat.get("forward_pe")) or None,
            "PB":           safe_float(rat.get("pb_ratio"))   or None,
            "EV/EBITDA":    safe_float(rat.get("ev_ebitda"))  or None,
            "Mkt Cap Cr":   (safe_float(rat.get("market_cap")) / CR)
                            if rat.get("market_cap") else None,
            "52W High":     safe_float(rat.get("52w_high")),
            "52W Low":      safe_float(rat.get("52w_low")),
            "Beta":         safe_float(rat.get("beta"))       or None,
            "Div Yield %":  (safe_float(rat.get("dividend_yield")) * 100)
                            if rat.get("dividend_yield") else None,
            # ── Return ratios ──
            "ROE %":        (safe_float(rat.get("roe")) * 100)
                            if rat.get("roe") else None,
            "ROA %":        (safe_float(rat.get("roa")) * 100)
                            if rat.get("roa") else None,
            "ROCE % (SC)":  _sc_latest(sc_r, "ROCE %"),
            "ROE % (SC)":   _sc_latest(sc_r, "ROE %"),
            "D/E":          safe_float(rat.get("debt_to_equity")) or None,
            "Curr Ratio":   safe_float(rat.get("current_ratio")) or None,
            "Prof Margin %":(safe_float(rat.get("profit_margin")) * 100)
                            if rat.get("profit_margin") else None,
            "Op Margin %":  (safe_float(rat.get("operating_margin")) * 100)
                            if rat.get("operating_margin") else None,
        }

        # ── P&L for each year from Screener ──
        for yr in pl_years:
            short = yr[-4:]
            row[f"Revenue {short}"]      = _sc_val(pl, yr, rev_key)
            row[f"EBITDA {short}"]       = _sc_val(pl, yr, ebitda_key)
            row[f"EBITDA Margin {short}"]= _sc_val(pl, yr, margin_key)
            row[f"Other Income {short}"] = _sc_val(pl, yr, "Other Income")
            row[f"Depreciation {short}"] = _sc_val(pl, yr, "Depreciation")
            row[f"Interest {short}"]     = _sc_val(pl, yr, "Interest")
            row[f"PBT {short}"]          = _sc_val(pl, yr, "Profit before tax")
            row[f"Tax% {short}"]         = _sc_val(pl, yr, "Tax %")
            row[f"Net Profit {short}"]   = _sc_val(pl, yr, "Net Profit")
            row[f"EPS {short}"]          = _sc_val(pl, yr, "EPS in Rs")
            row[f"Div Payout% {short}"]  = _sc_val(pl, yr, "Dividend Payout %")

        # ── Balance Sheet (latest year from Screener) ──
        row["Total Assets Cr"]    = _sc_latest(bs, "Total Assets")
        row["Borrowings Cr"]      = _sc_latest(bs, "Borrowings")
        row["Equity + Res Cr"]    = None
        eq  = _sc_latest(bs, "Equity Capital")
        res = _sc_latest(bs, "Reserves")
        if eq is not None and res is not None:
            row["Equity + Res Cr"] = eq + res
        row["CWIP Cr"]            = _sc_latest(bs, "CWIP")
        row["Investments Cr"]     = _sc_latest(bs, "Investments")

        # ── Cash Flow (latest year from Screener) ──
        row["CFO Cr"]      = _sc_latest(cf, "Cash from Operating Activity")
        row["CFI Cr"]      = _sc_latest(cf, "Cash from Investing Activity")
        row["CFF Cr"]      = _sc_latest(cf, "Cash from Financing Activity")
        row["Net CF Cr"]   = _sc_latest(cf, "Net Cash Flow")

        # ── Screener Ratios — historical (latest) ──
        row["Debtor Days"]       = _sc_latest(sc_r, "Debtor Days")
        row["WC Days"]           = _sc_latest(sc_r, "Working Capital Days")
        row["Cash Conv Cycle"]   = _sc_latest(sc_r, "Cash Conversion Cycle")

        rows.append(row)
    return rows


# ── Terminal tables ───────────────────────────────────────────────────────────
# Column widths (visual chars).  All colored strings use _rj/_lj so ANSI
# escape codes never inflate Python's padding calculations.

# TABLE 1 — VALUATION
# Total width: 2+11+2+28+2+12+2+8+2+7+2+6+2+8+2+6+2+6+2+8+2+7 = 123
_V_SYM, _V_CO, _V_MC  = 11, 28, 12
_V_PE,  _V_FPE, _V_PB = 8,  7,  6
_V_EV,  _V_ROE, _V_RO = 8,  6,  6
_V_DE,  _V_DIV         = 8,  7

def _vsep() -> str:
    w = 2+_V_SYM+2+_V_CO+2+_V_MC+2+_V_PE+2+_V_FPE+2+_V_PB+2+_V_EV+2+_V_ROE+2+_V_RO+2+_V_DE+2+_V_DIV
    return "  " + "─" * (w - 2)

def print_valuation_table(rows: list[dict]) -> None:
    W = 2+_V_SYM+2+_V_CO+2+_V_MC+2+_V_PE+2+_V_FPE+2+_V_PB+2+_V_EV+2+_V_ROE+2+_V_RO+2+_V_DE+2+_V_DIV
    print()
    print(_b(_c("═" * W)))
    print(_b(_c(f"  NIFTY 50 — VALUATION SNAPSHOT  |  {datetime.now().strftime('%d %b %Y')}")))
    print(_b(_c("═" * W)))
    # Header (no ANSI in raw strings → normal f-string alignment works)
    print("  " + _b(
        _lj("Symbol", _V_SYM) + "  " + _lj("Company", _V_CO) + "  " +
        _rj("Mkt Cap (Cr)", _V_MC) + "  " + _rj("PE", _V_PE) + "  " +
        _rj("Fwd PE", _V_FPE) + "  " + _rj("P/B", _V_PB) + "  " +
        _rj("EV/EBITDA", _V_EV) + "  " + _rj("ROE%", _V_ROE) + "  " +
        _rj("ROCE%", _V_RO) + "  " + _rj("D/E", _V_DE) + "  " +
        _rj("Div Yld%", _V_DIV)
    ))
    print(_vsep())
    for r in sorted(rows, key=lambda x: x.get("Mkt Cap Cr") or 0, reverse=True):
        roe  = r.get("ROE % (SC)") or r.get("ROE %")
        roce = r.get("ROCE % (SC)")
        print(
            "  " +
            _lj(r['Symbol'], _V_SYM) + "  " +
            _lj(_tr(r['Company'], _V_CO), _V_CO) + "  " +
            _rj(_mc_fmt(r["Mkt Cap Cr"]), _V_MC) + "  " +
            _rj(_pe_fmt(r['PE']), _V_PE) + "  " +
            _rj(_x_fmt(r['Fwd PE']), _V_FPE) + "  " +
            _rj(_x_fmt(r['PB']), _V_PB) + "  " +
            _rj(_x_fmt(r['EV/EBITDA']), _V_EV) + "  " +
            _rj(_pct_fmt(roe), _V_ROE) + "  " +
            _rj(_pct_fmt(roce), _V_RO) + "  " +
            _rj(_x_fmt(r['D/E']), _V_DE) + "  " +
            _rj(_p_fmt(r['Div Yield %']), _V_DIV)
        )
    print(_vsep())
    print(_d("  Colour — PE: green <20x  yellow 20–35x  red >35x  |  ROE/ROCE: green ≥15%  yellow 8–15%  red <8%"))
    print()


# TABLE 2a — P&L INCOME SUMMARY
# Numbers in ₹ Crore (plain integers, unit shown in header — no ₹/Cr in cells)
# Total width: 2+11+2+26+2+10+2+10+2+7+2+10+2+8+2+7 = 107
_PA_SYM, _PA_CO         = 11, 26
_PA_REV, _PA_EB, _PA_EM = 10, 10,  7
_PA_NP,  _PA_EPS, _PA_DP = 10, 8,  7

def _pa_sep() -> str:
    w = 2+_PA_SYM+2+_PA_CO+2+_PA_REV+2+_PA_EB+2+_PA_EM+2+_PA_NP+2+_PA_EPS+2+_PA_DP
    return "  " + "─" * (w - 2)

def print_pl_income_table(rows: list[dict], year: str = "2025") -> None:
    W = 2+_PA_SYM+2+_PA_CO+2+_PA_REV+2+_PA_EB+2+_PA_EM+2+_PA_NP+2+_PA_EPS+2+_PA_DP
    print()
    print(_b(_c("═" * W)))
    print(_b(_c(f"  NIFTY 50 — P&L INCOME SUMMARY  |  FY{year}  (all ₹ values in Crore)")))
    print(_b(_c("═" * W)))
    print("  " + _b(
        _lj("Symbol", _PA_SYM) + "  " + _lj("Company", _PA_CO) + "  " +
        _rj("Revenue", _PA_REV) + "  " + _rj("EBITDA", _PA_EB) + "  " +
        _rj("EBITDA%", _PA_EM) + "  " + _rj("Net Profit", _PA_NP) + "  " +
        _rj("EPS (₹)", _PA_EPS) + "  " + _rj("DivPay%", _PA_DP)
    ))
    print(_pa_sep())

    valid   = sorted(
        [r for r in rows if r.get(f"Revenue {year}")],
        key=lambda x: x.get(f"Revenue {year}") or 0, reverse=True
    )
    missing = [r for r in rows if not r.get(f"Revenue {year}")]

    for r in valid:
        def v(k, yr=year): return r.get(f"{k} {yr}")
        print(
            "  " +
            _lj(r['Symbol'], _PA_SYM) + "  " +
            _lj(_tr(r['Company'], _PA_CO), _PA_CO) + "  " +
            _rj(_n_fmt(v('Revenue')), _PA_REV) + "  " +
            _rj(_n_fmt(v('EBITDA')), _PA_EB) + "  " +
            _rj(_margin_fmt(v('EBITDA Margin')), _PA_EM) + "  " +
            _rj(_n_fmt(v('Net Profit')), _PA_NP) + "  " +
            _rj(_eps_fmt(v('EPS')), _PA_EPS) + "  " +
            _rj(_p_fmt(v('Div Payout%'), dec=0), _PA_DP)
        )
    if missing:
        print(_d(f"\n  No FY{year} data: {', '.join(r['Symbol'] for r in missing)}"))
    print(_pa_sep())
    print(_d(f"  EBITDA%: green ≥20%  yellow 12–20%  red <12%  |  Note: for banks EBITDA = Financing Profit"))
    print()


# TABLE 2b — P&L COST BREAKDOWN
# Total width: 2+11+2+26+2+9+2+9+2+9+2+10+2+5 = 91
_PB_SYM, _PB_CO             = 11, 26
_PB_DEP, _PB_INT, _PB_OI   = 9,  9,  9
_PB_PBT, _PB_TAX             = 10, 5

def _pb_sep() -> str:
    w = 2+_PB_SYM+2+_PB_CO+2+_PB_DEP+2+_PB_INT+2+_PB_OI+2+_PB_PBT+2+_PB_TAX
    return "  " + "─" * (w - 2)

def print_pl_cost_table(rows: list[dict], year: str = "2025") -> None:
    W = 2+_PB_SYM+2+_PB_CO+2+_PB_DEP+2+_PB_INT+2+_PB_OI+2+_PB_PBT+2+_PB_TAX
    print()
    print(_b(_c("═" * W)))
    print(_b(_c(f"  NIFTY 50 — P&L COST BREAKDOWN  |  FY{year}  (all ₹ values in Crore)")))
    print(_b(_c("═" * W)))
    print("  " + _b(
        _lj("Symbol", _PB_SYM) + "  " + _lj("Company", _PB_CO) + "  " +
        _rj("Deprec", _PB_DEP) + "  " + _rj("Interest", _PB_INT) + "  " +
        _rj("Other Inc", _PB_OI) + "  " + _rj("PBT", _PB_PBT) + "  " +
        _rj("Tax%", _PB_TAX)
    ))
    print(_pb_sep())

    valid = sorted(
        [r for r in rows if r.get(f"Revenue {year}")],
        key=lambda x: x.get(f"Revenue {year}") or 0, reverse=True
    )
    for r in valid:
        def v(k, yr=year): return r.get(f"{k} {yr}")
        print(
            "  " +
            _lj(r['Symbol'], _PB_SYM) + "  " +
            _lj(_tr(r['Company'], _PB_CO), _PB_CO) + "  " +
            _rj(_n_fmt(v('Depreciation')), _PB_DEP) + "  " +
            _rj(_n_fmt(v('Interest')), _PB_INT) + "  " +
            _rj(_n_fmt(v('Other Income')), _PB_OI) + "  " +
            _rj(_n_fmt(v('PBT')), _PB_PBT) + "  " +
            _rj(_tax_fmt(v('Tax%')), _PB_TAX)
        )
    print(_pb_sep())
    print()


# TABLE 3 — CASH FLOW + BALANCE SHEET
# Total width: 2+11+2+24+2+10+2+10+2+10+2+10+2+10+2+12 = 115
_CF_SYM, _CF_CO              = 11, 24
_CF_CFO, _CF_CFI, _CF_CFF   = 10, 10, 10
_CF_NCF, _CF_BOR, _CF_AST   = 10, 10, 12

def _cf_sep() -> str:
    w = 2+_CF_SYM+2+_CF_CO+2+_CF_CFO+2+_CF_CFI+2+_CF_CFF+2+_CF_NCF+2+_CF_BOR+2+_CF_AST
    return "  " + "─" * (w - 2)

def print_cashflow_table(rows: list[dict]) -> None:
    W = 2+_CF_SYM+2+_CF_CO+2+_CF_CFO+2+_CF_CFI+2+_CF_CFF+2+_CF_NCF+2+_CF_BOR+2+_CF_AST
    print()
    print(_b(_c("═" * W)))
    print(_b(_c("  NIFTY 50 — CASH FLOW & BALANCE SHEET  (Latest Year, ₹ Crore)")))
    print(_b(_c("═" * W)))
    print("  " + _b(
        _lj("Symbol", _CF_SYM) + "  " + _lj("Company", _CF_CO) + "  " +
        _rj("Op CF", _CF_CFO) + "  " + _rj("Inv CF", _CF_CFI) + "  " +
        _rj("Fin CF", _CF_CFF) + "  " + _rj("Net CF", _CF_NCF) + "  " +
        _rj("Borrowings", _CF_BOR) + "  " + _rj("Total Assets", _CF_AST)
    ))
    print(_cf_sep())
    for r in sorted(rows, key=lambda x: x.get("Mkt Cap Cr") or 0, reverse=True):
        print(
            "  " +
            _lj(r['Symbol'], _CF_SYM) + "  " +
            _lj(_tr(r['Company'], _CF_CO), _CF_CO) + "  " +
            _rj(_n_fmt(r.get('CFO Cr')), _CF_CFO) + "  " +
            _rj(_n_fmt(r.get('CFI Cr')), _CF_CFI) + "  " +
            _rj(_n_fmt(r.get('CFF Cr')), _CF_CFF) + "  " +
            _rj(_n_fmt(r.get('Net CF Cr')), _CF_NCF) + "  " +
            _rj(_n_fmt(r.get('Borrowings Cr')), _CF_BOR) + "  " +
            _rj(_n_fmt(r.get('Total Assets Cr')), _CF_AST)
        )
    print(_cf_sep())
    print(_d("  Op CF = Cash from Operations  |  Inv CF = Investing  |  Fin CF = Financing"))
    print()


# TABLE 4 — 5-YEAR P&L TREND (single company)
# Total width: 2+6+2+10+2+10+2+7+2+9+2+9+2+10+2+8+2+10 = 99
_TR_YR, _TR_REV, _TR_EB  = 6,  10, 10
_TR_EM, _TR_DEP, _TR_INT = 7,  9,  9
_TR_NP, _TR_EPS, _TR_CFO = 10, 8,  10

def _tr_sep() -> str:
    w = 2+_TR_YR+2+_TR_REV+2+_TR_EB+2+_TR_EM+2+_TR_DEP+2+_TR_INT+2+_TR_NP+2+_TR_EPS+2+_TR_CFO
    return "  " + "─" * (w - 2)

def print_pl_trend(rows: list[dict], symbol: str) -> None:
    match = [r for r in rows if r["Symbol"] == symbol.upper()]
    if not match:
        print(f"  Symbol {symbol} not found.")
        return
    r = match[0]
    is_bank   = r["Type"] == "Bank/NBFC"
    eb_label  = "Fin Profit" if is_bank else "EBITDA"
    em_label  = "Fin Mgn%" if is_bank else "EBITDA%"
    W = 2+_TR_YR+2+_TR_REV+2+_TR_EB+2+_TR_EM+2+_TR_DEP+2+_TR_INT+2+_TR_NP+2+_TR_EPS+2+_TR_CFO
    print()
    print(_b(_c("═" * W)))
    print(_b(_c(f"  {r['Company']} ({r['Symbol']}) — 5-Year P&L Trend  (₹ Crore)")))
    print(_b(_c("═" * W)))
    print("  " + _b(
        _lj("FY", _TR_YR) + "  " + _rj("Revenue", _TR_REV) + "  " +
        _rj(eb_label, _TR_EB) + "  " + _rj(em_label, _TR_EM) + "  " +
        _rj("Deprec", _TR_DEP) + "  " + _rj("Interest", _TR_INT) + "  " +
        _rj("Net Profit", _TR_NP) + "  " + _rj("EPS (₹)", _TR_EPS) + "  " +
        _rj("Op CF", _TR_CFO)
    ))
    print(_tr_sep())
    for yr in ["2021", "2022", "2023", "2024", "2025"]:
        def v(k, y=yr): return r.get(f"{k} {y}")
        print(
            "  " +
            _lj(f"FY{yr}", _TR_YR) + "  " +
            _rj(_n_fmt(v('Revenue')), _TR_REV) + "  " +
            _rj(_n_fmt(v('EBITDA')), _TR_EB) + "  " +
            _rj(_margin_fmt(v('EBITDA Margin')), _TR_EM) + "  " +
            _rj(_n_fmt(v('Depreciation')), _TR_DEP) + "  " +
            _rj(_n_fmt(v('Interest')), _TR_INT) + "  " +
            _rj(_n_fmt(v('Net Profit')), _TR_NP) + "  " +
            _rj(_eps_fmt(v('EPS')), _TR_EPS) + "  " +
            _rj(_n_fmt(r.get('CFO Cr')), _TR_CFO)
        )
    print(_tr_sep())
    print()


# ── CSV export ────────────────────────────────────────────────────────────────
def save_csv(rows: list[dict], output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    path = os.path.join(output_dir, f"nifty50_dataset_{date_str}.csv")
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return path
