"""
Advanced terminal reports: DCF valuation, composite scores, FII/DII flows,
and portfolio optimizer output.

All tables use ANSI-safe width helpers so columns align correctly in terminal.
"""
from __future__ import annotations

import re as _re
from typing import Optional

from src.valuation.dcf_engine    import DCFResult
from src.scoring.composite_score import ScoreResult
from src.data_pipeline.fetchers.fii_dii_fetcher import FlowSummary, FlowDay
from src.valuation.portfolio_optimizer          import PortfolioResult


# ── ANSI helpers (shared with dataset_report) ─────────────────────────────────
_ANSI_RE = _re.compile(r'\x1B\[[0-9;]*[a-zA-Z]')

def _vlen(s: str) -> int:
    return len(_ANSI_RE.sub('', s))

def _rj(s: str, w: int) -> str:
    return ' ' * max(0, w - _vlen(s)) + s

def _lj(s: str, w: int) -> str:
    return s + ' ' * max(0, w - _vlen(s))

def _tr(s: str, n: int) -> str:
    return (s[:n - 1] + '…') if len(s) > n else s

def _g(s):  return f"\033[92m{s}\033[0m"
def _r(s):  return f"\033[91m{s}\033[0m"
def _y(s):  return f"\033[93m{s}\033[0m"
def _b(s):  return f"\033[1m{s}\033[0m"
def _c(s):  return f"\033[96m{s}\033[0m"
def _m(s):  return f"\033[95m{s}\033[0m"
def _d(s):  return f"\033[2m{s}\033[0m"
def _ul(s): return f"\033[4m{s}\033[0m"

def _pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None: return _d("—")
    s = f"{v:+.{decimals}f}%"
    return _g(s) if v >= 0 else _r(s)

def _money(v: Optional[float]) -> str:
    if v is None: return _d("—")
    if abs(v) >= 1000:
        return f"₹{v/100:,.0f}K Cr"
    return f"₹{v:,.0f} Cr"

def _verdict_color(verdict: str) -> str:
    v = verdict.upper()
    if "STRONG BUY"  in v or "TOP PICK" in v: return _g(_b(verdict))
    if "BUY"         in v:                     return _g(verdict)
    if "HOLD"        in v:                     return _y(verdict)
    if "SELL"        in v:                     return _r(verdict)
    if "AVOID"       in v:                     return _r(verdict)
    return _d(verdict)


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 1 — DCF VALUATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

_DCF_W = 100

def print_dcf_table(results: list[DCFResult]) -> None:
    # Columns: Symbol(10) | IV(10) | CMP(8) | MoS(8) | IV Range(20) | WACC(6) | FCF CAGR(9) | Verdict(14)
    _S  = 10
    _IV = 10
    _CM = 8
    _MOS = 8
    _RNG = 20
    _WC  = 6
    _FC  = 9
    _VD  = 14

    total = 2 + _S+2+_IV+2+_CM+2+_MOS+2+_RNG+2+_WC+2+_FC+2+_VD

    print()
    print(_b(_m("═" * total)))
    print(_b(_m("  NIFTY 50 — DCF INTRINSIC VALUE  |  10-Year Free Cash Flow Model  |  10,000 Monte Carlo Simulations")))
    print(_b(_m("═" * total)))
    print("  " + _b(
        _lj("Symbol",   _S)  + "  " +
        _rj("IV (₹)",   _IV) + "  " +
        _rj("CMP (₹)",  _CM) + "  " +
        _rj("MoS %",    _MOS)+ "  " +
        _lj("IV Range (P10–P90)", _RNG) + "  " +
        _rj("WACC",     _WC) + "  " +
        _rj("FCF CAGR", _FC) + "  " +
        _lj("Verdict",  _VD)
    ))
    print("  " + "─" * (total - 2))

    for r in sorted(results, key=lambda x: x.margin_of_safety or -99, reverse=True):
        if r.intrinsic_value <= 0:
            iv_s   = _d("—")
            cmp_s  = _d("—")
            mos_s  = _d("N/A")
            rng_s  = _d("—")
            wacc_s = _d("—")
            fc_s   = _d("—")
        else:
            iv_s  = f"{r.intrinsic_value:,.0f}"
            cmp_s = f"{r.cmp:,.0f}" if r.cmp else _d("—")

            if r.margin_of_safety is not None:
                pct = r.margin_of_safety * 100
                mos_s = _g(f"+{pct:.1f}%") if pct >= 0 else _r(f"{pct:.1f}%")
            else:
                mos_s = _d("—")

            lo = f"{r.iv_low:,.0f}" if r.iv_low > 0 else "?"
            hi = f"{r.iv_high:,.0f}" if r.iv_high > 0 else "?"
            rng_s  = f"{lo} – {hi}"
            wacc_s = f"{r.wacc*100:.1f}%"
            fc_s   = f"{r.fcf_cagr*100:+.1f}%"

        print(
            "  " +
            _lj(r.symbol, _S)  + "  " +
            _rj(iv_s,     _IV) + "  " +
            _rj(cmp_s,    _CM) + "  " +
            _rj(mos_s,    _MOS)+ "  " +
            _lj(rng_s,    _RNG)+ "  " +
            _rj(wacc_s,   _WC) + "  " +
            _rj(fc_s,     _FC) + "  " +
            _lj(_verdict_color(r.verdict), _VD)
        )

    print("  " + "─" * (total - 2))
    print(_d("  IV = Intrinsic Value per share  |  MoS = Margin of Safety  |  P10–P90 = 80% confidence interval"))
    print(_d("  Model: 10-yr FCF projection + Gordon Growth terminal value  |  WACC auto-estimated from balance sheet"))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 2 — COMPOSITE QUALITY SCORE LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def print_score_table(results: list[ScoreResult]) -> None:
    _RK  = 3
    _S   = 11
    _CS  = 7
    _GR  = 4
    _QL  = 7
    _GW  = 7
    _EQ  = 8
    _PIO = 4
    _RC  = 7
    _OM  = 7
    _DE  = 6
    _IC  = 5
    _VD  = 10

    total = 2 + _RK+2+_S+2+_CS+2+_GR+2+_QL+2+_GW+2+_EQ+2+_PIO+2+_RC+2+_OM+2+_DE+2+_IC+2+_VD

    def _score_color(v: float) -> str:
        s = f"{v:.1f}"
        if v >= 75: return _g(_b(s))
        if v >= 55: return _g(s)
        if v >= 35: return _y(s)
        return _r(s)

    def _grade_color(g: str) -> str:
        if g == "A+": return _g(_b(g))
        if g == "A":  return _g(g)
        if g.startswith("B"): return _y(g)
        return _r(g)

    def _de_color(v: float) -> str:
        s = f"{v:.2f}"
        if v < 0.3: return _g(s)
        if v < 1.0: return _y(s)
        return _r(s)

    def _ic_color(v: float) -> str:
        s = f"{v:.1f}x" if v < 99 else _d("∞")
        if v >= 5:  return _g(s if isinstance(s, str) else s)
        if v >= 2:  return _y(s)
        return _r(s)

    print()
    print(_b(_c("═" * total)))
    print(_b(_c("  NIFTY 50 — MULTI-FACTOR COMPOSITE SCORE  |  Quality · Growth · Earnings Quality · Piotroski")))
    print(_b(_c("═" * total)))
    print("  " + _b(
        _rj("#",        _RK) + "  " +
        _lj("Symbol",   _S)  + "  " +
        _rj("Score",    _CS) + "  " +
        _lj("Gr",       _GR) + "  " +
        _rj("Quality",  _QL) + "  " +
        _rj("Growth",   _GW) + "  " +
        _rj("Earn Q",   _EQ) + "  " +
        _rj("Pio",      _PIO)+ "  " +
        _rj("ROCE%",    _RC) + "  " +
        _rj("OPM%",     _OM) + "  " +
        _rj("D/E",      _DE) + "  " +
        _rj("IC",       _IC) + "  " +
        _lj("Verdict",  _VD)
    ))
    print("  " + "─" * (total - 2))

    for i, r in enumerate(results, 1):
        if r.grade == "N/A":
            continue
        print(
            "  " +
            _rj(str(i),                       _RK) + "  " +
            _lj(r.symbol,                     _S)  + "  " +
            _rj(_score_color(r.composite),    _CS) + "  " +
            _lj(_grade_color(r.grade),        _GR) + "  " +
            _rj(f"{r.quality:.1f}",           _QL) + "  " +
            _rj(f"{r.growth:.1f}",            _GW) + "  " +
            _rj(f"{r.earnings_quality:.1f}",  _EQ) + "  " +
            _rj(f"{r.piotroski}/9",           _PIO)+ "  " +
            _rj(f"{r.roce_pct:.0f}th",        _RC) + "  " +
            _rj(f"{r.opm_pct:.0f}th",         _OM) + "  " +
            _rj(_de_color(r.de_ratio),        _DE) + "  " +
            _rj(_ic_color(r.interest_coverage), _IC) + "  " +
            _lj(_verdict_color(r.verdict),    _VD)
        )

    print("  " + "─" * (total - 2))
    print(_d("  Score = 0–100 percentile rank within Nifty 50  |  Quality/Growth/Earn Q = pillar scores"))
    print(_d("  Pio = Piotroski F-Score (0–9)  |  ROCE/OPM = percentile rank  |  IC = Interest Coverage (EBITDA/Interest)"))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 3 — FII / DII INSTITUTIONAL FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def print_fii_dii_table(summary: FlowSummary) -> None:
    total = 100

    signal_color = {
        "RISK-ON":  _g,
        "RISK-OFF": _r,
        "MIXED":    _y,
        "NEUTRAL":  _d,
        "UNKNOWN":  _d,
    }.get(summary.market_signal, _d)

    print()
    print(_b(_c("═" * total)))
    print(_b(_c(f"  FII / DII INSTITUTIONAL FLOW  |  Last 30 Trading Days  |  Signal: ") +
             signal_color(_b(summary.market_signal))))
    print(_b(_c("═" * total)))
    print(f"  {'Last updated:':<20} {summary.last_updated}")
    print()

    # Summary row
    def _net_color(v: float) -> str:
        s = f"{'+'if v>=0 else ''}{v:,.0f} Cr"
        return _g(s) if v >= 0 else _r(s)

    fii_trend_s = _g(summary.fii_trend) if "BUYING" in summary.fii_trend else (
                  _r(summary.fii_trend) if "SELLING" in summary.fii_trend else _y(summary.fii_trend))
    dii_trend_s = _g(summary.dii_trend) if "BUYING" in summary.dii_trend else (
                  _r(summary.dii_trend) if "SELLING" in summary.dii_trend else _y(summary.dii_trend))

    print(f"  {'Metric':<30} {'FII':>18}  {'DII':>18}  {'Combined':>18}")
    print("  " + "─" * 88)
    print(f"  {'30-Day Net Flow':<30} {_rj(_net_color(summary.fii_30d_net), 18)}  "
          f"{_rj(_net_color(summary.dii_30d_net), 18)}  "
          f"{_rj(_net_color(summary.total_30d_net), 18)}")
    print(f"  {'Trend':<30} {_rj(fii_trend_s, 18)}  {_rj(dii_trend_s, 18)}")
    print()

    if summary.days:
        _D  = 12
        _FB = 10
        _FS = 10
        _FN = 10
        _DB = 10
        _DS = 10
        _DN = 10
        _TN = 10

        print("  " + _b(
            _lj("Date",     _D)  + "  " +
            _rj("FII Buy",  _FB) + "  " +
            _rj("FII Sell", _FS) + "  " +
            _rj("FII Net",  _FN) + "  " +
            _rj("DII Buy",  _DB) + "  " +
            _rj("DII Sell", _DS) + "  " +
            _rj("DII Net",  _DN) + "  " +
            _rj("Total Net",_TN)
        ))
        print("  " + "─" * 88)

        for day in summary.days[:15]:
            print(
                "  " +
                _lj(day.date,                         _D)  + "  " +
                _rj(f"{day.fii_buy:,.0f}",            _FB) + "  " +
                _rj(f"{day.fii_sell:,.0f}",           _FS) + "  " +
                _rj(_net_color(day.fii_net),          _FN) + "  " +
                _rj(f"{day.dii_buy:,.0f}",            _DB) + "  " +
                _rj(f"{day.dii_sell:,.0f}",           _DS) + "  " +
                _rj(_net_color(day.dii_net),          _DN) + "  " +
                _rj(_net_color(day.total_net),        _TN)
            )

    print("  " + "─" * 88)
    print(_d("  All values in ₹ Crore  |  FII = Foreign Institutional Investors  |  DII = Domestic Institutional Investors"))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 4 — PORTFOLIO OPTIMIZER OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def print_portfolio_table(result: PortfolioResult) -> None:
    total = 90

    print()
    print(_b(_m("═" * total)))
    print(_b(_m(f"  OPTIMAL PORTFOLIO  |  Capital: ₹{result.total_capital:,.0f}  |  Markowitz Sharpe Maximisation")))
    print(_b(_m("═" * total)))
    print()

    # Stats banner
    ret_s = _g(f"{result.expected_return*100:.1f}%") if result.expected_return >= 0.12 else _y(f"{result.expected_return*100:.1f}%")
    vol_s = _g(f"{result.volatility*100:.1f}%")      if result.volatility <= 0.18 else _y(f"{result.volatility*100:.1f}%")
    sr_s  = _g(f"{result.sharpe_ratio:.2f}")          if result.sharpe_ratio >= 0.5 else _y(f"{result.sharpe_ratio:.2f}")

    print(f"  Expected Return (annual)  : {ret_s}")
    print(f"  Portfolio Volatility      : {vol_s}")
    print(f"  Sharpe Ratio              : {sr_s}")
    print(f"  Max Drawdown (est.)       : {_r(f'{result.max_drawdown_est*100:.1f}%')}")
    print(f"  Diversification           : {_g(result.diversification) if 'Well' in result.diversification else _y(result.diversification)}")
    print(f"  Verdict                   : {_verdict_color(result.verdict)}")
    print()

    _SY = 12
    _WG = 8
    _AM = 14
    _SH = 8
    _CM = 10

    print("  " + _b(
        _lj("Symbol",  _SY) + "  " +
        _rj("Weight",  _WG) + "  " +
        _rj("Amount",  _AM) + "  " +
        _rj("Shares",  _SH) + "  " +
        _rj("CMP (₹)", _CM)
    ))
    print("  " + "─" * (total - 2))

    bar_total = 30
    for a in result.allocations:
        bar_len = int(a.weight * bar_total)
        bar = _g("█" * bar_len) + _d("░" * (bar_total - bar_len))
        cmp_s = f"{a.cmp:,.0f}" if a.cmp else _d("—")
        sh_s  = str(a.shares)   if a.shares else _d("—")
        print(
            "  " +
            _lj(a.symbol,           _SY) + "  " +
            _rj(f"{a.weight*100:.1f}%", _WG) + "  " +
            _rj(f"₹{a.amount:,.0f}", _AM) + "  " +
            _rj(sh_s,               _SH) + "  " +
            _rj(cmp_s,              _CM) + "  " + bar
        )

    print("  " + "─" * (total - 2))
    if result.notes:
        for note in result.notes:
            print(_d(f"  ⚠  {note}"))
    print(_d("  Weights optimised for maximum Sharpe ratio across 5,000 random portfolios"))
    print(_d("  Expected return & volatility are forward estimates based on fundamental analysis"))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 5 — SECTOR PEER RANKING
# ═══════════════════════════════════════════════════════════════════════════════

_SECTOR_MAP: dict[str, str] = {
    "HDFCBANK":"Banking", "ICICIBANK":"Banking", "SBIN":"Banking",
    "AXISBANK":"Banking", "KOTAKBANK":"Banking", "INDUSINDBK":"Banking",
    "TCS":"IT", "INFY":"IT", "HCLTECH":"IT", "WIPRO":"IT", "TECHM":"IT",
    "HINDUNILVR":"FMCG", "ITC":"FMCG", "NESTLEIND":"FMCG",
    "BRITANNIA":"FMCG", "TATACONSUM":"FMCG", "TITAN":"FMCG", "ASIANPAINT":"FMCG",
    "SUNPHARMA":"Pharma", "DRREDDY":"Pharma", "CIPLA":"Pharma",
    "MARUTI":"Auto", "TATAMOTORS":"Auto", "M&M":"Auto",
    "BAJAJ-AUTO":"Auto", "EICHERMOT":"Auto", "HEROMOTOCO":"Auto",
    "RELIANCE":"Energy", "ONGC":"Energy", "BPCL":"Energy",
    "NTPC":"Energy", "POWERGRID":"Energy", "COALINDIA":"Energy",
    "TATASTEEL":"Metals", "JSWSTEEL":"Metals", "HINDALCO":"Metals",
    "LT":"Infra", "ADANIENT":"Infra", "ADANIPORTS":"Infra", "BEL":"Infra",
    "ULTRACEMCO":"Cement", "GRASIM":"Cement",
    "BHARTIARTL":"Telecom",
    "BAJFINANCE":"Fin-Serv", "BAJAJFINSV":"Fin-Serv",
    "SHRIRAMFIN":"Fin-Serv",
    "HDFCLIFE":"Insurance", "SBILIFE":"Insurance",
    "APOLLOHOSP":"Healthcare", "ZOMATO":"Consumer Tech",
}

def print_sector_ranking(score_results: list[ScoreResult]) -> None:
    score_map = {r.symbol: r for r in score_results}

    # Group by sector
    sectors: dict[str, list[ScoreResult]] = {}
    for sym, sec in _SECTOR_MAP.items():
        if sym in score_map:
            sectors.setdefault(sec, []).append(score_map[sym])

    total = 90
    print()
    print(_b(_c("═" * total)))
    print(_b(_c("  SECTOR PEER RANKING  |  Composite Score Within Each Sector")))
    print(_b(_c("═" * total)))

    for sec_name, members in sorted(sectors.items()):
        ranked = sorted(members, key=lambda r: r.composite, reverse=True)
        print()
        print("  " + _b(_ul(f"  {sec_name}  ({len(ranked)} companies)")))
        print("  " + _b(
            "  " +
            _rj("#",       3) + "  " +
            _lj("Symbol",  11)+ "  " +
            _rj("Score",   6) + "  " +
            _lj("Grade",   4) + "  " +
            _rj("Quality", 7) + "  " +
            _rj("Growth",  7) + "  " +
            _rj("Piotroski",9)+ "  " +
            _lj("Verdict", 10)
        ))
        for rank, r in enumerate(ranked, 1):
            sc_s = (_g(_b(f"{r.composite:.1f}")) if r.composite >= 75 else
                    _g(f"{r.composite:.1f}")      if r.composite >= 55 else
                    _y(f"{r.composite:.1f}")      if r.composite >= 35 else
                    _r(f"{r.composite:.1f}"))
            prefix = "  🥇" if rank == 1 else ("  🥈" if rank == 2 else ("  🥉" if rank == 3 else "    "))
            print(
                prefix +
                _rj(str(rank), 3) + "  " +
                _lj(r.symbol,  11)+ "  " +
                _rj(sc_s,       6)+ "  " +
                _lj(r.grade,    4)+ "  " +
                _rj(f"{r.quality:.1f}",  7) + "  " +
                _rj(f"{r.growth:.1f}",   7) + "  " +
                _rj(f"{r.piotroski}/9",  9) + "  " +
                _lj(_verdict_color(r.verdict), 10)
            )

    print()
    print(_d("  Scores are percentile-ranked within the full Nifty 50 universe, not just within the sector"))
    print()
