"""
Professional PDF Equity Research Report Generator.

Generates a Goldman-style 6-page PDF per company with:
  - Cover page with key stats and verdict
  - 5-year P&L snapshot table
  - DCF valuation with Monte Carlo range
  - Multi-factor composite score breakdown
  - AI-written investment thesis (3 paragraphs)
  - Risk factors and key metrics

Requires fpdf2: pip install fpdf2
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fpdf import FPDF, XPos, YPos


# ── Layout constants ──────────────────────────────────────────────────────────
_W, _H   = 210, 297   # A4 mm
_MARGIN  = 18
_BODY_W  = _W - 2 * _MARGIN

# Unicode font paths (tried in order; first found wins)
_UNICODE_FONTS = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]

def _load_font(pdf: "EquityResearchPDF") -> str:
    """Register best available Unicode font and return its name, or 'Helvetica'."""
    for path in _UNICODE_FONTS:
        if os.path.exists(path):
            try:
                pdf.add_font("UniFont",        fname=path)
                pdf.add_font("UniFont", "B",   fname=path)
                pdf.add_font("UniFont", "I",   fname=path)
                pdf.add_font("UniFont", "BI",  fname=path)
                return "UniFont"
            except Exception:
                continue
    return "Helvetica"


def _sanitize(text: str) -> str:
    """Strip characters outside Latin-1 when no Unicode font is available."""
    return text.encode("latin-1", errors="replace").decode("latin-1")

# Colour palette (RGB)
_NAVY    = (15,  40,  80)
_GOLD    = (180, 140,  40)
_GREEN   = (30,  140,  60)
_RED     = (180,  40,  40)
_GREY    = (120, 120, 120)
_LGREY   = (240, 240, 240)
_WHITE   = (255, 255, 255)
_BLACK   = (20,  20,  20)


class EquityResearchPDF(FPDF):

    def header(self):
        pass   # custom header per page

    def footer(self):
        self.set_y(-12)
        self.set_font(getattr(self, "_active_font", "Helvetica"), "I", 7)
        self.set_text_color(*_GREY)
        self.cell(0, 6,
                  f"EquityLens AI Research  |  For informational purposes only  |  Not investment advice  |  Page {self.page_no()}",
                  align="C")
        self.set_text_color(*_BLACK)


def _set(pdf: FPDF, size: int, bold: bool = False, italic: bool = False, color=_BLACK,
         font: str = "Helvetica"):
    style = ("B" if bold else "") + ("I" if italic else "")
    pdf.set_font(font, style, size)
    pdf.set_text_color(*color)


def _hrule(pdf: FPDF, color=_NAVY, thickness: float = 0.5):
    pdf.set_draw_color(*color)
    pdf.set_line_width(thickness)
    pdf.line(_MARGIN, pdf.get_y(), _W - _MARGIN, pdf.get_y())
    pdf.set_line_width(0.2)


def _filled_rect(pdf: FPDF, x, y, w, h, fill, radius: float = 0):
    pdf.set_fill_color(*fill)
    pdf.rect(x, y, w, h, "F")


def _verdict_color(verdict: str):
    v = verdict.upper()
    if "STRONG BUY" in v or "TOP PICK" in v: return _GREEN
    if "BUY"        in v:                     return (50, 160, 80)
    if "HOLD"       in v:                     return (160, 120, 20)
    return _RED


def generate_pdf(
    symbol:       str,
    company_name: str,
    dcf_result,              # DCFResult
    score_result,            # ScoreResult
    ai_thesis:    str = "",
    output_dir:   str = "reports",
    cache_dir:    str = "data/raw/screener",
) -> str:
    """Generate PDF and return the output file path."""

    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(output_dir, f"{symbol}_EquityResearch_{date_str}.pdf")

    # Load raw data
    cache_path = Path(cache_dir) / f"{symbol}.json"
    raw: dict = {}
    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)

    pl_rows  = raw.get("profit_loss",  {}).get("rows", {})
    pl_years = raw.get("profit_loss",  {}).get("years", [])
    bs_rows  = raw.get("balance_sheet",{}).get("rows", {})
    cf_rows  = raw.get("cash_flow",    {}).get("rows", {})
    rat_rows = raw.get("ratios",       {}).get("rows", {})

    # Trim TTM
    n = len(pl_years)
    cut = n - 1 if pl_years and "TTM" in str(pl_years[-1]) else n
    years_5 = pl_years[max(0, cut-5):cut]

    def _row(rows, key, idx_range):
        vals = rows.get(key, [])
        return [vals[i] if i < len(vals) else None for i in idx_range]

    idx5 = list(range(max(0, cut-5), cut))

    sales  = _row(pl_rows, "Sales",        idx5)
    np_    = _row(pl_rows, "Net Profit",   idx5)
    eps    = _row(pl_rows, "EPS in Rs",    idx5)
    opm    = _row(pl_rows, "OPM %",        idx5)
    cfo    = _row(cf_rows, "Cash from Operating Activity", idx5)
    roce   = _row(rat_rows, "ROCE %",      idx5)

    # ── Build PDF ─────────────────────────────────────────────────────────────
    pdf = EquityResearchPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(_MARGIN, _MARGIN, _MARGIN)
    pdf.set_auto_page_break(auto=True, margin=15)

    _FONT = _load_font(pdf)   # "UniFont" if a Unicode TTF was found, else "Helvetica"
    pdf._active_font = _FONT  # expose to footer()

    def _s(size: int, bold: bool = False, italic: bool = False, color=_BLACK):
        _set(pdf, size, bold=bold, italic=italic, color=color, font=_FONT)

    def _t(text: str) -> str:
        """Return text safe for the active font (sanitise to Latin-1 if needed)."""
        return text if _FONT == "UniFont" else _sanitize(text)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # Top navy banner
    _filled_rect(pdf, 0, 0, _W, 55, _NAVY)
    pdf.set_y(10)
    _s(9, bold=True, color=_WHITE)
    pdf.cell(_W, 6, "EQUITYLENS  ·  AI EQUITY RESEARCH", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _s(6, italic=True, color=(180, 200, 230))
    pdf.cell(_W, 5, f"Generated: {datetime.now().strftime('%d %B %Y')}  |  Data: Screener.in / NSE", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(20)
    _s(28, bold=True, color=_WHITE)
    pdf.cell(_W, 12, company_name.upper(), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _s(13, color=(200, 220, 255))
    pdf.cell(_W, 7, f"NSE: {symbol}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Verdict badge
    verdict      = dcf_result.verdict if dcf_result else score_result.verdict
    vcolor       = _verdict_color(verdict)
    badge_w, badge_h = 70, 12
    bx = (_W - badge_w) / 2
    pdf.set_y(42)
    _filled_rect(pdf, bx, pdf.get_y(), badge_w, badge_h, vcolor)
    _s(11, bold=True, color=_WHITE)
    pdf.set_xy(bx, pdf.get_y() + 2.5)
    pdf.cell(badge_w, 7, verdict, align="C")

    # ── Key Metrics Grid ──────────────────────────────────────────────────────
    pdf.set_y(62)
    metrics = []

    if dcf_result and dcf_result.intrinsic_value > 0:
        metrics.append(("DCF Intrinsic Value", f"Rs.{dcf_result.intrinsic_value:,.0f}"))
    if dcf_result and dcf_result.cmp:
        metrics.append(("Current Market Price", f"Rs.{dcf_result.cmp:,.0f}"))
    if dcf_result and dcf_result.margin_of_safety is not None:
        mos_pct = dcf_result.margin_of_safety * 100
        metrics.append(("Margin of Safety", f"{mos_pct:+.1f}%"))

    if score_result:
        metrics.append(("Composite Score", f"{score_result.composite:.1f} / 100  ({score_result.grade})"))
        metrics.append(("Piotroski F-Score", f"{score_result.piotroski} / 9"))
        metrics.append(("ROCE (3Y avg)", f"{score_result.roce_pct:.0f}th percentile"))
        metrics.append(("Revenue CAGR 3Y",   f"{score_result.revenue_cagr_3y:+.1f}%"))
        metrics.append(("EPS CAGR 3Y",       f"{score_result.eps_cagr_3y:+.1f}%"))

    if dcf_result:
        metrics.append(("WACC", f"{dcf_result.wacc*100:.1f}%"))
        metrics.append(("FCF CAGR (hist.)", f"{dcf_result.fcf_cagr*100:+.1f}%"))

    cols = 2
    col_w = _BODY_W / cols
    row_h = 11
    _s(8, bold=True, color=_NAVY)
    pdf.cell(_BODY_W, 5, "KEY METRICS AT A GLANCE", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _hrule(pdf)
    pdf.ln(2)

    for i, (label, value) in enumerate(metrics):
        col = i % cols
        x_pos = _MARGIN + col * col_w
        if col == 0 and i > 0:
            pass   # new row (natural flow)
        y_pos = pdf.get_y() if col == 0 else pdf.get_y()
        if col == 0:
            _filled_rect(pdf, _MARGIN, pdf.get_y(), _BODY_W, row_h,
                          _LGREY if (i // cols) % 2 == 0 else _WHITE)

        pdf.set_x(x_pos + 2)
        _s(8, color=_GREY)
        pdf.cell(col_w * 0.45, row_h, label, align="L")
        _s(8, bold=True, color=_BLACK)
        pdf.cell(col_w * 0.50, row_h, value, align="R",
                 new_x=XPos.LMARGIN if col == cols - 1 else XPos.RIGHT,
                 new_y=YPos.NEXT    if col == cols - 1 else YPos.TMARGIN)

    # ── Disclaimer box ────────────────────────────────────────────────────────
    pdf.ln(4)
    _filled_rect(pdf, _MARGIN, pdf.get_y(), _BODY_W, 10, (255, 248, 220))
    pdf.set_x(_MARGIN + 2)
    _s(7, italic=True, color=(120, 80, 20))
    pdf.cell(_BODY_W - 4, 10,
             "This report is for educational/informational purposes only and does not constitute investment advice. "
             "Past performance is not indicative of future results.", align="L")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — FINANCIAL SUMMARY (5-Year P&L Table)
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    _s(13, bold=True, color=_NAVY)
    pdf.cell(_BODY_W, 8, "FINANCIAL SUMMARY — 5-YEAR TREND", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _hrule(pdf)
    pdf.ln(3)

    # Build 5-year table
    label_col = 42
    yr_cols   = len(years_5)
    yr_w      = (_BODY_W - label_col) / max(yr_cols, 1)

    # Header row
    _filled_rect(pdf, _MARGIN, pdf.get_y(), _BODY_W, 7, _NAVY)
    pdf.set_x(_MARGIN)
    _s(7, bold=True, color=_WHITE)
    pdf.cell(label_col, 7, "  Metric (Rs. Crore)", align="L")
    for yr in years_5:
        pdf.cell(yr_w, 7, str(yr)[-7:], align="C")
    pdf.ln(7)

    def _table_row(label: str, values: list, fmt=None, row_idx=0, is_pct=False):
        bg = _LGREY if row_idx % 2 == 0 else _WHITE
        _filled_rect(pdf, _MARGIN, pdf.get_y(), _BODY_W, 6.5, bg)
        pdf.set_x(_MARGIN + 2)
        _s(7, color=_GREY)
        pdf.cell(label_col - 2, 6.5, label, align="L")
        for v in values:
            if v is None:
                txt = "—"
                _s(7, color=_GREY)
            else:
                txt = f"{v:.1f}{'%' if is_pct else ''}" if is_pct else f"{v:,.0f}"
                _s(7, bold=True, color=_GREEN if not is_pct or v > 0 else _RED)
            pdf.cell(yr_w, 6.5, txt, align="C")
        pdf.ln(6.5)

    table_rows = [
        ("Revenue (Sales)",         sales,  False),
        ("Net Profit",              np_,    False),
        ("EPS (Rs.)",                 eps,    False),
        ("Op. Profit Margin %",     opm,    True),
        ("ROCE %",                  roce,   True),
        ("Cash from Operations",    cfo,    False),
    ]
    for i, (lbl, vals, is_p) in enumerate(table_rows):
        _table_row(lbl, vals, row_idx=i, is_pct=is_p)

    # ── CAGR summary ──────────────────────────────────────────────────────────
    pdf.ln(4)
    _s(10, bold=True, color=_NAVY)
    pdf.cell(_BODY_W, 6, "GROWTH RATES (CAGR)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _hrule(pdf)
    pdf.ln(2)

    if score_result:
        cagr_data = [
            ("Revenue CAGR (3Y)",    f"{score_result.revenue_cagr_3y:+.1f}%"),
            ("EPS CAGR (3Y)",        f"{score_result.eps_cagr_3y:+.1f}%"),
            ("FCF CAGR (3Y)",        f"{score_result.fcf_cagr_3y:+.1f}%"),
            ("D/E Ratio",            f"{score_result.de_ratio:.2f}x"),
            ("Interest Coverage",    f"{score_result.interest_coverage:.1f}x"),
        ]
        col_w2 = _BODY_W / 2
        for i, (lbl, val) in enumerate(cagr_data):
            col = i % 2
            x_pos = _MARGIN + col * col_w2
            if col == 0:
                bg = _LGREY if (i // 2) % 2 == 0 else _WHITE
                _filled_rect(pdf, _MARGIN, pdf.get_y(), _BODY_W, 7, bg)
            pdf.set_x(x_pos + 2)
            _s(8, color=_GREY)
            pdf.cell(col_w2 * 0.55, 7, lbl, align="L")
            _s(8, bold=True, color=_BLACK)
            pdf.cell(col_w2 * 0.40, 7, val, align="R",
                     new_x=XPos.LMARGIN if col == 1 else XPos.RIGHT,
                     new_y=YPos.NEXT    if col == 1 else YPos.TMARGIN)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — DCF VALUATION
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    _s(13, bold=True, color=_NAVY)
    pdf.cell(_BODY_W, 8, "DCF VALUATION  ·  Monte Carlo Analysis", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _hrule(pdf)
    pdf.ln(4)

    if dcf_result and dcf_result.intrinsic_value > 0:
        # DCF parameter table
        dcf_params = [
            ("Model",               "10-Year FCF Projection + Gordon Growth Terminal Value"),
            ("Base FCF",            f"Rs.{dcf_result.pv_fcf + dcf_result.terminal_value:.0f} Cr enterprise value"),
            ("WACC",                f"{dcf_result.wacc*100:.2f}%  (auto-estimated from balance sheet)"),
            ("Terminal Growth Rate",f"5.5%  (India long-run nominal GDP)"),
            ("FCF CAGR (hist.)",    f"{dcf_result.fcf_cagr*100:+.1f}%  (used as base-case projection)"),
            ("Simulations",         "10,000 Monte Carlo iterations  (WACC ±1.5%  |  CAGR ±4%)"),
            ("Shares Outstanding",  f"{dcf_result.shares_cr:.1f} Crore"),
        ]
        for i, (lbl, val) in enumerate(dcf_params):
            bg = _LGREY if i % 2 == 0 else _WHITE
            _filled_rect(pdf, _MARGIN, pdf.get_y(), _BODY_W, 7, bg)
            pdf.set_x(_MARGIN + 2)
            _s(8, color=_GREY)
            pdf.cell(60, 7, lbl, align="L")
            _s(8, bold=False, color=_BLACK)
            pdf.cell(_BODY_W - 62, 7, val, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(6)

        # Valuation summary box
        _filled_rect(pdf, _MARGIN, pdf.get_y(), _BODY_W, 40, _NAVY)
        pdf.set_y(pdf.get_y() + 5)
        pdf.set_x(_MARGIN)
        _s(9, bold=True, color=_WHITE)
        pdf.cell(_BODY_W, 6, "VALUATION SUMMARY", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        thirds = _BODY_W / 3
        labels = ["Base-Case IV (Rs.)", "P10 Bear Case (Rs.)", "P90 Bull Case (Rs.)"]
        values = [
            f"Rs.{dcf_result.intrinsic_value:,.0f}",
            f"Rs.{dcf_result.iv_low:,.0f}",
            f"Rs.{dcf_result.iv_high:,.0f}",
        ]
        pdf.set_x(_MARGIN)
        for i, (lbl, val) in enumerate(zip(labels, values)):
            pdf.set_x(_MARGIN + i * thirds)
            _s(7, color=(180, 200, 230))
            pdf.cell(thirds, 5, lbl, align="C", new_x=XPos.RIGHT, new_y=YPos.TMARGIN)
        pdf.ln(5)
        pdf.set_x(_MARGIN)
        for i, (lbl, val) in enumerate(zip(labels, values)):
            pdf.set_x(_MARGIN + i * thirds)
            _s(14, bold=True, color=_WHITE)
            pdf.cell(thirds, 10, val, align="C", new_x=XPos.RIGHT, new_y=YPos.TMARGIN)
        pdf.ln(10)

        if dcf_result.cmp and dcf_result.margin_of_safety is not None:
            mos   = dcf_result.margin_of_safety * 100
            mos_c = _GREEN if mos >= 0 else _RED
            pdf.set_x(_MARGIN)
            _s(7, color=(180, 200, 230))
            pdf.cell(_BODY_W / 2, 5, f"Current Price: Rs.{dcf_result.cmp:,.0f}", align="C")
            _s(7, bold=True, color=_WHITE)
            pdf.cell(_BODY_W / 2, 5, f"Margin of Safety: {mos:+.1f}%", align="C",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(8)
        if dcf_result.notes:
            _s(7, italic=True, color=_GREY)
            for note in dcf_result.notes:
                pdf.cell(_BODY_W, 5, f"⚠  {note}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        _s(9, color=_GREY)
        pdf.cell(_BODY_W, 8, "DCF data insufficient — FCF history too erratic or negative.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — COMPOSITE SCORE BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    _s(13, bold=True, color=_NAVY)
    pdf.cell(_BODY_W, 8, "MULTI-FACTOR COMPOSITE SCORE  ·  Quality · Growth · Earnings Quality · Piotroski",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _hrule(pdf)
    pdf.ln(4)

    if score_result:
        # Big score circle (simulated as a box)
        score = score_result.composite
        sc    = _verdict_color(score_result.verdict)
        _filled_rect(pdf, _MARGIN, pdf.get_y(), _BODY_W, 22, sc)
        pdf.set_y(pdf.get_y() + 3)
        _s(22, bold=True, color=_WHITE)
        pdf.cell(_BODY_W, 10, f"{score:.1f} / 100  ({score_result.grade})", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        _s(9, color=_WHITE)
        pdf.cell(_BODY_W, 6, score_result.verdict, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(6)

        # Pillar breakdown
        pillars = [
            ("Quality Score",         f"{score_result.quality:.1f}",          "ROCE, Operating Margin, D/E, Interest Coverage"),
            ("Growth Score",          f"{score_result.growth:.1f}",           "Revenue CAGR, EPS CAGR, FCF CAGR"),
            ("Earnings Quality",      f"{score_result.earnings_quality:.1f}", "CFO/Net Profit ratio, Accruals ratio"),
            ("Piotroski F-Score",     f"{score_result.piotroski} / 9",        "9-point financial health test (Profitability · Leverage · Efficiency)"),
        ]
        for i, (pillar, val, desc) in enumerate(pillars):
            bg = _LGREY if i % 2 == 0 else _WHITE
            _filled_rect(pdf, _MARGIN, pdf.get_y(), _BODY_W, 13, bg)
            pdf.set_x(_MARGIN + 2)
            _s(9, bold=True, color=_NAVY)
            pdf.cell(50, 7, pillar, align="L")
            _s(12, bold=True, color=_GREEN if float(val.split()[0]) >= 50 else _RED)
            pdf.cell(20, 7, val, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_x(_MARGIN + 4)
            _s(7, italic=True, color=_GREY)
            pdf.cell(_BODY_W - 4, 6, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Raw metrics
        pdf.ln(4)
        _s(10, bold=True, color=_NAVY)
        pdf.cell(_BODY_W, 6, "RAW METRICS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        _hrule(pdf)
        pdf.ln(2)

        raw_metrics = [
            ("ROCE %",             f"{score_result.roce_pct:.0f}th percentile"),
            ("OPM %",              f"{score_result.opm_pct:.0f}th percentile"),
            ("Debt-to-Equity",     f"{score_result.de_ratio:.2f}x"),
            ("Interest Coverage",  f"{score_result.interest_coverage:.1f}x"),
            ("Revenue CAGR 3Y",    f"{score_result.revenue_cagr_3y:+.1f}%"),
            ("EPS CAGR 3Y",        f"{score_result.eps_cagr_3y:+.1f}%"),
            ("FCF CAGR 3Y",        f"{score_result.fcf_cagr_3y:+.1f}%"),
        ]
        col_w3 = _BODY_W / 2
        for i, (lbl, val) in enumerate(raw_metrics):
            col = i % 2
            if col == 0:
                bg = _LGREY if (i // 2) % 2 == 0 else _WHITE
                _filled_rect(pdf, _MARGIN, pdf.get_y(), _BODY_W, 7, bg)
            pdf.set_x(_MARGIN + col * col_w3 + 2)
            _s(8, color=_GREY)
            pdf.cell(col_w3 * 0.55, 7, lbl, align="L")
            _s(8, bold=True, color=_BLACK)
            pdf.cell(col_w3 * 0.40, 7, val, align="R",
                     new_x=XPos.LMARGIN if col == 1 else XPos.RIGHT,
                     new_y=YPos.NEXT    if col == 1 else YPos.TMARGIN)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5 — AI INVESTMENT THESIS
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    _s(13, bold=True, color=_NAVY)
    pdf.cell(_BODY_W, 8, "INVESTMENT THESIS  ·  AI-Generated Analysis", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _hrule(pdf)
    pdf.ln(4)

    if ai_thesis:
        _s(9, color=_BLACK)
        pdf.set_x(_MARGIN)
        pdf.multi_cell(_BODY_W, 6, ai_thesis)
    else:
        _s(9, italic=True, color=_GREY)
        summary_lines = [
            f"{company_name} ({symbol}) is a constituent of the Nifty 50 index.",
        ]
        if score_result:
            summary_lines.append(
                f"The company scores {score_result.composite:.1f}/100 on our multi-factor composite model, "
                f"ranking it in the {score_result.verdict} category with a {score_result.grade} grade."
            )
            summary_lines.append(
                f"Key strengths: ROCE at {score_result.roce_pct:.0f}th percentile within Nifty 50, "
                f"EPS growing at {score_result.eps_cagr_3y:+.1f}% per year (3Y CAGR), "
                f"Revenue expanding at {score_result.revenue_cagr_3y:+.1f}% annually. "
                f"D/E ratio stands at {score_result.de_ratio:.2f}x with interest coverage of {score_result.interest_coverage:.1f}x."
            )
        if dcf_result and dcf_result.intrinsic_value > 0:
            mos_txt = ""
            if dcf_result.margin_of_safety is not None:
                mos_txt = (f"The DCF model implies a margin of safety of {dcf_result.margin_of_safety*100:+.1f}% "
                           f"versus the current market price of Rs.{dcf_result.cmp:,.0f}. ")
            summary_lines.append(
                f"Our DCF valuation, using a {dcf_result.wacc*100:.1f}% WACC and {dcf_result.fcf_cagr*100:+.1f}% "
                f"historical FCF CAGR, yields a base-case intrinsic value of Rs.{dcf_result.intrinsic_value:,.0f} per share "
                f"(P10–P90 range: Rs.{dcf_result.iv_low:,.0f}–Rs.{dcf_result.iv_high:,.0f}). {mos_txt}"
                f"The verdict is: {dcf_result.verdict}."
            )
        for line in summary_lines:
            _s(9, color=_BLACK)
            pdf.multi_cell(_BODY_W, 6, line)
            pdf.ln(3)

    # Risk factors
    pdf.ln(5)
    _s(10, bold=True, color=_NAVY)
    pdf.cell(_BODY_W, 6, "STANDARD RISK FACTORS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _hrule(pdf)
    pdf.ln(2)
    risks = [
        "Market Risk: Broad market downturns may affect price irrespective of fundamentals.",
        "Earnings Risk: Future earnings may deviate from historical trends used in this model.",
        "Valuation Risk: DCF models are sensitive to WACC and terminal growth rate assumptions.",
        "Sector Risk: Regulatory changes, commodity cycles, or competitive pressures specific to the industry.",
        "Liquidity Risk: Lower-volume stocks may be harder to exit at favourable prices.",
    ]
    _s(8, color=_BLACK)
    for risk in risks:
        pdf.set_x(_MARGIN + 3)
        pdf.cell(3, 6, "•", align="L")
        pdf.set_x(_MARGIN + 7)
        pdf.multi_cell(_BODY_W - 7, 6, risk)
        pdf.ln(1)

    pdf.output(out_path)
    return out_path
