"""
Research pack: a PDF for any set of Nifty 50 companies, built from a free-text
request ("last 5 years of P&L for the top 5 IT companies, their future goals,
and whether they are investable").

    spec = parse_request(text, llm)          # src/reports/request.py
    path = build_research_pack(spec)

Every number is computed in Python from the Screener cache, the live price and
the valuation/score engines. The only model-written text is the management
outlook, and each of its points carries a quote verified against the earnings-
call transcript (src/facts/outlook.py). Disclosures and the model's backtested
track record are generated in code, so they cannot be left out.

Sections (any subset, chosen by the request):
  comparison     one row per company: size, growth, margins, quality, model view
  pnl / balance_sheet / cash_flow / ratios
                 the statement exactly as reported, last N full years, every row
  outlook        management's stated goals, quoted and sourced
  investability  valuation range vs price, P(undervalued), reverse DCF, quality
                 score — preceded by the model's track record
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fpdf import FPDF, XPos, YPos

from config.nifty50_tickers import NIFTY50_TICKERS
from src.data.company import CORPORATE_ACTIONS, CompanyData, load_company
from src.data.sectors import sector_label
from src.reports.disclosure import AI_CONTENT_NOTICE, SEBI_DISCLAIMER, data_provenance
from src.reports.request import SECTIONS, ReportSpec
from src.utils.logger import get_logger

logger = get_logger(__name__)

_W, _MARGIN = 210, 16
_BODY = _W - 2 * _MARGIN

_NAVY, _GOLD, _GREY = (15, 40, 80), (180, 140, 40), (110, 110, 110)
_LIGHT, _ZEBRA, _BLACK = (232, 237, 245), (247, 248, 250), (25, 25, 25)
_GREEN, _AMBER, _RED = (30, 130, 60), (170, 120, 20), (170, 40, 40)

_FONTS = {          # style → candidates; the first that exists is used
    "":  ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
          "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "B": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}

_STATEMENT_TITLES = {"pnl": ("profit_loss", "Profit & loss"),
                     "balance_sheet": ("balance_sheet", "Balance sheet"),
                     "cash_flow": ("cash_flow", "Cash flow"),
                     "ratios": ("ratios", "Key ratios")}

_CATEGORY_LABELS = {"growth": "Growth", "margins": "Margins", "deals_demand": "Deals & demand",
                    "strategy": "Strategy", "capital_allocation": "Capital allocation",
                    "hiring": "Hiring", "risks": "Risks flagged"}


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class CompanyPack:
    symbol:   str
    name:     str
    sector:   str
    data:     CompanyData
    price:    Optional[float] = None
    mcap_cr:  Optional[float] = None
    dcf:      Any = None              # DCFResult
    score:    Any = None              # ScoreResult
    rank:     Optional[int] = None    # composite-score rank in the Nifty 50
    outlook:  Any = None              # Outlook
    notes:    list[str] = field(default_factory=list)


@dataclass
class PackData:
    spec:         ReportSpec
    companies:    list[CompanyPack]
    track_record: Optional[dict] = None
    rf_label:     str = ""
    rf_is_live:   bool = True
    fundamentals_as_of: str = ""
    fundamentals_age:   Optional[float] = None


def gather(spec: ReportSpec, outlook_extractor=None, dcf_engine=None, scorer=None,
           price_fn=None) -> PackData:
    """Collect everything the requested sections need. Engines are injectable for tests."""
    from src.backtest.track_record import load_track_record
    from src.utils.cache_manager import freshness_report

    sections = set(spec.sections)
    need_value = bool(sections & {"investability", "comparison"})

    if price_fn is None:
        from src.valuation.dcf_engine import _fetch_cmp as price_fn
    if need_value and dcf_engine is None:
        from src.valuation.dcf_engine import DCFEngine
        dcf_engine = DCFEngine(quiet=True)
    scores, ranks = {}, {}
    if need_value:
        if scorer is None:
            from src.scoring.composite_score import CompositeScorer
            scorer = CompositeScorer()
        ranked = sorted((r for r in scorer.score_all(list(NIFTY50_TICKERS)) if r.grade != "N/A"),
                        key=lambda r: -r.composite)
        scores = {r.symbol: r for r in ranked}
        ranks  = {r.symbol: i + 1 for i, r in enumerate(ranked)}
    if "outlook" in sections and outlook_extractor is None:
        from src.facts.outlook import OutlookExtractor
        outlook_extractor = OutlookExtractor()

    packs = []
    for sym in spec.symbols:
        cd = load_company(sym)
        if cd is None:
            continue
        cp = CompanyPack(sym, NIFTY50_TICKERS.get(sym, sym), sector_label(sym), cd)
        if sym in CORPORATE_ACTIONS:
            cp.notes.append(CORPORATE_ACTIONS[sym])
        cp.price = None if sym == "TATAMOTORS" else price_fn(sym)
        np_, eps = cd.series("profit_loss", "Net Profit"), cd.series("profit_loss", "EPS in Rs")
        if cp.price and np_ and eps and eps[-1]:
            cp.mcap_cr = cp.price * np_[-1] / eps[-1]
        if need_value:
            cp.dcf = dcf_engine.value(sym)
            cp.score, cp.rank = scores.get(sym), ranks.get(sym)
        if "outlook" in sections:
            cp.outlook = outlook_extractor.outlook(sym)
        packs.append(cp)

    fresh = freshness_report(spec.symbols)
    oldest = max((v for v in fresh.values() if v.get("age_days") is not None),
                 key=lambda v: v["age_days"], default={})
    rf = getattr(dcf_engine, "rf", None)
    return PackData(spec=spec, companies=packs,
                    track_record=load_track_record() if "investability" in sections else None,
                    rf_label=rf.label if rf else "", rf_is_live=rf.is_live if rf else True,
                    fundamentals_as_of=oldest.get("last_modified", ""),
                    fundamentals_age=oldest.get("age_days"))


# ── Formatting ────────────────────────────────────────────────────────────────

def _num(v: Any, decimals: int = 0) -> str:
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _pct(v: Any, decimals: int = 1, signed: bool = False) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):{'+' if signed else ''}.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def _rs(v: Any, decimals: int = 0) -> str:
    return "—" if v is None else f"Rs {_num(v, decimals)}"


def _cagr(series: list, years: int) -> Optional[float]:
    vals = [v for v in series[-(years + 1):] if v is not None]
    if len(vals) < 2 or vals[0] <= 0 or vals[-1] <= 0:
        return None
    return ((vals[-1] / vals[0]) ** (1 / (len(vals) - 1)) - 1) * 100


def _verdict_colour(verdict: str):
    v = (verdict or "").upper()
    if "UNDERVALUED" in v:
        return _GREEN
    if "FAIRLY" in v:
        return _AMBER
    if "OVERVALUED" in v:
        return _RED
    return _GREY


# ── PDF ───────────────────────────────────────────────────────────────────────

class _PackPDF(FPDF):

    def __init__(self, title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.pack_title = title
        self.set_margins(_MARGIN, _MARGIN, _MARGIN)
        self.set_auto_page_break(auto=True, margin=16)
        self.font_name = "Helvetica"
        self.unicode = False
        paths = {style: next((p for p in cands if os.path.exists(p)), None)
                 for style, cands in _FONTS.items()}
        if paths[""]:
            self.add_font("Pack", "", paths[""])
            self.add_font("Pack", "B", paths["B"] or paths[""])
            self.font_name, self.unicode = "Pack", True
        self.c_margin = 1.6            # cell padding, so adjacent header labels don't touch

    def multi_cell(self, *args, **kwargs):
        kwargs.setdefault("align", "L")   # fpdf justifies by default, which leaves gaps
        return super().multi_cell(*args, **kwargs)

    def t(self, text: str) -> str:
        """Text safe for the active font. The rupee sign is written as "Rs"."""
        text = str(text).replace("₹", "Rs ")
        return text if self.unicode else text.encode("latin-1", "replace").decode("latin-1")

    def style(self, size: float, bold: bool = False, colour=_BLACK):
        self.set_font(self.font_name, "B" if bold else "", size)
        self.set_text_color(*colour)

    def header(self):
        if self.page_no() == 1:
            return
        self.style(7.5, colour=_GREY)
        self.cell(0, 5, self.t(self.pack_title), align="L")
        self.cell(0, 5, self.t(datetime.now().strftime("%d %b %Y")), align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*_LIGHT)
        self.line(_MARGIN, self.get_y(), _W - _MARGIN, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-11)
        self.style(6.8, colour=_GREY)
        self.cell(0, 5, self.t(f"EquityLens research pack · Educational use only — not investment "
                               f"advice; not SEBI-registered · Page {self.page_no()}"), align="C")

    # building blocks

    def h1(self, text: str):
        self.style(15, bold=True, colour=_NAVY)
        self.multi_cell(0, 7.5, self.t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*_GOLD)
        self.set_line_width(0.6)
        self.line(_MARGIN, self.get_y() + 1, _MARGIN + 30, self.get_y() + 1)
        self.set_line_width(0.2)
        self.ln(4)

    def h2(self, text: str, sub: str = ""):
        if self.get_y() > 250:
            self.add_page()
        self.ln(1)
        self.style(11.5, bold=True, colour=_NAVY)
        self.cell(0, 6.5, self.t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if sub:
            self.style(7.8, colour=_GREY)
            self.multi_cell(0, 4, self.t(sub), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)

    def para(self, text: str, size: float = 8.8, colour=_BLACK, bold: bool = False, h: float = 4.4):
        self.style(size, bold=bold, colour=colour)
        self.multi_cell(0, h, self.t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float],
              aligns: Optional[list[str]] = None, size: float = 7.6,
              bold_rows: set[int] = frozenset(), colours: Optional[dict] = None):
        aligns = aligns or ["L"] + ["R"] * (len(headers) - 1)
        line_h = size * 0.62
        if self.get_y() + line_h * 2.6 > 280:
            self.add_page()
        self.set_fill_color(*_NAVY)
        self.style(size, bold=True, colour=(255, 255, 255))
        for h, w, a in zip(headers, widths, aligns):
            self.cell(w, line_h + 1.6, self.t(h), align=a, fill=True)
        self.ln()
        for i, row in enumerate(rows):
            if self.get_y() + line_h + 1.2 > 281:
                self.add_page()
            self.set_fill_color(*(_ZEBRA if i % 2 else (255, 255, 255)))
            for j, (v, w, a) in enumerate(zip(row, widths, aligns)):
                colour = (colours or {}).get((i, j), _BLACK)
                self.style(size, bold=(i in bold_rows) or j == 0, colour=colour)
                self.cell(w, line_h + 1.2, self.t(v), align=a, fill=True)
            self.ln()
        self.ln(2)


def build_research_pack(spec: ReportSpec, output_dir: str = "reports",
                        data: Optional[PackData] = None) -> str:
    """Render the pack for `spec` and return the PDF path."""
    data = data or gather(spec)
    if not data.companies:
        raise ValueError("None of the requested companies has data in the cache.")
    names = ", ".join(c.name for c in data.companies)
    title = "Research pack — " + (names if len(names) < 70 else f"{len(data.companies)} companies")

    pdf = _PackPDF(title)
    _cover(pdf, data)
    for section in spec.sections:
        if section == "comparison" and len(data.companies) > 1:
            _comparison(pdf, data)
        elif section in _STATEMENT_TITLES:
            _statements(pdf, data, section)
        elif section == "outlook":
            _outlook(pdf, data)
        elif section == "investability":
            _investability(pdf, data)
    _disclosures(pdf, data)

    os.makedirs(output_dir, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", "_".join(c.symbol for c in data.companies))[:60]
    path = os.path.join(output_dir, f"ResearchPack_{slug}_{datetime.now():%Y%m%d_%H%M}.pdf")
    pdf.output(path)
    return path


# ── Sections ──────────────────────────────────────────────────────────────────

def _cover(pdf: _PackPDF, data: PackData):
    pdf.add_page()
    pdf.set_fill_color(*_NAVY)
    pdf.rect(0, 0, _W, 46, "F")
    pdf.set_xy(_MARGIN, 13)
    pdf.style(21, bold=True, colour=(255, 255, 255))
    pdf.cell(0, 10, pdf.t("Research pack"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.style(10, colour=(215, 222, 235))
    pdf.multi_cell(0, 5, pdf.t(", ".join(c.name for c in data.companies)),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(54)

    pdf.para("Your request", size=8, colour=_GREY, bold=True)
    pdf.para(f"“{data.spec.request.strip()}”", size=9.5)
    pdf.ln(3)
    rows = [("Companies", data.spec.selection or "As requested"),
            ("History", f"Last {data.spec.years} full fiscal years"),
            ("Contents", "; ".join(SECTIONS[s] for s in data.spec.sections
                                   if s != "comparison" or len(data.companies) > 1)),
            ("Prepared", datetime.now().strftime("%d %b %Y, %H:%M"))]
    for k, v in rows:
        pdf.style(8.6, bold=True, colour=_NAVY)
        pdf.cell(28, 5, pdf.t(k))
        pdf.style(8.6)
        pdf.multi_cell(0, 5, pdf.t(v), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    pdf.table(["Company", "NSE", "Sector", "Price", "Market cap (Rs Cr)"],
              [[c.name, c.symbol, c.sector, _rs(c.price), _num(c.mcap_cr)] for c in data.companies],
              [62, 26, 30, 26, 34], aligns=["L", "L", "L", "R", "R"], size=8)
    for c in data.companies:
        for n in c.notes:
            pdf.para(f"{c.symbol}: {n}", size=7.4, colour=_RED)
    pdf.para("Figures are in Rs crore unless stated. Fiscal years end in March for most "
             "companies; columns are full years as reported (consolidated where available).",
             size=7.4, colour=_GREY)


def _comparison(pdf: _PackPDF, data: PackData):
    pdf.add_page()
    pdf.h1("Side-by-side comparison")
    y = data.spec.years
    rows, colours = [], {}
    for i, c in enumerate(data.companies):
        cd = c.data
        sales = cd.aligned("profit_loss", "Sales", "Revenue")
        np_   = cd.aligned("profit_loss", "Net Profit")
        roc   = cd.aligned("ratios", "ROE %" if c.data.is_lender else "ROCE %")
        margin = (np_[-1] / sales[-1] * 100) if sales and np_ and sales[-1] and np_[-1] is not None else None
        verdict = getattr(c.dcf, "verdict", "") if c.dcf and c.dcf.intrinsic_value > 0 else "—"
        rows.append([c.name, _num(c.mcap_cr), _num(sales[-1] if sales else None),
                     _pct(_cagr(sales, y - 1)), _num(np_[-1] if np_ else None), _pct(margin),
                     _pct(roc[-1] if roc else None),
                     f"{c.score.composite:.0f} ({c.score.grade})" if c.score else "—",
                     verdict.title() if verdict != "—" else "—"])
        colours[(i, 8)] = _verdict_colour(verdict)
    pdf.table(["Company", "Mkt cap", "Revenue", f"Rev CAGR {y - 1}y", "Net profit",
               "Net margin", "ROCE/ROE", "Quality", "Model view"],
              rows, [33, 17, 17, 20, 18, 17, 17, 14, 25], size=7.1, colours=colours)
    pdf.para("Revenue and net profit: latest full year, Rs crore. ROE is shown for lenders, ROCE "
             "for others. Quality is the composite score (0–100) ranked within the Nifty 50. "
             "Model view is the valuation model's probability band — see \"Is it investable?\" "
             "for what it means and how reliable it has been.", size=7.2, colour=_GREY)
    _comparison_chart(pdf, data)


def _comparison_chart(pdf: _PackPDF, data: PackData):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for c in data.companies:
        yrs   = c.data.years("profit_loss")[-data.spec.years:]
        sales = c.data.aligned("profit_loss", "Sales", "Revenue")[-data.spec.years:]
        np_   = c.data.aligned("profit_loss", "Net Profit")[-data.spec.years:]
        x = [y[-4:] for y in yrs]
        axes[0].plot(x, sales, marker="o", linewidth=1.6, label=c.symbol)
        axes[1].plot(x, np_, marker="o", linewidth=1.6, label=c.symbol)
    for ax, title in zip(axes, ("Revenue (Rs Cr)", "Net profit (Rs Cr)")):
        ax.set_title(title, fontsize=9, color="#0F2850")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), fontsize=7.5,
               frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _place_figure(pdf, fig, height=62)


def _place_figure(pdf: _PackPDF, fig, height: float):
    import matplotlib.pyplot as plt
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if pdf.get_y() + height > 280:
        pdf.add_page()
    pdf.image(tmp.name, x=_MARGIN, w=_BODY, h=height)
    pdf.ln(2)
    os.unlink(tmp.name)


def _statements(pdf: _PackPDF, data: PackData, section: str):
    key, title = _STATEMENT_TITLES[section]
    pdf.add_page()
    pdf.h1(title)
    for c in data.companies:
        years = c.data.years(key)[-data.spec.years:]
        rows_src = (c.data.raw.get(key) or {}).get("rows") or {}
        if not years or not rows_src:
            pdf.h2(f"{c.name} ({c.symbol})")
            pdf.para("Not available for this company.", colour=_GREY)
            continue
        rows, bold = [], set()
        for label in rows_src:
            vals = c.data.aligned(key, label)[-len(years):]
            if not any(v is not None for v in vals):
                continue
            dec = 2 if "EPS" in label else 0
            fmt = (lambda v: _pct(v, 0 if abs(v or 0) >= 100 else 1)) if label.endswith("%") \
                else (lambda v, d=dec: _num(v, d))
            if label in ("Net Profit", "Operating Profit", "Total Assets", "Total Liabilities",
                         "Net Cash Flow", "Profit before tax"):
                bold.add(len(rows))
            rows.append([label] + [fmt(v) for v in vals])
        label_w = 46
        col_w = (_BODY - label_w) / len(years)
        pdf.h2(f"{c.name} ({c.symbol})",
               f"Rs crore unless stated · {'consolidated' if c.data.raw.get('consolidated') else 'standalone'} "
               f"· source: Screener.in")
        pdf.table(["Fiscal year"] + [y for y in years], rows, [label_w] + [col_w] * len(years),
                  size=7.2 if len(years) <= 7 else 6.4, bold_rows=bold)
        if section == "pnl":
            _pnl_chart(pdf, c, years)


def _pnl_chart(pdf: _PackPDF, c: CompanyPack, years: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    sales = [v or 0 for v in c.data.aligned("profit_loss", "Sales", "Revenue")[-len(years):]]
    np_   = [v or 0 for v in c.data.aligned("profit_loss", "Net Profit")[-len(years):]]
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.bar(x - 0.2, sales, 0.4, color="#0F2850", label="Revenue")
    ax.bar(x + 0.2, np_, 0.4, color="#1E8C3C", label="Net profit")
    ax.set_xticks(x, [y for y in years], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    _place_figure(pdf, fig, height=42)


def _outlook(pdf: _PackPDF, data: PackData):
    pdf.add_page()
    pdf.h1("Management's stated goals and outlook")
    pdf.para("What each company's management said on its latest earnings calls. Every point is "
             "summarised from a verbatim quote, and the quote has been checked word-for-word "
             "against the official transcript. These are management's own statements and targets, "
             "not forecasts by this report.", size=7.8, colour=_GREY)
    pdf.ln(2)
    for c in data.companies:
        o = c.outlook
        srcs = ", ".join(s["label"] for s in (o.sources if o else []))
        pdf.h2(f"{c.name} ({c.symbol})", f"From earnings calls: {srcs}" if srcs else "")
        if not o or not o.available:
            pdf.para(o.note if o else "Outlook not requested.", colour=_GREY)
            continue
        for p in o.points:
            if pdf.get_y() > 262:
                pdf.add_page()
            pdf.style(8.6, bold=True, colour=_NAVY)
            pdf.cell(32, 4.6, pdf.t(_CATEGORY_LABELS.get(p.category, p.category)))
            pdf.style(8.6)
            pdf.multi_cell(0, 4.6, pdf.t(p.statement), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_x(_MARGIN + 32)
            pdf.style(7.4, colour=_GREY)
            pdf.multi_cell(0, 3.9, pdf.t(f"“{p.quote}” — {p.source} call"),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1.6)


def _investability(pdf: _PackPDF, data: PackData):
    from src.backtest.track_record import track_record_lines
    pdf.add_page()
    pdf.h1("Is it investable?")
    pdf.para("This section does not say buy or sell. It shows what the valuation model estimates "
             "each business is worth against today's price, how sure the model is, and — first — "
             "how reliable these signals have been in the past.", size=8.4)
    pdf.ln(2)

    # Track record box
    lines = track_record_lines(data.track_record)
    y0 = pdf.get_y()
    pdf.set_fill_color(*_LIGHT)
    pdf.rect(_MARGIN, y0, _BODY, 6 + 4.4 * sum(1 + len(l) // 120 for l in lines), "F")
    pdf.set_xy(_MARGIN + 3, y0 + 2)
    pdf.style(8.6, bold=True, colour=_NAVY)
    pdf.cell(0, 4.6, pdf.t("How reliable has this model been?"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for line in lines:
        pdf.set_x(_MARGIN + 3)
        pdf.style(7.9)
        pdf.multi_cell(_BODY - 6, 4.1, pdf.t(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(max(pdf.get_y(), y0) + 4)

    for c in data.companies:
        d = c.dcf
        pdf.h2(f"{c.name} ({c.symbol})")
        if d is None or d.intrinsic_value <= 0:
            pdf.para("The valuation model could not value this company with the available data.",
                     colour=_GREY)
        else:
            model = "residual-income model (for lenders)" if d.model == "RIM" else "cash-flow (DCF) model"
            rows = [
                ["Price", _rs(d.cmp)],
                ["Model value (middle estimate)", _rs(d.intrinsic_value)],
                ["Plausible range (10th–90th percentile)", f"{_rs(d.iv_low)} – {_rs(d.iv_high)}"],
                ["Chance the value is above the price",
                 _pct(d.prob_undervalued * 100, 0) if d.prob_undervalued is not None else "—"],
                ["Model view", (d.verdict or "—").title()],
            ]
            if d.implied_growth is not None:
                rows.append(["Growth the price assumes (reverse DCF)",
                             f"{d.implied_growth * 100:.1f}% a year to start"])
            pdf.table(["Valuation", f"{model}"], rows, [92, _BODY - 92], aligns=["L", "R"],
                      size=7.8, colours={(4, 1): _verdict_colour(d.verdict)})
        s = c.score
        if s is not None:
            pdf.para(f"Quality score {s.composite:.0f}/100 (grade {s.grade}), ranked {c.rank} of "
                     f"{len(NIFTY50_TICKERS)} in the Nifty 50 — quality {s.quality:.0f}, growth "
                     f"{s.growth:.0f}, earnings quality {s.earnings_quality:.0f}, "
                     f"Piotroski {s.piotroski}/9.", size=8)
        for n in getattr(d, "notes", None) or []:
            if "stale" in n.lower() or "not live" in n.lower():
                pdf.para(n, size=7.2, colour=_RED)
        pdf.ln(1)

    pdf.para("How to read this: a high chance that value exceeds price means the model's "
             "assumptions, applied to the reported numbers, give a higher value than the market "
             "price — not that the price will rise. Small changes in growth or discount-rate "
             "assumptions move these values a lot.", size=7.4, colour=_GREY)


def _disclosures(pdf: _PackPDF, data: PackData):
    pdf.add_page()
    pdf.h1("Sources and disclosures")
    pdf.para(SEBI_DISCLAIMER, size=8)
    pdf.ln(2)
    if "outlook" in data.spec.sections:
        pdf.para("Management outlook: summarised by a language model from earnings-call "
                 "transcripts; each point's quote was verified against the transcript by code, and "
                 "points that failed were removed. " + AI_CONTENT_NOTICE, size=7.6, colour=_GREY)
        pdf.ln(2)
    for line in data_provenance(rf_label=data.rf_label, rf_is_live=data.rf_is_live,
                                fundamentals_as_of=data.fundamentals_as_of or None,
                                fundamentals_age_days=data.fundamentals_age):
        pdf.para(line, size=7.6)
    transcripts = [(c.symbol, s) for c in data.companies if c.outlook for s in c.outlook.sources]
    if transcripts:
        pdf.ln(2)
        pdf.para("Earnings-call transcripts used", size=8, bold=True, colour=_NAVY)
        for sym, s in transcripts:
            pdf.style(6.8, colour=_GREY)
            pdf.multi_cell(0, 3.6, pdf.t(f"{sym} {s['label']}: {s['url']}"),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
