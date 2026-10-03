"""
Tests for free-text report requests and the research-pack PDF.
Offline: market caps, prices, valuations, scores and outlooks are stubs; only
the local Screener cache is read.
Run with: pytest tests/test_reports_pack.py -v
"""
from types import SimpleNamespace

import pytest
from pypdf import PdfReader

from src.data import load_company
from src.facts.outlook import Outlook, OutlookPoint
from src.reports.request import ReportSpec, parse_request, parse_rules
from src.reports.research_pack import build_research_pack, gather

CAPS = {"TCS": 12.0, "INFY": 6.0, "HCLTECH": 4.0, "WIPRO": 2.5, "TECHM": 1.4,
        "HDFCBANK": 9.0, "ICICIBANK": 8.0, "SBIN": 7.0, "AXISBANK": 3.0, "KOTAKBANK": 3.5,
        "INDUSINDBK": 0.5}


def _caps(symbols):
    return {s: CAPS.get(s, 1.0) for s in symbols}


class TestParseRequest:

    def test_the_motivating_request(self):
        spec = parse_request("I want last 5 years of Profit and loss of Top 5 IT compaines listed in "
                             "the nifty fifty and their fututre goals and is it investable or not",
                             market_caps=_caps)
        assert spec.symbols == ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"]
        assert spec.sections == ["comparison", "pnl", "outlook", "investability"]
        assert spec.years == 5 and "by market capitalisation" in spec.selection

    @pytest.mark.parametrize("text, symbols", [
        ("compare TCS vs Infosys", ["TCS", "INFY"]),
        ("SBI Life and HDFC Bank", ["SBILIFE", "HDFCBANK"]),           # "SBI Life" is not SBI
        ("Tech Mahindra plans", ["TECHM"]),                            # not the IT sector
        ("top 3 banks", ["HDFCBANK", "ICICIBANK", "SBIN"]),
        ("L&T and Dr Reddy's", ["LT", "DRREDDY"]),
    ])
    def test_companies(self, text, symbols):
        assert parse_request(text, market_caps=_caps).symbols == symbols

    def test_it_as_a_pronoun_is_not_a_sector(self):
        assert parse_rules("Is it investable? Tell me about Wipro").sector is None
        assert parse_rules("the IT sector").sector == "IT"
        assert parse_rules("best it companies").sector == "IT"

    def test_years_and_sections(self):
        spec = parse_request("HDFC Bank balance sheet and cash flow for three years", market_caps=_caps)
        assert spec.years == 3
        assert spec.sections == ["balance_sheet", "cash_flow"]

    def test_defaults_and_limits(self):
        spec = parse_request("Wipro please", market_caps=_caps)
        assert spec.sections == ["pnl", "outlook", "investability"]   # one company: no comparison
        assert parse_request("TCS for 40 years", market_caps=_caps).years == 10
        assert len(parse_request("top 25 Nifty 50 companies", market_caps=_caps).symbols) == 10

    def test_no_company_asks_back_without_guessing(self):
        spec = parse_request("make me a pdf", market_caps=_caps)
        assert not spec.ok and "Which companies" in spec.errors[0]

    def test_llm_is_used_only_when_rules_find_nothing_and_is_validated(self):
        class LLM:
            calls = 0

            def complete(self, **kw):
                LLM.calls += 1
                return '{"tickers": ["TCS", "FAKECO"], "sections": ["pnl", "bogus"], "years": 4}'

        # No name, sector or "top N" the rules know — only then is the model asked.
        assert parse_request("the firm N. Chandrasekaran once ran", llm=LLM(),
                             market_caps=_caps).symbols == ["TCS"]
        spec = parse_request("the firm N. Chandrasekaran once ran", llm=LLM(), market_caps=_caps)
        assert spec.sections == ["pnl"] and spec.years == 4
        parse_request("Infosys P&L", llm=LLM(), market_caps=_caps)
        assert LLM.calls == 2                      # the rules handled the third request

    @pytest.mark.parametrize("text, expected", [
        ("give me a pdf of TCS", True), ("PDF please", True),
        ("can I download a report on Infosys", True), ("what is TCS's ROE?", False),
        ("compare TCS and Infosys", False),
    ])
    def test_wants_pdf(self, text, expected):
        from src.reports.request import wants_pdf
        assert wants_pdf(text) is expected

    def test_ranking_falls_back_to_revenue_without_prices(self):
        spec = parse_request("top 2 IT companies", market_caps=lambda s: {x: None for x in s})
        assert "revenue" in spec.selection and len(spec.symbols) == 2


# ── PDF ───────────────────────────────────────────────────────────────────────

class _DCF:
    rf = SimpleNamespace(label="6.80% — test", is_live=True)

    def value(self, symbol):
        return SimpleNamespace(intrinsic_value=1200.0, cmp=1000.0, iv_low=900.0, iv_high=1500.0,
                               prob_undervalued=0.72, verdict="LEANS UNDERVALUED", model="DCF",
                               implied_growth=0.06, margin_of_safety=0.2, notes=[])


class _Scorer:
    def score_all(self, symbols):
        return [SimpleNamespace(symbol=s, composite=80 - i, grade="A", quality=70, growth=60,
                                earnings_quality=50, piotroski=7) for i, s in enumerate(symbols)]


class _Outlooks:
    def outlook(self, symbol):
        return Outlook(symbol, points=[OutlookPoint("margins", "Margin guidance kept at 20–22%.",
                                                    "Our operating margin guidance remains 20% to 22%.",
                                                    "Q1 FY27", "https://bse/x.pdf")],
                       sources=[{"label": "Q1 FY27", "url": "https://bse/x.pdf",
                                 "period_end": "2026-06-30"}])


@pytest.fixture
def need_cache():
    if any(load_company(s) is None for s in ("TCS", "INFY", "HDFCBANK")):
        pytest.skip("local Screener cache incomplete")


def _pdf_text(path) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(path).pages)


def test_pack_contains_requested_sections_and_mandatory_text(tmp_path, need_cache, monkeypatch):
    monkeypatch.setattr("src.backtest.track_record.TRACK_RECORD_PATH", tmp_path / "none.json")
    spec = ReportSpec("TCS and Infosys P&L, goals, investable?", ["TCS", "INFY"],
                      ["comparison", "pnl", "outlook", "investability"], years=5,
                      selection="Companies named in the request")
    data = gather(spec, outlook_extractor=_Outlooks(), dcf_engine=_DCF(), scorer=_Scorer(),
                  price_fn=lambda s: 1000.0)
    text = _pdf_text(build_research_pack(spec, output_dir=str(tmp_path), data=data))

    for heading in ("Side-by-side comparison", "Profit & loss", "Management's stated goals",
                    "Is it investable?", "Sources and disclosures"):
        assert heading in text
    assert "Our operating margin guidance remains 20% to 22%." in text      # quote shown
    assert "not yet been backtested" in text                                # no track record file
    assert "not registered with SEBI" in text                               # disclaimer always
    tcs = load_company("TCS")
    last_sales = tcs.series("profit_loss", "Sales")[-1]
    assert f"{last_sales:,.0f}" in text                                     # numbers from the cache
    assert tcs.years("profit_loss")[-1] in text


def test_bank_statement_uses_its_own_rows(tmp_path, need_cache):
    spec = ReportSpec("HDFC Bank balance sheet", ["HDFCBANK"], ["balance_sheet"], years=3)
    data = gather(spec, price_fn=lambda s: None)
    text = _pdf_text(build_research_pack(spec, output_dir=str(tmp_path), data=data))
    hdfc = load_company("HDFCBANK")
    for row in list(hdfc.raw["balance_sheet"]["rows"])[:3]:
        assert row in text
    assert "Is it investable?" not in text and "Management's stated goals" not in text
