"""
Tests for fact sheets and the deterministic answerer.
The statistics are checked against hand-computed values.
Run with: pytest tests/test_facts.py -v
"""
import math

import pytest

from src.data.company import CompanyData
from src.facts.answerer import answer_from_facts, detect_metrics, is_lookup, possessive
from src.facts.fact_sheet import FactSheet, _market, _metric_rows, _series, cagr, percentile_rank, trend_stats


class TestStatistics:

    def test_cagr(self):
        assert cagr([100, 110, 121], 2) == pytest.approx(0.10)
        assert cagr([100, 200], 1) == pytest.approx(1.0)

    @pytest.mark.parametrize("vals", [[0, 10, 20], [-5, 10, 20], [10, 20, None], [10]])
    def test_cagr_undefined(self, vals):
        assert cagr(vals, 2) is None

    def test_cagr_ignores_interior_gaps(self):
        # Only the endpoints enter the formula.
        assert cagr([10, None, 40], 2) == pytest.approx(1.0)

    def test_trend_slope_of_a_straight_line(self):
        s = trend_stats([1.0, 3.0, 5.0, 7.0, 9.0])     # x = 1 + 2t
        assert s["slope"] == pytest.approx(2.0)
        assert s["mean"] == pytest.approx(5.0)
        assert s["sd"] == pytest.approx(math.sqrt(10))  # sample sd of 1,3,5,7,9
        assert s["z"] == pytest.approx(4 / math.sqrt(10))

    def test_trend_uses_last_five_and_skips_gaps(self):
        s = trend_stats([100, 100, 1.0, None, 3.0, 4.0, 5.0])
        # Tail = [1, None, 3, 4, 5] at t = 0, 2, 3, 4 → OLS slope = 1.0
        assert s["slope"] == pytest.approx(1.0)

    def test_trend_needs_three_points(self):
        assert trend_stats([1.0, 2.0])["slope"] is None

    def test_percentile_rank_bounds_and_direction(self):
        pop = [1, 2, 3, 4, 5]
        assert percentile_rank(5, pop) == 100
        assert percentile_rank(1, pop) == 0
        assert percentile_rank(3, pop) == 50
        assert percentile_rank(1, pop, higher_is_better=False) == 100

    def test_percentile_rank_ties_and_small_populations(self):
        assert percentile_rank(2, [2, 2, 2]) == 50
        assert percentile_rank(1, [1]) is None
        assert percentile_rank(None, [1, 2, 3]) is None


def _company(symbol="TESTCO", lender=False):
    years = ["Mar 2021", "Mar 2022", "Mar 2023", "Mar 2024", "Mar 2025", "TTM"]
    pl_rows = {
        "Sales": [100, 110, 121, 133.1, 146.41, 150],
        "OPM %": [20, 21, 22, 23, 24, 24],
        "Net Profit": [10, 11, 12, 13, 14, 15],
        "EPS in Rs": [1, 1.1, 1.2, 1.3, 1.4, 1.5],
        "Interest": [1, 1, 1, 1, 2, 2],
        "Profit before tax": [13, 14, 15, 16, 18, 19],
        "Dividend Payout %": [50, 50, 50, 50, 50, 50],
        "Tax %": [25, 25, 25, 25, 25, 25],
    }
    bs_years = years[:-1]
    raw = {
        "profit_loss": {"years": years, "rows": pl_rows},
        "balance_sheet": {"years": bs_years, "rows": {
            "Equity Capital": [10] * 5, "Reserves": [40, 50, 60, 70, 90],
            "Borrowings": [20, 20, 20, 20, 20]}},
        "cash_flow": {"years": bs_years, "rows": {
            "Cash from Operating Activity": [12, 12, 13, 14, 15],
            "Free Cash Flow": [8, 8, 9, 10, 11]}},
        "ratios": {"years": bs_years, "rows": {"ROCE %": [18, 19, 20, 21, 22]}},
    }
    return CompanyData(symbol, raw)


class TestFactSheetMetrics:

    def test_series_derivations(self):
        s = _series(_company())
        assert s["revenue"][-1] == pytest.approx(146.41)
        assert s["book_value"][-1] == 100
        assert s["debt_to_equity"][-1] == pytest.approx(0.2)
        assert s["roe"][-1] == pytest.approx(14 / 90 * 100)    # on average equity (80, 100)
        assert s["interest_coverage"][-1] == pytest.approx((18 + 2) / 2)
        assert s["cfo_to_profit"][-1] == pytest.approx(15 / 14)

    def test_metric_rows_growth(self):
        cd = _company()
        rows = _metric_rows(_series(cd), cd.years())
        assert rows["revenue"]["cagr_3y"] == pytest.approx(0.10)
        assert rows["revenue"]["yoy_growth"] == pytest.approx(0.10)
        assert "cagr_3y" not in rows["operating_margin"]           # ratios get no CAGR
        assert rows["operating_margin"]["trend_slope"] == pytest.approx(1.0)

    def test_market_multiples_use_ttm_eps(self):
        cd = _company()
        m = _market(cd, _series(cd), price=30.0)
        assert m["pe"] == pytest.approx(30 / 1.5)
        assert m["pe_basis"] == "TTM EPS"
        assert m["shares_cr"] == pytest.approx(10)
        assert m["pb"] == pytest.approx(30 / (100 / 10))
        assert m["dividend_yield"] == pytest.approx(1.4 * 0.5 / 30 * 100)

    def test_market_without_price(self):
        cd = _company()
        m = _market(cd, _series(cd), price=None)
        assert "pe" not in m


def _sheet(**overrides):
    cd = _company()
    sheet = FactSheet(symbol="TESTCO", name="Test Co", sector="IT", is_lender=False,
                      years=cd.years(), data_as_of="2026-10-01", source="unit test")
    sheet.metrics = _metric_rows(_series(cd), sheet.years)
    sheet.market = _market(cd, _series(cd), price=30.0)
    for k, v in overrides.items():
        setattr(sheet, k, v)
    return sheet


class TestAnswerer:

    @pytest.mark.parametrize("q, expected", [
        ("What is the ROE?", ["roe"]),
        ("What is the earnings per share?", ["eps"]),           # not also net profit
        ("What is the P/E ratio?", ["pe"]),
        ("How much debt?", ["borrowings"]),
        ("What is the debt to equity?", ["debt_to_equity"]),    # not also borrowings
        ("Revenue and profit?", ["revenue", "net_profit"]),
        ("What happened last quarter?", []),
    ])
    def test_detect_metrics(self, q, expected):
        assert sorted(detect_metrics(q)) == sorted(expected)

    @pytest.mark.parametrize("q", ["Should I buy it?", "Why did margins fall?",
                                   "Is TCS better than Infosys?", "Is it overvalued?"])
    def test_judgement_questions_are_not_lookups(self, q):
        assert not is_lookup(q)

    def test_lookup_answer_contains_the_computed_number(self):
        a = answer_from_facts("What is the operating margin?", _sheet())
        assert a is not None and a.source == "facts"
        assert "24.0%" in a.text and "23.0%" in a.text
        assert "Source: unit test" in a.text

    def test_growth_question_reports_cagr(self):
        a = answer_from_facts("How fast has revenue grown?", _sheet())
        assert "10.0% a year over 3 years" in a.text

    def test_two_metrics_or_judgement_defers_to_llm(self):
        assert answer_from_facts("Revenue and profit?", _sheet()) is None
        assert answer_from_facts("Should I buy for the ROE?", _sheet()) is None

    def test_missing_price_is_explained_not_guessed(self):
        sheet = _sheet(market={"price": None})
        a = answer_from_facts("What is the P/E ratio?", sheet)
        assert "not available" in a.text

    def test_flags_are_appended(self):
        a = answer_from_facts("What is the ROE?", _sheet(flags=["Demerged."]))
        assert "Note: Demerged." in a.text

    def test_possessive(self):
        assert possessive("Tata Consultancy Services") == "Tata Consultancy Services'"
        assert possessive("Infosys") == "Infosys'"
        assert possessive("HDFC Bank") == "HDFC Bank's"
