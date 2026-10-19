"""
Tests for the benchmark scorer and the verified-figures lookup.
These run locally — no API keys required.
"""

from src.eval import benchmark, scorer
from src.rag.financial_facts import FinancialFacts


class TestScorer:

    def test_amount_units_convert_to_crore(self):
        q = {"unit": "crore", "answer": 255324.0}
        for answer in ["₹2,55,324 Crore", "about ₹2.55 lakh crore", "₹2.55 trillion"]:
            assert scorer.score(q, answer)["correct"], answer

    def test_unit_slip_is_wrong(self):
        q = {"unit": "crore", "answer": 255324.0}
        assert not scorer.score(q, "₹130 Crore")["correct"]
        assert not scorer.score(q, "2.7 lakh crore")["correct"]

    def test_percent_tolerance(self):
        q = {"unit": "percent", "answer": 19.13}
        assert scorer.score(q, "net margin of 19.1%")["correct"]
        assert not scorer.score(q, "about 21%")["correct"]

    def test_compare_handles_name_variants(self):
        q = {"unit": "ticker", "answer": "DRREDDY",
             "names": {"DRREDDY": "Dr Reddy's Laboratories", "BEL": "Bharat Electronics"}}
        assert scorer.score(q, "Dr Reddy’s Laboratories had the higher revenue")["correct"]
        assert scorer.score(q, "Dr. Reddy's wins")["correct"]
        assert not scorer.score(q, "BEL had higher revenue than Dr Reddy's")["correct"]

    def test_abstain(self):
        q = {"unit": "abstain", "answer": None}
        assert scorer.score(q, "FY2033 has not been reported yet.")["correct"]
        assert not scorer.score(q, "Net profit was ₹90,000 crore.")["correct"]


class TestFinancialFacts:

    def test_contains_gold_answer_for_every_benchmark_question(self):
        facts = FinancialFacts()
        for q in benchmark.build():
            block = facts.for_question(q["question"], q["ticker"])
            if q["unit"] == "abstain":
                assert "NO DATA" in block, q["id"]
            elif q["unit"] == "ticker":
                assert all(f"({s})" in block for s in q["names"]), q["id"]
            else:
                numbers = scorer.extract_numbers(block, q["unit"])
                assert scorer._numeric_match(float(q["answer"]), numbers, q["unit"]), q["id"]

    def test_detects_aliases_and_multiple_companies(self):
        facts = FinancialFacts()
        assert facts.detect_symbols("Is Airtel bigger than Infosys?") == ["BHARTIARTL", "INFY"]
        assert facts.detect_symbols("HDFC Bank vs HDFC Life") == ["HDFCBANK", "HDFCLIFE"]

    def test_detects_short_fiscal_years(self):
        assert FinancialFacts().detect_years("TCS revenue in FY24 vs March 2025") == [2024, 2025]

    def test_unknown_company_returns_empty(self):
        assert FinancialFacts().for_question("What is a P/E ratio?") == ""
