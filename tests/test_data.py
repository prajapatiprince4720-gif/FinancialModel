"""
Tests for the data layer: snapshot history and the shared company loader.
Offline; every test uses a temporary database and synthetic payloads.
Run with: pytest tests/test_data.py -v
"""
import json
from datetime import date, datetime, timezone

import pytest

from src.data.company import CompanyData, _full_year_cut, fiscal_year_available_date, load_company
from src.data.snapshots import SnapshotStore, backfill_from_cache, content_hash


def _payload(np_values, price=100.0):
    years = ["Mar 2022", "Mar 2023", "Mar 2024", "TTM"]
    return {
        "current_price": price,
        "profit_loss": {"years": years, "rows": {"Net Profit": np_values, "EPS in Rs": [1, 2, 3, 4]}},
        "balance_sheet": {"years": years[:3], "rows": {"Reserves": [10, 20, 30]}},
        "cash_flow": {"years": [], "rows": {}},
        "ratios": {"years": [], "rows": {}},
    }


class TestSnapshotStore:

    @pytest.fixture
    def store(self, tmp_path):
        return SnapshotStore(tmp_path / "snap.db")

    def test_identical_content_is_not_duplicated(self, store):
        assert store.save("TCS", _payload([1, 2, 3, 4])) is True
        assert store.save("TCS", _payload([1, 2, 3, 4])) is False
        assert store.count() == 1

    def test_price_change_alone_is_not_a_new_snapshot(self, store):
        store.save("TCS", _payload([1, 2, 3, 4], price=100))
        assert store.save("TCS", _payload([1, 2, 3, 4], price=250)) is False
        assert content_hash(_payload([1], 1)) == content_hash(_payload([1], 2))

    def test_as_of_returns_data_known_at_that_date(self, store):
        t1 = datetime(2026, 4, 23, tzinfo=timezone.utc)
        t2 = datetime(2026, 10, 1, tzinfo=timezone.utc)
        store.save("TCS", _payload([1, 2, 3, 4]), fetched_at=t1)
        store.save("TCS", _payload([1, 2, 3, 9]), fetched_at=t2)

        assert store.as_of("TCS", "2026-01-01") is None
        assert store.as_of("TCS", "2026-06-30")["profit_loss"]["rows"]["Net Profit"][-1] == 4
        assert store.as_of("TCS", "2026-12-31")["profit_loss"]["rows"]["Net Profit"][-1] == 9
        assert [h["fetched_at"][:10] for h in store.history("TCS")] == ["2026-04-23", "2026-10-01"]

    def test_backfill_is_idempotent_and_skips_empty_files(self, tmp_path):
        cache = tmp_path / "screener"
        cache.mkdir()
        (cache / "TCS.json").write_text(json.dumps(_payload([1, 2, 3, 4])))
        (cache / "EMPTY.json").write_text(json.dumps({}))
        store = SnapshotStore(tmp_path / "snap.db")

        assert backfill_from_cache(cache, store) == 1
        assert backfill_from_cache(cache, store) == 0
        assert store.symbols() == ["TCS"]


class TestCompanyLoader:

    @pytest.mark.parametrize("years, expected", [
        (["Mar 2023", "Mar 2024", "TTM"], 2),
        (["Mar 2023", "Mar 2024"], 2),
        (["Mar 2024", "Mar 2025", "Sep 2025"], 2),        # ZOMATO-style interim column
        (["Dec 2022", "Dec 2023", "TTM"], 2),             # December fiscal year
        ([], 0),
    ])
    def test_full_year_cut(self, years, expected):
        assert _full_year_cut(years) == expected

    def test_each_statement_trimmed_on_its_own_columns(self):
        cd = CompanyData("X", _payload([1, 2, 3, 4]))
        assert cd.years("profit_loss") == ["Mar 2022", "Mar 2023", "Mar 2024"]
        # The balance sheet has no TTM column, so its last year must survive.
        assert cd.series("balance_sheet", "Reserves") == [10, 20, 30]
        assert cd.ttm("profit_loss", "Net Profit") == 4

    def test_aligned_keeps_gaps_series_drops_them(self):
        cd = CompanyData("X", _payload([1, None, 3, 4]))
        assert cd.aligned("profit_loss", "Net Profit") == [1, None, 3]
        assert cd.series("profit_loss", "Net Profit") == [1, 3]

    def test_first_present_key_wins(self):
        raw = _payload([1, 2, 3, 4])
        raw["balance_sheet"]["rows"]["Borrowing"] = [5, 6, 7]
        cd = CompanyData("X", raw)
        assert cd.series("balance_sheet", "Borrowings", "Borrowing") == [5, 6, 7]

    def test_lender_classification(self):
        assert CompanyData("HDFCBANK", {}).is_lender
        assert CompanyData("HDFCLIFE", {}).is_insurer
        assert not CompanyData("TCS", {}).is_lender

    def test_load_missing_symbol_returns_none(self, tmp_path):
        assert load_company("NOPE", cache_dir=tmp_path) is None


class TestReportingLag:

    @pytest.mark.parametrize("label, expected", [
        ("Mar 2024", date(2024, 5, 31)),
        ("Dec 2023", date(2024, 2, 29)),    # leap year
        ("Oct 2023", date(2023, 12, 31)),
        ("Nov 2023", date(2024, 1, 31)),
        ("TTM", None),
    ])
    def test_available_date(self, label, expected):
        assert fiscal_year_available_date(label) == expected
