"""
Tests for the walk-forward backtest.
Offline: prices are synthetic and fundamentals come from the local Screener cache.
Run with: pytest tests/test_backtest.py -v
"""
from datetime import date

import pandas as pd
import pytest

from src.backtest import BENCHMARK, Observation, PriceHistory, historical_rf, run_backtest
from src.backtest.engine import _period_stats
from src.data import load_company

SYMS = ["TCS", "INFY", "ITC", "HDFCBANK", "SBIN", "RELIANCE"]


def _prices(symbols, start="2020-05-25", end="2021-06-10", growth=None):
    """Business-day prices; each symbol grows linearly by `growth[sym]` over the window."""
    idx = pd.bdate_range(start, end)
    growth = growth or {}
    cols = {s: [100 * (1 + growth.get(s, 0.0) * i / (len(idx) - 1)) for i in range(len(idx))]
            for s in [*symbols, BENCHMARK]}
    frame = pd.DataFrame(cols, index=idx)
    return PriceHistory(frame, frame.copy())


class TestPriceHistory:

    def test_weekend_uses_last_trading_day(self):
        p = _prices(["TCS"])
        assert p.close_on("TCS", date(2020, 5, 30)) == p.close_on("TCS", date(2020, 5, 29))

    def test_stale_or_unknown_price_is_missing(self):
        p = _prices(["TCS"])
        assert p.close_on("TCS", date(2020, 1, 1)) is None     # before history
        assert p.close_on("TCS", date(2022, 1, 1)) is None     # long after it
        assert p.close_on("NOPE", date(2020, 6, 1)) is None

    def test_total_return(self):
        p = _prices(["TCS"], growth={"TCS": 0.5})
        assert p.total_return("TCS", date(2020, 5, 25), date(2021, 6, 10)) == pytest.approx(0.5)


class TestPeriodStats:

    def _obs(self, rows):
        return [Observation(date(2020, 6, 1), s, 100.0, score, puv, None, ret)
                for s, score, puv, ret in rows]

    def test_perfect_ranking_and_hit_rate(self):
        rows = [(f"S{i}", float(i), 0.9 if i >= 8 else 0.1, i / 100) for i in range(10)]
        p = _period_stats(date(2020, 6, 1), self._obs(rows), benchmark=0.02)
        assert p.score_ic == pytest.approx(1.0)
        assert p.dcf_ic > 0
        assert p.top_quintile_return == pytest.approx((0.08 + 0.09) / 2)
        assert (p.undervalued_n, p.undervalued_hits) == (2, 2)
        assert p.undervalued_hit_rate == 1.0

    def test_factor_ic_per_factor(self):
        obs = self._obs([(f"S{i}", float(i), 0.1, i / 100) for i in range(8)])
        for i, o in enumerate(obs):
            o.factors = {"quality": float(i), "growth": float(-i)}
        obs[0].factors.pop("growth")          # a missing value is dropped, not zero-filled
        p = _period_stats(date(2020, 6, 1), obs, benchmark=None)
        assert p.factor_ic["quality"] == pytest.approx(1.0)
        assert p.factor_ic["growth"] == pytest.approx(-1.0)

    def test_no_undervalued_names(self):
        rows = [(f"S{i}", float(i), 0.2, i / 100) for i in range(6)]
        p = _period_stats(date(2020, 6, 1), self._obs(rows), benchmark=None)
        assert p.undervalued_n == 0
        assert p.undervalued_return is None and p.undervalued_hit_rate is None


class TestRunBacktest:

    @pytest.fixture(autouse=True)
    def _need_cache(self):
        if any(load_company(s) is None for s in SYMS):
            pytest.skip("local Screener cache incomplete")

    @pytest.fixture(autouse=True)
    def _no_live_price(self, monkeypatch):
        def _fail(symbol):
            raise AssertionError("backtest fetched a live price")
        monkeypatch.setattr("src.valuation.dcf_engine._fetch_cmp", _fail)

    def test_views_see_only_published_years_and_the_day_price(self, monkeypatch):
        from src.valuation.dcf_engine import DCFEngine
        seen = {}
        real_value = DCFEngine.value

        def spy(self, symbol, as_of=None, data=None):
            seen[symbol] = (data, self.rf.rate)
            return real_value(self, symbol, data=data)

        monkeypatch.setattr(DCFEngine, "value", spy)
        prices = _prices(SYMS, growth={"TCS": 0.3})
        result = run_backtest([date(2020, 6, 1)], symbols=SYMS, prices=prices)

        assert len(result.periods) == 1 and set(seen) == set(SYMS)
        for view, rf in seen.values():
            assert view.is_historical
            assert view.years("profit_loss")[-1].startswith("Mar 2020")
            assert view.current_price == prices.close_on(view.symbol, date(2020, 6, 1))
            assert rf == historical_rf(date(2020, 6, 1)).rate
        by_sym = {o.symbol: o for o in result.observations}
        assert by_sym["TCS"].fwd_return > by_sym["INFY"].fwd_return
        assert {"quality", "growth", "earnings_yield"} <= set(by_sym["TCS"].factors)
        assert set(result.summary()["factor_ic"]) >= {"quality", "earnings_yield"}

    def test_horizon_past_last_price_is_skipped(self):
        result = run_backtest([date(2020, 6, 1), date(2021, 6, 1)], symbols=SYMS, prices=_prices(SYMS))
        assert [p.when for p in result.periods] == [date(2020, 6, 1)]
        assert result.skipped and result.skipped[0].startswith("2021-06-01")

    def test_excluded_symbol_never_enters(self):
        result = run_backtest([date(2020, 6, 1)], symbols=[*SYMS, "TATAMOTORS"],
                              prices=_prices([*SYMS, "TATAMOTORS"]))
        assert "TATAMOTORS" not in {o.symbol for o in result.observations}


def test_historical_rf_uses_latest_year_not_after_date():
    assert historical_rf(date(2020, 6, 1)).rate == historical_rf(date(2020, 12, 31)).rate
    assert historical_rf(date(2030, 1, 1)).rate == historical_rf(date(2025, 6, 1)).rate
    assert not historical_rf(date(2020, 6, 1)).is_live
