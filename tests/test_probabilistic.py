"""
Tests for the probabilistic DCF, checked against closed-form results.
Run with: pytest tests/test_probabilistic.py -v
"""
import math

import numpy as np
import pytest

from src.valuation.probabilistic import (
    CORRELATION, SIGMA, VARIABLES, DCFInputs, fade_path, implied_initial_growth,
    sample_assumptions, sensitivities, simulate, value_per_share, verdict_from_probability,
)


def _inputs(**kw):
    base = dict(base_cf=100.0, g0=0.12, rf=0.07, beta=1.0, erp=0.05, shares=10.0,
                equity_cash_flows=True, g_inf=0.05, horizon=10, half_life=4.0)
    base.update(kw)
    return DCFInputs(**base)


class TestFadeAndValue:

    def test_fade_starts_near_g0_and_converges_to_g_inf(self):
        path = fade_path(0.20, 0.05, horizon=40, half_life=4.0)[0]
        assert path[0] == pytest.approx(0.05 + 0.15 * 2 ** (-1 / 4))
        assert path[3] == pytest.approx(0.05 + 0.15 / 2)        # one half-life at t = 4
        assert path[-1] == pytest.approx(0.05 + 0.15 * 2 ** -10)  # ten half-lives
        assert path[9] == pytest.approx(0.05 + 0.15 * 2 ** -2.5)  # ~18% of the gap left at year 10
        assert np.all(np.diff(path) < 0)

    def test_constant_growth_equals_gordon_closed_form(self):
        """With g₀ = g∞ the model is a growing perpetuity: V = CF₀(1+g)/(r−g)."""
        inp = _inputs(g0=0.05, g_inf=0.05)
        r = inp.discount_rate()                                  # 0.07 + 0.05 = 0.12
        gordon = 100 * 1.05 / (r - 0.05) / 10
        assert value_per_share(inp) == pytest.approx(gordon, rel=1e-10)

    def test_net_debt_deducted_with_limited_liability(self):
        inp = _inputs(equity_cash_flows=False, w_equity=1.0)
        gross = value_per_share(inp)
        assert value_per_share(_inputs(equity_cash_flows=False, net_debt=200)) == pytest.approx(gross - 20)
        assert value_per_share(_inputs(equity_cash_flows=False, net_debt=1e9)) == 0.0

    def test_equity_cash_flows_ignore_debt(self):
        assert value_per_share(_inputs(net_debt=1e9)) > 0


class TestCalculus:

    def test_discount_rate_sensitivity_matches_analytic_derivative(self):
        """Gordon: ∂V/∂r = −V/(r−g), so % change per +1pp = −1/(r−g)."""
        inp = _inputs(g0=0.05, g_inf=0.05)
        r = inp.discount_rate()
        s = sensitivities(inp)
        assert s["discount_rate"] == pytest.approx(-1 / (r - 0.05), rel=1e-4)

    def test_growth_sensitivities_have_the_right_sign(self):
        s = sensitivities(_inputs())
        assert s["discount_rate"] < 0 < s["initial_growth"]
        assert s["terminal_growth"] > 0

    @pytest.mark.parametrize("true_g", [-0.05, 0.03, 0.12, 0.30])
    def test_reverse_dcf_recovers_the_growth_that_set_the_price(self, true_g):
        inp = _inputs()
        price = value_per_share(inp, g0=true_g)
        res = implied_initial_growth(inp, price)
        assert res.status == "solved"
        assert res.g0 == pytest.approx(true_g, abs=1e-5)
        assert res.iterations < 20                       # Newton, not pure bisection

    def test_reverse_dcf_out_of_range(self):
        inp = _inputs()
        assert implied_initial_growth(inp, 1e9).status == "above_range"
        assert implied_initial_growth(inp, 1e-6).status == "below_range"
        assert implied_initial_growth(inp, None).status == "no_price"


class TestSampling:

    def test_correlation_matrix_is_valid(self):
        assert np.allclose(CORRELATION, CORRELATION.T)
        assert np.all(np.linalg.eigvalsh(CORRELATION) > 0)      # positive definite

    def test_draws_have_target_spread_and_correlation(self):
        # Centre every variable mid-bounds so clipping barely bites.
        inp = _inputs(g0=0.12, g_inf=0.055, erp=0.06, beta=1.0)
        d = sample_assumptions(inp, n=200_000, seed=1)
        for name in VARIABLES:
            assert d[name].std() == pytest.approx(SIGMA[name], rel=0.08)
        x = np.column_stack([d[v] for v in VARIABLES])
        emp = np.corrcoef(x, rowvar=False)
        assert np.allclose(emp, CORRELATION, atol=0.05)

    def test_draws_are_fat_tailed(self):
        """Student-t (ν=5) has excess kurtosis 6; a normal has 0."""
        d = sample_assumptions(_inputs(), n=200_000, seed=2)
        z = (d["beta"] - d["beta"].mean()) / d["beta"].std()
        assert (z ** 4).mean() - 3 > 1.0

    def test_seeded_runs_are_reproducible(self):
        a = sample_assumptions(_inputs(), n=1000, seed=7)["g0"]
        b = sample_assumptions(_inputs(), n=1000, seed=7)["g0"]
        assert np.array_equal(a, b)


class TestSimulation:

    def test_probability_and_median_never_contradict(self):
        inp = _inputs()
        for price in (5.0, 50.0, 150.0, 400.0, 2000.0):
            pv = simulate(inp, price, n=5000)
            if pv.median > price:
                assert pv.prob_undervalued > 0.5
            elif pv.median < price:
                assert pv.prob_undervalued < 0.5

    def test_percentiles_are_ordered(self):
        pv = simulate(_inputs(), 100.0, n=5000)
        assert pv.p10 <= pv.p25 <= pv.median <= pv.p75 <= pv.p90
        assert 0 < pv.terminal_share < 1

    def test_no_price_gives_no_probability(self):
        pv = simulate(_inputs(), None, n=1000)
        assert pv.prob_undervalued is None
        assert pv.implied.status == "no_price"

    @pytest.mark.parametrize("p, label", [
        (0.95, "LIKELY UNDERVALUED"), (0.65, "LEANS UNDERVALUED"), (0.5, "FAIRLY VALUED"),
        (0.30, "LEANS OVERVALUED"), (0.05, "LIKELY OVERVALUED"), (None, "NO PRICE DATA"),
    ])
    def test_verdict_bands(self, p, label):
        assert verdict_from_probability(p) == label
