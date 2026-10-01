"""
Tests for the probabilistic value-driver model, checked against closed forms.
Run with: pytest tests/test_probabilistic.py -v
"""
import numpy as np
import pytest

from src.utils.stats import log_linear_growth, robust_sd
from src.valuation.probabilistic import (
    CORRELATION, DEFAULT_SIGMA, VARIABLES, ValueDriverInputs, fade, implied_initial_growth,
    sample_assumptions, sensitivities, simulate, value_per_share, verdict_from_probability,
)


def _inputs(**kw):
    base = dict(nopat0=100.0, g0=0.12, roic0=0.25, rf=0.07, beta=1.0, erp=0.05,
                shares=10.0, g_inf=0.05, moat=0.5, horizon=10, half_life=4.0)
    base.update(kw)
    return ValueDriverInputs(**base)          # all-equity: r = 0.07 + 1.0·0.05 = 0.12


class TestValueDriverIdentities:

    def test_constant_growth_and_returns_match_mckinsey_formula(self):
        """g₀ = g∞ and ROIC = RONIC∞ (moat = 1): V = NOPAT₁(1 − g/RONIC)/(r − g)."""
        inp = _inputs(g0=0.05, g_inf=0.05, roic0=0.20, moat=1.0)
        r = inp.discount_rate()
        expected = 100 * 1.05 * (1 - 0.05 / 0.20) / (r - 0.05) / 10
        assert value_per_share(inp) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("g0", [-0.05, 0.0, 0.10, 0.30])
    def test_growth_adds_nothing_when_returns_equal_cost_of_capital(self, g0):
        """ROIC = RONIC = r ⇒ V = NOPAT₁/r whatever the growth path."""
        inp = _inputs(g0=g0, roic0=0.12, moat=1.0)               # r = 0.12
        nopat1 = 100 * (1 + inp.g_inf + (g0 - inp.g_inf) * 2 ** (-1 / 4))
        assert value_per_share(inp) == pytest.approx(nopat1 / 0.12 / 10, rel=1e-9)

    def test_growth_creates_value_only_above_cost_of_capital(self):
        hi = _inputs(roic0=0.30, moat=1.0)
        lo = _inputs(roic0=0.08, moat=1.0)
        assert value_per_share(hi, g0=0.20) > value_per_share(hi, g0=0.05)
        assert value_per_share(lo, g0=0.20) < value_per_share(lo, g0=0.05)

    def test_debt_deducted_with_limited_liability(self):
        gross = value_per_share(_inputs())
        assert value_per_share(_inputs(debt=200)) == pytest.approx(gross - 20)
        assert value_per_share(_inputs(debt=1e9)) == 0.0

    def test_fade_halves_the_gap_every_half_life(self):
        path = fade(0.20, 0.05, horizon=8, half_life=4.0)[0]
        assert path[3] == pytest.approx(0.05 + 0.15 / 2)
        assert path[7] == pytest.approx(0.05 + 0.15 / 4)


class TestCalculus:

    def test_discount_rate_sensitivity_matches_analytic_derivative(self):
        """No-growth perpetuity V = NOPAT/r: ∂V/∂r·0.01/V = −0.01/r."""
        inp = _inputs(g0=0.0, g_inf=0.0, roic0=0.12, moat=1.0)
        s = sensitivities(inp)
        # With g = 0 the value is NOPAT/r; % change per +1pp = −1/r (in %·pp units).
        assert s["discount_rate"] == pytest.approx(-1 / inp.discount_rate(), rel=1e-4)

    def test_sensitivity_signs(self):
        s = sensitivities(_inputs())
        assert s["discount_rate"] < 0
        assert s["initial_growth"] > 0 and s["roic"] > 0 and s["moat_durability"] > 0

    @pytest.mark.parametrize("true_g", [-0.05, 0.03, 0.12, 0.30])
    def test_reverse_dcf_recovers_growth(self, true_g):
        inp = _inputs()
        res = implied_initial_growth(inp, value_per_share(inp, g0=true_g))
        assert res.status == "solved"
        assert res.g0 == pytest.approx(true_g, abs=1e-5)
        assert res.iterations < 20

    def test_reverse_dcf_when_growth_destroys_value(self):
        """ROIC < r: value falls with growth, and the solver must still find the root."""
        inp = _inputs(roic0=0.06, moat=1.0)
        res = implied_initial_growth(inp, value_per_share(inp, g0=0.10))
        assert res.status == "solved" and res.g0 == pytest.approx(0.10, abs=1e-5)

    def test_reverse_dcf_degenerate_cases(self):
        inp = _inputs()
        assert implied_initial_growth(inp, 1e9).status == "above_range"
        assert implied_initial_growth(inp, 1e-6).status == "below_range"
        assert implied_initial_growth(inp, None).status == "no_price"


class TestSampling:

    def test_correlation_matrix_is_positive_definite(self):
        assert np.allclose(CORRELATION, CORRELATION.T)
        assert np.all(np.linalg.eigvalsh(CORRELATION) > 0)

    def test_draws_have_target_spread_and_correlation(self):
        inp = _inputs(g0=0.12, g_inf=0.055, erp=0.06, beta=1.0, roic0=0.30, moat=0.5)
        d = sample_assumptions(inp, n=200_000, seed=1)
        for name in ("g0", "g_inf", "erp", "beta", "roic"):
            assert d[name].std() == pytest.approx(inp.sigma()[name], rel=0.08)
        x = np.column_stack([d[v] for v in VARIABLES if v != "moat"])
        idx = [i for i, v in enumerate(VARIABLES) if v != "moat"]
        assert np.allclose(np.corrcoef(x, rowvar=False), CORRELATION[np.ix_(idx, idx)], atol=0.05)

    def test_draws_are_fat_tailed(self):
        d = sample_assumptions(_inputs(), n=200_000, seed=2)
        z = (d["beta"] - d["beta"].mean()) / d["beta"].std()
        assert (z ** 4).mean() - 3 > 1.0          # Student-t ν=5 → excess kurtosis 6

    def test_reproducible(self):
        a = sample_assumptions(_inputs(), n=500, seed=7)["g0"]
        assert np.array_equal(a, sample_assumptions(_inputs(), n=500, seed=7)["g0"])


class TestSimulation:

    def test_probability_and_median_never_contradict(self):
        inp = _inputs()
        for price in (20.0, 100.0, 200.0, 400.0, 2000.0):
            pv = simulate(inp, price, n=4000)
            if pv.median > price:
                assert pv.prob_undervalued > 0.5
            elif pv.median < price:
                assert pv.prob_undervalued < 0.5

    def test_percentiles_ordered(self):
        pv = simulate(_inputs(), 100.0, n=4000)
        assert pv.p10 <= pv.p25 <= pv.median <= pv.p75 <= pv.p90
        assert 0 < pv.terminal_share < 1

    @pytest.mark.parametrize("p, label", [
        (0.95, "LIKELY UNDERVALUED"), (0.65, "LEANS UNDERVALUED"), (0.5, "FAIRLY VALUED"),
        (0.30, "LEANS OVERVALUED"), (0.05, "LIKELY OVERVALUED"), (None, "NO PRICE DATA"),
    ])
    def test_verdict_bands(self, p, label):
        assert verdict_from_probability(p) == label


class TestRobustStats:

    def test_robust_sd_ignores_one_outlier(self):
        clean = [0.20, 0.22, 0.18, 0.21, 0.19]
        assert robust_sd(clean + [3.0]) == pytest.approx(robust_sd(clean), rel=0.5)
        assert np.std(clean + [3.0], ddof=1) > 1.0

    def test_log_linear_growth_exact_on_geometric_series(self):
        assert log_linear_growth([100 * 1.1 ** t for t in range(6)]) == pytest.approx(0.10)

    def test_log_linear_growth_resists_a_depressed_base_year(self):
        series = [70, 100, 110, 121, 133.1, 146.41]          # FY21-style dip, then 10%
        cagr = (series[-1] / series[0]) ** (1 / 5) - 1
        assert cagr > 0.15
        assert log_linear_growth(series) < cagr

    def test_log_linear_growth_needs_four_positive_points(self):
        assert log_linear_growth([1, 2, None, -1, 3]) is None
