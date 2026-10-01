r"""
Probabilistic DCF: intrinsic value as a distribution, not a point.

The question an investor actually has is "how likely is it that this is worth
more than I'd pay?" — so that is what this module answers.

1. Growth fade (calculus)
   Growth starts at g₀ (estimated from history) and decays exponentially toward
   the long-run rate g∞:

       g(t) = g∞ + (g₀ − g∞)·e^(−λt),   λ = ln 2 / half-life

   The old engine held g₀ flat for ten years and then dropped to g∞ in one step,
   which both overstated high growers (RELIANCE at 22% for a decade) and put a
   kink in the cash-flow path. Here growth converges smoothly. With a 4-year
   half-life and a 10-year horizon, 2^(−2.5) ≈ 18% of the initial gap remains at
   year 10 — a residual step into the terminal rate, far smaller than the old one
   (a 20% grower steps from 7.7% rather than from 20% to 5.5%).

       CF_t = CF₀ · ∏_{s=1..t} (1 + g(s))
       V    = Σ_{t=1..N} CF_t / (1+r)^t  +  CF_N·(1+g∞) / (r − g∞) / (1+r)^N

2. Inputs as correlated random variables (probability, linear algebra)
   (g₀, g∞, ERP, β) are drawn jointly from a multivariate Student-t:

       Z = L·ε,  ε ~ N(0, I),  L = cholesky(C)      — correlation C
       X = Z / √(W/ν),  W ~ χ²_ν                    — fat tails, ν = 5

   Correlation matters: an inflation shock raises nominal long-run growth *and*
   the return investors demand, so g∞ and the discount rate move together and
   partly offset. Independent draws overstate the spread of outcomes.
   Student-t rather than normal because valuation surprises cluster in the tails.

3. Probability of undervaluation
   P(V > price) is the share of simulated values above the price. The reported
   intrinsic value is the median of the distribution, so P > 50% exactly when
   the intrinsic value exceeds the price — the number and the probability can
   never contradict each other.

4. Sensitivities (partial derivatives)
   ∂V/∂r, ∂V/∂g₀, ∂V/∂g∞ by central differences, expressed as "% change in value
   per +1 percentage point". Tells the reader which assumption the answer
   hinges on.

5. Reverse DCF (Newton's method)
   Solve V(g₀) = price for g₀ — the starting growth the market is pricing in.
   V is increasing in g₀, so a safeguarded Newton iteration (bisection fallback
   inside a shrinking bracket) always converges.

Equity is a limited-liability claim: per scenario, equity = max(EV − net debt, 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Distribution assumptions ──────────────────────────────────────────────────
# These are priors, not facts. They are stated here, in one place, so they can be
# tested (Phase 4: calibration — do 70% probabilities come true 70% of the time?)
# and replaced by estimates (Phase 2: empirical-Bayes growth, regression betas).

VARIABLES = ("g0", "g_inf", "erp", "beta")

SIGMA = {
    "g0":    0.04,     # starting growth: ±4pp (1σ)
    "g_inf": 0.0075,   # long-run nominal growth: ±0.75pp
    "erp":   0.010,    # equity risk premium: ±1pp
    "beta":  0.15,     # sector beta → company beta
}

# Rows/cols in VARIABLES order.
CORRELATION = np.array([
    #  g0     g_inf   erp    beta
    [ 1.00,  0.30,  -0.20,  0.00],   # g0: persistent growers have higher g∞; risk-off cuts both
    [ 0.30,  1.00,   0.40,  0.00],   # g_inf: inflation lifts nominal growth and required return
    [-0.20,  0.40,   1.00,  0.00],   # erp
    [ 0.00,  0.00,   0.00,  1.00],   # beta: firm-specific, independent
])

BOUNDS = {
    "g0":    (-0.10, 0.35),
    "g_inf": (0.030, 0.075),
    "erp":   (0.030, 0.100),
    "beta":  (0.40, 2.00),
}

STUDENT_T_DOF = 5
MIN_SPREAD    = 0.015      # r − g∞ floor; Gordon growth explodes as the spread → 0
N_SIMS        = 10_000


# ── Inputs ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DCFInputs:
    base_cf:  float            # ₹ Cr, starting annual cash flow
    g0:       float            # starting growth, point estimate
    rf:       float            # risk-free rate (observed, not simulated)
    beta:     float
    erp:      float
    shares:   float            # crore shares
    net_debt: float = 0.0      # ₹ Cr, deducted from EV (0 for equity cash flows)
    w_equity: float = 1.0      # capital-structure weights for WACC
    w_debt:   float = 0.0
    kd_after_tax: float = 0.0
    equity_cash_flows: bool = False   # True → discount at cost of equity
    g_inf:    float = 0.055
    horizon:  int   = 10
    half_life: float = 4.0     # years for (g₀ − g∞) to halve

    def discount_rate(self, beta=None, erp=None):
        """Cost of equity for equity cash flows, else WACC. Vectorises over arrays."""
        beta = self.beta if beta is None else beta
        erp  = self.erp if erp is None else erp
        ke = self.rf + beta * erp
        if self.equity_cash_flows:
            return ke
        return self.w_equity * ke + self.w_debt * self.kd_after_tax


# ── Core valuation (vectorised) ───────────────────────────────────────────────

def fade_path(g0, g_inf, horizon: int, half_life: float) -> np.ndarray:
    """Growth for years 1..N: g∞ + (g₀ − g∞)·e^(−λt). Shape (n, horizon)."""
    g0    = np.atleast_1d(np.asarray(g0, dtype=float))[:, None]
    g_inf = np.atleast_1d(np.asarray(g_inf, dtype=float))[:, None]
    lam   = math.log(2) / half_life
    t     = np.arange(1, horizon + 1, dtype=float)[None, :]
    return g_inf + (g0 - g_inf) * np.exp(-lam * t)


def value_components(inp: DCFInputs, g0, g_inf, r) -> dict[str, np.ndarray]:
    """
    PV of explicit cash flows, PV of terminal value, enterprise value and equity
    value per share, for arrays of (g₀, g∞, r) of equal length.
    """
    g0, g_inf, r = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (g0, g_inf, r))
    g_inf = np.minimum(g_inf, r - MIN_SPREAD)

    growth = fade_path(g0, g_inf, inp.horizon, inp.half_life)          # (n, N)
    cf     = inp.base_cf * np.cumprod(1.0 + growth, axis=1)            # (n, N)
    t      = np.arange(1, inp.horizon + 1, dtype=float)[None, :]
    disc   = (1.0 + r[:, None]) ** t
    pv_explicit = (cf / disc).sum(axis=1)
    tv          = cf[:, -1] * (1.0 + g_inf) / (r - g_inf)
    pv_terminal = tv / disc[:, -1]
    ev          = pv_explicit + pv_terminal
    debt        = 0.0 if inp.equity_cash_flows else inp.net_debt
    equity      = np.maximum(ev - debt, 0.0)
    return {
        "pv_explicit": pv_explicit, "pv_terminal": pv_terminal,
        "ev": ev, "per_share": equity / inp.shares,
    }


def value_per_share(inp: DCFInputs, g0=None, g_inf=None, r=None) -> float:
    """Scalar value at given (or base-case) assumptions."""
    g0    = inp.g0 if g0 is None else g0
    g_inf = inp.g_inf if g_inf is None else g_inf
    r     = inp.discount_rate() if r is None else r
    return float(value_components(inp, g0, g_inf, r)["per_share"][0])


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_assumptions(inp: DCFInputs, n: int = N_SIMS, seed: int = 42) -> dict[str, np.ndarray]:
    """Correlated, fat-tailed draws of (g₀, g∞, ERP, β) and the implied discount rate."""
    rng = np.random.default_rng(seed)
    L   = np.linalg.cholesky(CORRELATION)
    z   = rng.standard_normal((n, len(VARIABLES))) @ L.T              # corr = C
    w   = rng.chisquare(STUDENT_T_DOF, n)
    t   = z / np.sqrt(w / STUDENT_T_DOF)[:, None]
    # A t_ν variable has variance ν/(ν−2); rescale so SIGMA is the standard deviation.
    t  *= math.sqrt((STUDENT_T_DOF - 2) / STUDENT_T_DOF)

    centre = {"g0": inp.g0, "g_inf": inp.g_inf, "erp": inp.erp, "beta": inp.beta}
    draws  = {}
    for i, name in enumerate(VARIABLES):
        lo, hi = BOUNDS[name]
        draws[name] = np.clip(centre[name] + SIGMA[name] * t[:, i], lo, hi)
    draws["r"] = inp.discount_rate(beta=draws["beta"], erp=draws["erp"])
    return draws


# ── Calculus: sensitivities and reverse DCF ───────────────────────────────────

def sensitivities(inp: DCFInputs, h: float = 1e-4) -> dict[str, Optional[float]]:
    """
    % change in base-case value per +1 percentage point in each input.

    Central difference: ∂V/∂θ ≈ [V(θ+h) − V(θ−h)] / 2h, error O(h²).
    """
    base = value_per_share(inp)
    if base <= 0:
        return {"discount_rate": None, "initial_growth": None, "terminal_growth": None}
    r, g0, gi = inp.discount_rate(), inp.g0, inp.g_inf

    def pct_per_pp(dv_dtheta: float) -> float:
        return dv_dtheta * 0.01 / base * 100

    d_r  = (value_per_share(inp, r=r + h)      - value_per_share(inp, r=r - h))      / (2 * h)
    d_g0 = (value_per_share(inp, g0=g0 + h)    - value_per_share(inp, g0=g0 - h))    / (2 * h)
    d_gi = (value_per_share(inp, g_inf=gi + h) - value_per_share(inp, g_inf=gi - h)) / (2 * h)
    return {
        "discount_rate":   pct_per_pp(d_r),
        "initial_growth":  pct_per_pp(d_g0),
        "terminal_growth": pct_per_pp(d_gi),
    }


@dataclass
class ImpliedGrowth:
    g0: Optional[float]
    status: str                 # "solved" | "below_range" | "above_range" | "no_price"
    iterations: int = 0


def implied_initial_growth(inp: DCFInputs, price: Optional[float],
                           lo: float = -0.30, hi: float = 0.80,
                           tol: float = 1e-6, max_iter: int = 60) -> ImpliedGrowth:
    """
    Starting growth g₀ at which base-case value equals `price` (reverse DCF).

    Safeguarded Newton: f(g) = V(g) − price is increasing in g. Keep a bracket
    [lo, hi] with f(lo) < 0 < f(hi); take the Newton step when it stays inside the
    bracket, otherwise bisect. Converges quadratically near the root and can
    never diverge.
    """
    if not price or price <= 0:
        return ImpliedGrowth(None, "no_price")

    def f(g: float) -> float:
        return value_per_share(inp, g0=g) - price

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0:
        return ImpliedGrowth(lo, "below_range")     # price below value even at lo growth
    if f_hi < 0:
        return ImpliedGrowth(hi, "above_range")     # price needs more than hi growth

    g = min(max(inp.g0, lo), hi)
    for i in range(1, max_iter + 1):
        fg = f(g)
        if abs(fg) < tol * price:
            return ImpliedGrowth(g, "solved", i)
        if fg < 0:
            lo = g
        else:
            hi = g
        h = 1e-5
        slope = (f(g + h) - f(g - h)) / (2 * h)
        step  = g - fg / slope if slope > 0 else None
        g = step if step is not None and lo < step < hi else (lo + hi) / 2
        if hi - lo < 1e-9:
            break
    return ImpliedGrowth(g, "solved", max_iter)


# ── Verdict from probability ──────────────────────────────────────────────────

VERDICT_BANDS = (
    (0.80, "LIKELY UNDERVALUED"),
    (0.60, "LEANS UNDERVALUED"),
    (0.40, "FAIRLY VALUED"),
    (0.20, "LEANS OVERVALUED"),
    (0.00, "LIKELY OVERVALUED"),
)


def verdict_from_probability(p: Optional[float]) -> str:
    if p is None:
        return "NO PRICE DATA"
    for floor, label in VERDICT_BANDS:
        if p >= floor:
            return label
    return VERDICT_BANDS[-1][1]


# ── Full simulation ───────────────────────────────────────────────────────────

@dataclass
class ProbValuation:
    median:      float
    mean:        float
    base_case:   float
    p10: float
    p25: float
    p75: float
    p90: float
    prob_undervalued: Optional[float]
    discount_rate:    float
    pv_explicit:      float          # ₹ Cr, base case
    pv_terminal:      float          # ₹ Cr, base case
    terminal_share:   float          # share of EV from the terminal value
    sensitivities:    dict[str, Optional[float]] = field(default_factory=dict)
    implied:          Optional[ImpliedGrowth] = None
    zero_equity_share: float = 0.0   # share of scenarios where debt ≥ EV


def simulate(inp: DCFInputs, price: Optional[float], n: int = N_SIMS, seed: int = 42) -> ProbValuation:
    draws = sample_assumptions(inp, n=n, seed=seed)
    vals  = value_components(inp, draws["g0"], draws["g_inf"], draws["r"])["per_share"]

    base = value_components(inp, inp.g0, inp.g_inf, inp.discount_rate())
    ev   = float(base["ev"][0])
    p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
    prob = float(np.mean(vals > price)) if price and price > 0 else None

    return ProbValuation(
        median=float(p50), mean=float(vals.mean()),
        base_case=float(base["per_share"][0]),
        p10=float(p10), p25=float(p25), p75=float(p75), p90=float(p90),
        prob_undervalued=prob,
        discount_rate=float(inp.discount_rate()),
        pv_explicit=float(base["pv_explicit"][0]),
        pv_terminal=float(base["pv_terminal"][0]),
        terminal_share=float(base["pv_terminal"][0] / ev) if ev > 0 else 0.0,
        sensitivities=sensitivities(inp),
        implied=implied_initial_growth(inp, price),
        zero_equity_share=float(np.mean(vals <= 0)),
    )
