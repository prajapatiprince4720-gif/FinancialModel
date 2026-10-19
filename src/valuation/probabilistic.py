r"""
Probabilistic value-driver valuation: intrinsic value as a distribution.

The question an investor actually has is "how likely is it that this is worth
more than I'd pay?" — so that is what this module answers.

1. Value-driver model (why not plain free cash flow)
   Growth is not free: to grow operating profit (NOPAT) a company must invest,
   and how much depends on the return it earns on new capital (RONIC):

       I_t   = (NOPAT_{t+1} − NOPAT_t) / RONIC_t        investment needed
       FCF_t = NOPAT_t − I_t
       TV_N  = NOPAT_{N+1} · (1 − g∞/RONIC∞) / (r − g∞)   (McKinsey value-driver formula)

   Reported free cash flow subtracts *all* capex, growth capex included, so the
   earlier FCF-DCF valued fast-growing and capex-heavy companies near zero
   (ULTRACEMCO ₹521 vs a ₹12,010 price). Here growth is credited and charged at
   the company's own measured return on capital. Two identities follow, both
   tested: if RONIC = r growth adds nothing (V = NOPAT₁/r for any growth path);
   with constant g and RONIC it reduces to V = NOPAT₁(1 − g/RONIC)/(r − g).

2. Fade (calculus)
   Growth and returns both decay exponentially toward their long-run values:
       g(t)     = g∞     + (g₀    − g∞)    ·e^(−λt)
       RONIC(t) = RONIC∞ + (ROIC₀ − RONIC∞)·e^(−λt),   λ = ln 2 / half-life
       RONIC∞   = r + φ·(ROIC₀ − r)
   φ ∈ [0, 1] is moat durability: the share of today's excess return that
   competition leaves in place for ever. φ = 0 is perfect competition.

3. Inputs as correlated random variables (probability, linear algebra)
   (g₀, g∞, ERP, β, ROIC₀, φ) are drawn jointly from a multivariate Student-t:
       Z = L·ε, ε ~ N(0, I), L = cholesky(C);   X = Z / √(W/ν), W ~ χ²_ν, ν = 5
   An inflation shock raises nominal long-run growth *and* the return investors
   demand, so g∞ and the discount rate move together; independent draws would
   overstate the spread. Student-t because valuation surprises cluster in tails.

4. P(undervalued), sensitivities, reverse DCF
   P(V > price) is the share of scenarios above the price; the reported value is
   the median, so P > 50% exactly when value > price. Sensitivities are central-
   difference partial derivatives. The reverse DCF solves V(g₀) = price with a
   safeguarded Newton iteration; V may rise *or fall* with g₀ (growth destroys
   value when ROIC < r), so the bracket is direction-agnostic.

Equity is a limited-liability claim: per scenario, equity = max(EV − debt, 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Distribution assumptions ──────────────────────────────────────────────────
# Priors, not facts. Stated in one place so Phase 4 can test them (are 70%
# probabilities right 70% of the time?) and Phase 2 can replace them with
# estimates (empirical-Bayes growth, regression betas).

VARIABLES = ("g0", "g_inf", "erp", "beta", "roic", "moat")

DEFAULT_SIGMA = {
    "g0":    0.04,     # starting growth: ±4pp (1σ)
    "g_inf": 0.0075,   # long-run nominal growth: ±0.75pp
    "erp":   0.010,    # equity risk premium: ±1pp
    "beta":  0.15,     # sector beta → company beta
    "roic":  0.03,     # overridden per company by its own robust ROIC spread
    "moat":  0.20,     # durability of excess returns
}

# Rows/cols in VARIABLES order.
CORRELATION = np.array([
    #  g0     g_inf   erp    beta   roic   moat
    [ 1.00,  0.30,  -0.20,  0.00,  0.30,  0.00],   # g0
    [ 0.30,  1.00,   0.40,  0.00,  0.00,  0.00],   # g_inf: inflation lifts growth and required return
    [-0.20,  0.40,   1.00,  0.00,  0.00,  0.00],   # erp: risk-off cuts growth expectations
    [ 0.00,  0.00,   0.00,  1.00,  0.00,  0.00],   # beta
    [ 0.30,  0.00,   0.00,  0.00,  1.00,  0.00],   # roic: high-return firms tend to grow faster
    [ 0.00,  0.00,   0.00,  0.00,  0.00,  1.00],   # moat
])

BOUNDS = {
    "g0":    (-0.10, 0.35),
    "g_inf": (0.030, 0.075),
    "erp":   (0.030, 0.100),
    "beta":  (0.40, 2.00),
    "roic":  (0.02, 1.00),
    "moat":  (0.00, 1.00),
}

MOAT_PRIOR    = 0.5        # neutral: half of today's excess return survives
STUDENT_T_DOF = 5
MIN_SPREAD    = 0.015      # r − g∞ floor; the terminal formula explodes as it → 0
N_SIMS        = 10_000


# ── Inputs ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValueDriverInputs:
    nopat0:   float            # ₹ Cr, normalised current after-tax operating profit
    g0:       float            # starting growth, point estimate
    roic0:    float            # after-tax return on invested capital, point estimate
    rf:       float            # risk-free rate (observed, not simulated)
    beta:     float
    erp:      float
    shares:   float            # crore shares
    debt:     float = 0.0      # ₹ Cr, deducted from EV
    w_equity: float = 1.0
    w_debt:   float = 0.0
    kd_after_tax: float = 0.0
    g_inf:    float = 0.055
    moat:     float = MOAT_PRIOR
    roic_sd:  float = DEFAULT_SIGMA["roic"]
    horizon:  int   = 10
    half_life: float = 4.0

    def discount_rate(self, beta=None, erp=None):
        """WACC. Vectorises over arrays of beta / erp."""
        beta = self.beta if beta is None else beta
        erp  = self.erp if erp is None else erp
        return self.w_equity * (self.rf + beta * erp) + self.w_debt * self.kd_after_tax

    def centre(self) -> dict[str, float]:
        return {"g0": self.g0, "g_inf": self.g_inf, "erp": self.erp,
                "beta": self.beta, "roic": self.roic0, "moat": self.moat}

    def sigma(self) -> dict[str, float]:
        return {**DEFAULT_SIGMA, "roic": self.roic_sd}


# ── Core valuation (vectorised) ───────────────────────────────────────────────

def fade(start, end, horizon: int, half_life: float) -> np.ndarray:
    """end + (start − end)·e^(−λt) for t = 1..N. Shape (n, horizon)."""
    start = np.atleast_1d(np.asarray(start, dtype=float))[:, None]
    end   = np.atleast_1d(np.asarray(end, dtype=float))[:, None]
    lam   = math.log(2) / half_life
    t     = np.arange(1, horizon + 1, dtype=float)[None, :]
    return end + (start - end) * np.exp(-lam * t)


def value_components(inp: ValueDriverInputs, g0, g_inf, r, roic0, moat) -> dict[str, np.ndarray]:
    """Explicit-period PV, terminal PV, EV and equity value per share, vectorised."""
    g0, g_inf, r, roic0, moat = (
        np.atleast_1d(np.asarray(x, dtype=float)) for x in (g0, g_inf, r, roic0, moat)
    )
    g_inf     = np.minimum(g_inf, r - MIN_SPREAD)
    ronic_inf = np.maximum(r + moat * (roic0 - r), BOUNDS["roic"][0])

    N      = inp.horizon
    growth = fade(g0, g_inf, N, inp.half_life)                              # g_1..g_N
    ronic  = np.maximum(fade(roic0, ronic_inf, N, inp.half_life), BOUNDS["roic"][0])

    # NOPAT_1..NOPAT_{N+1}; year N+1 grows at g∞.
    growth_ext = np.concatenate([growth, g_inf[:, None]], axis=1)
    nopat      = inp.nopat0 * np.cumprod(1.0 + growth_ext, axis=1)          # (n, N+1)

    invest = (nopat[:, 1:] - nopat[:, :-1]) / ronic                         # I_1..I_N
    fcf    = nopat[:, :-1] - invest
    t      = np.arange(1, N + 1, dtype=float)[None, :]
    disc   = (1.0 + r[:, None]) ** t

    pv_explicit = (fcf / disc).sum(axis=1)
    tv          = nopat[:, -1] * (1.0 - g_inf / ronic_inf) / (r - g_inf)
    pv_terminal = tv / disc[:, -1]
    ev          = pv_explicit + pv_terminal
    equity      = np.maximum(ev - inp.debt, 0.0)
    return {"pv_explicit": pv_explicit, "pv_terminal": pv_terminal,
            "ev": ev, "per_share": equity / inp.shares}


def value_per_share(inp: ValueDriverInputs, **overrides) -> float:
    """Scalar value at the central assumptions, with any of g0/g_inf/r/roic0/moat overridden."""
    args = {"g0": inp.g0, "g_inf": inp.g_inf, "r": inp.discount_rate(),
            "roic0": inp.roic0, "moat": inp.moat}
    args.update(overrides)
    return float(value_components(inp, **args)["per_share"][0])


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_assumptions(inp: ValueDriverInputs, n: int = N_SIMS, seed: int = 42) -> dict[str, np.ndarray]:
    """Correlated, fat-tailed draws of every variable, plus the implied discount rate."""
    rng = np.random.default_rng(seed)
    L   = np.linalg.cholesky(CORRELATION)
    z   = rng.standard_normal((n, len(VARIABLES))) @ L.T          # correlation C
    w   = rng.chisquare(STUDENT_T_DOF, n)
    t   = z / np.sqrt(w / STUDENT_T_DOF)[:, None]
    # A t_ν variable has variance ν/(ν−2); rescale so sigma is the standard deviation.
    t  *= math.sqrt((STUDENT_T_DOF - 2) / STUDENT_T_DOF)

    centre, sigma = inp.centre(), inp.sigma()
    draws = {}
    for i, name in enumerate(VARIABLES):
        lo, hi = BOUNDS[name]
        draws[name] = np.clip(centre[name] + sigma[name] * t[:, i], lo, hi)
    draws["r"] = inp.discount_rate(beta=draws["beta"], erp=draws["erp"])
    return draws


# ── Calculus: sensitivities and reverse DCF ───────────────────────────────────

def sensitivities(inp: ValueDriverInputs, h: float = 1e-4) -> dict[str, Optional[float]]:
    """
    % change in base-case value per +1 percentage point (per +0.1 for moat).

    Central difference ∂V/∂θ ≈ [V(θ+h) − V(θ−h)] / 2h, error O(h²).
    """
    base = value_per_share(inp)
    keys = ("discount_rate", "initial_growth", "terminal_growth", "roic", "moat_durability")
    if base <= 0:
        return {k: None for k in keys}
    params = {"discount_rate": ("r", inp.discount_rate(), 0.01),
              "initial_growth": ("g0", inp.g0, 0.01),
              "terminal_growth": ("g_inf", inp.g_inf, 0.01),
              "roic": ("roic0", inp.roic0, 0.01),
              "moat_durability": ("moat", inp.moat, 0.10)}
    out = {}
    for key, (arg, x, unit) in params.items():
        d = (value_per_share(inp, **{arg: x + h}) - value_per_share(inp, **{arg: x - h})) / (2 * h)
        out[key] = d * unit / base * 100
    return out


@dataclass
class ImpliedGrowth:
    g0: Optional[float]
    status: str        # "solved" | "below_range" | "above_range" | "insensitive" | "no_price"
    iterations: int = 0


def implied_initial_growth(inp: ValueDriverInputs, price: Optional[float],
                           lo: float = -0.30, hi: float = 0.80,
                           tol: float = 1e-6, max_iter: int = 60) -> ImpliedGrowth:
    """
    Starting growth g₀ at which base-case value equals `price` (reverse DCF).

    Safeguarded Newton on f(g) = V(g) − price, keeping a sign-change bracket
    [a, b]. V usually rises with g, but falls when ROIC < r (growth destroys
    value) and is flat when ROIC = r, so the bracket does not assume a direction.
    Newton steps are taken when they land inside the bracket, else bisection.
    """
    if not price or price <= 0:
        return ImpliedGrowth(None, "no_price")

    def f(g: float) -> float:
        return value_per_share(inp, g0=g) - price

    a, b = lo, hi
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        rising = fb > fa
        if abs(fb - fa) < 1e-9 * price:
            return ImpliedGrowth(None, "insensitive")
        # No sign change: price is outside what any growth in [lo, hi] supports.
        too_high = (fa < 0 and fb < 0)
        if too_high:
            return ImpliedGrowth(hi if rising else lo, "above_range")
        return ImpliedGrowth(lo if rising else hi, "below_range")

    g = min(max(inp.g0, a), b)
    for i in range(1, max_iter + 1):
        fg = f(g)
        if abs(fg) < tol * price:
            return ImpliedGrowth(g, "solved", i)
        if (fg < 0) == (fa < 0):
            a, fa = g, fg
        else:
            b, fb = g, fg
        h = 1e-5
        slope = (f(g + h) - f(g - h)) / (2 * h)
        step = g - fg / slope if slope != 0 else None
        lo_b, hi_b = min(a, b), max(a, b)
        g = step if step is not None and lo_b < step < hi_b else (a + b) / 2
        if abs(b - a) < 1e-9:
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


def simulate(inp: ValueDriverInputs, price: Optional[float],
             n: int = N_SIMS, seed: int = 42) -> ProbValuation:
    d    = sample_assumptions(inp, n=n, seed=seed)
    vals = value_components(inp, d["g0"], d["g_inf"], d["r"], d["roic"], d["moat"])["per_share"]

    base = value_components(inp, inp.g0, inp.g_inf, inp.discount_rate(), inp.roic0, inp.moat)
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
