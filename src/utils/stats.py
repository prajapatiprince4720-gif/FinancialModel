"""Small, robust statistics used by the valuation and fact-sheet layers."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def finite(values: Sequence[Optional[float]]) -> np.ndarray:
    return np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)


def robust_sd(values: Sequence[Optional[float]]) -> Optional[float]:
    """
    Standard deviation estimated from the median absolute deviation.

    σ ≈ 1.4826 · median(|x − median(x)|). For normal data this equals the usual
    standard deviation; unlike it, one wild year (NESTLEIND's ROIC when its
    equity briefly shrank) cannot blow it up.
    """
    x = finite(values)
    if len(x) < 3:
        return None
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def log_linear_growth(values: Sequence[Optional[float]], window: int = 6) -> Optional[float]:
    """
    Annual growth from an OLS fit of ln(x_t) = a + b·t over the last `window` years.

    Returns e^b − 1. A CAGR uses only the first and last points, so a single
    depressed base year (FY21) inflates it; the regression uses every year.
    Needs at least four positive points.
    """
    tail = list(values)[-window:]
    pts = [(t, v) for t, v in enumerate(tail) if v is not None and math.isfinite(v) and v > 0]
    if len(pts) < 4:
        return None
    t = np.array([p[0] for p in pts], dtype=float)
    y = np.log(np.array([p[1] for p in pts], dtype=float))
    tc = t - t.mean()
    b = float((tc @ (y - y.mean())) / (tc @ tc))
    return math.exp(b) - 1
