"""Consistent human formatting for fact-sheet values (Indian conventions)."""
from __future__ import annotations

import math
from typing import Any


def fmt_value(value: Any, unit: str) -> str:
    """Render a number in its unit. None/NaN renders as 'n/a'."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if unit == "cr":
        return f"₹{value:,.0f} Cr"
    if unit == "rs":
        return f"₹{value:,.2f}"
    if unit == "pct":
        return f"{value:.1f}%"
    if unit == "ratio_pct":            # stored as a fraction, shown as percent
        return f"{value * 100:.1f}%"
    if unit == "x":
        return f"{value:.2f}x"
    if unit == "days":
        return f"{value:.0f} days"
    return f"{value:,.2f}"
