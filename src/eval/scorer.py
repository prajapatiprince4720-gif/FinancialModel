"""
Scores a free-text answer against a benchmark gold answer.

Numbers are extracted with their Indian/international scale words and converted
to the gold unit (₹ Crore for amounts), so "₹2.55 lakh crore", "2,55,324 Cr" and
"₹2.55 trillion" all resolve to ~255,000 crore.
"""

import re
from typing import Any

# multiplier to convert an amount in the named scale into ₹ Crore
_TO_CRORE = {
    "lakh crore": 1e5, "lakh-crore": 1e5, "lac crore": 1e5,
    "trillion": 1e5, "tn": 1e5,
    "thousand crore": 1e3,
    "crore": 1.0, "crores": 1.0, "cr": 1.0,
    "billion": 100.0, "bn": 100.0,
    "million": 0.1, "mn": 0.1,
    "lakh": 1e-2, "lakhs": 1e-2,
}
_SCALE_RE = "|".join(sorted((re.escape(k) for k in _TO_CRORE), key=len, reverse=True))
_NUM_RE = re.compile(rf"(-?\d[\d,]*\.?\d*)\s*({_SCALE_RE})?\b", re.IGNORECASE)

# Relative tolerance for reported amounts; absolute tolerance (points) for percents
TOLERANCE = {"crore": 0.02, "rupees": 0.02, "ratio": 0.05}
PERCENT_ABS_TOLERANCE = 1.0
RATIO_ABS_TOLERANCE = 0.05

ABSTAIN_PATTERNS = [
    "not available", "not yet", "no data", "don't have", "do not have", "doesn't have",
    "cannot provide", "can't provide", "unable to", "has not been reported",
    "not been reported", "not reported", "hasn't been", "has not been released",
    "not been released", "in the future", "future year", "hasn't happened", "not occurred",
    "no information", "isn't available", "is not available", "does not exist",
]


def extract_numbers(text: str, unit: str) -> list[float]:
    """Extract candidate numbers from text, converted into the gold unit."""
    out: list[float] = []
    for raw, scale in _NUM_RE.findall(text.replace("‑", "-").replace("−", "-")):
        try:
            value = float(raw.replace(",", "").rstrip("."))
        except ValueError:
            continue
        if unit == "crore":
            out.append(value * _TO_CRORE.get((scale or "crore").lower(), 1.0))
        else:
            out.append(value)
    return out


def _numeric_match(gold: float, candidates: list[float], unit: str) -> bool:
    for c in candidates:
        if unit == "percent":
            if abs(c - gold) <= PERCENT_ABS_TOLERANCE:
                return True
        elif unit == "ratio":
            if abs(c - gold) <= max(RATIO_ABS_TOLERANCE, abs(gold) * TOLERANCE["ratio"]):
                return True
        elif gold == 0:
            if abs(c) < 1e-9:
                return True
        elif abs(c - gold) / abs(gold) <= TOLERANCE.get(unit, 0.02):
            return True
    return False


def _normalize(text: str) -> str:
    return text.lower().replace("\u2019", "'").replace("\u2018", "'").replace(".", "")


_GENERIC_WORDS = {"bank", "india", "industries", "limited", "ltd", "the", "tata", "bajaj", "hdfc", "adani"}


def _aliases(symbol: str, name: str) -> list[str]:
    """Full name, symbol, name without 'Ltd'-style suffix, and its first distinctive word."""
    full = _normalize(name)
    words = [w for w in re.split(r"[\s&]+", full) if w]
    out = {full, symbol.lower(), " ".join(words[:2])}
    distinctive = [w.rstrip("'s") for w in words if w not in _GENERIC_WORDS and len(w) > 3]
    if distinctive:
        out.add(distinctive[0])
    return sorted(out, key=len, reverse=True)


def abstained(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ABSTAIN_PATTERNS)


def score(question: dict[str, Any], answer: str) -> dict[str, Any]:
    """Return {correct: bool, abstained: bool, reason: str}."""
    unit, gold = question["unit"], question["answer"]
    did_abstain = abstained(answer)

    if unit == "abstain":
        return {"correct": did_abstain, "abstained": did_abstain,
                "reason": "abstained" if did_abstain else "hallucinated an answer for a future year"}

    if unit == "ticker":
        names = question["names"]
        winner = gold.replace(".NS", "")
        loser = next(s for s in names if s != winner)
        t = _normalize(answer)
        # the first company name/symbol mentioned is treated as the pick
        def first_pos(sym: str) -> int:
            hits = [t.find(x) for x in _aliases(sym, names[sym]) if t.find(x) >= 0]
            return min(hits) if hits else 10**9
        correct = first_pos(winner) < first_pos(loser)
        return {"correct": correct, "abstained": did_abstain,
                "reason": "" if correct else f"picked {loser} or no clear pick"}

    candidates = extract_numbers(answer, unit)
    correct = _numeric_match(float(gold), candidates, unit)
    reason = "" if correct else ("abstained" if did_abstain else f"no number within tolerance of {gold}")
    return {"correct": correct, "abstained": did_abstain, "reason": reason}
