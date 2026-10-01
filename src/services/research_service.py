"""
Research service — the shared layer behind both the CLI and the web API.

Two problems this solves.

1. Cost. Composite scoring is cross-sectional: a company's ROCE percentile only
   means something relative to the other 49, so `score_all` must process every
   company. The CLI could afford to redo that per invocation; a web request
   cannot. Scores are therefore computed once and cached until the underlying
   Screener files change (detected by their newest mtime), so a stale cache is
   impossible by construction rather than by a timer.

2. Honesty. Every payload carries the freshness of the data it was built from and
   the provenance of the risk-free rate, so a UI can show the reader that a
   number rests on five-month-old fundamentals instead of hiding it.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from config.nifty50_tickers import NIFTY50_TICKERS
from src.reports.disclosure import is_report_trustworthy, provenance_from_result
from src.reports.report_data import load_report_financials
from src.scoring.composite_score import CompositeScorer
from src.utils.cache_manager import freshness_report
from src.utils.logger import get_logger
from src.utils.macro_rates import get_rf_context
from src.valuation.dcf_engine import DCFEngine, _SYM_SECTOR

logger = get_logger(__name__)

_SCREENER_DIR = Path("data/raw/screener")


def normalise_symbol(raw: str) -> str:
    """Accept 'tcs', 'TCS.NS', 'tcs.bo' and return the canonical NSE symbol."""
    return (raw or "").strip().upper().replace(".NS", "").replace(".BO", "")


def jsonable(value: Any) -> Any:
    """
    Coerce numpy scalars and non-finite floats into JSON-safe Python values.

    The scoring layer is numpy-based, so percentiles arrive as np.float64, which
    no JSON encoder will accept; and NaN/Infinity are not valid JSON even though
    Python's json module emits them by default. Both become None or a plain float
    here, once, at the service boundary.
    """
    import math

    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    # numpy scalars expose .item(); bool must be checked before int.
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


class ResearchService:
    """Thread-safe, cached access to valuations, scores and report payloads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._score_cache: dict[str, Any] = {}
        self._score_fingerprint: Optional[str] = None
        self._rf_context = None

    # ── Universe ──────────────────────────────────────────────────────────────

    @staticmethod
    def universe() -> list[dict[str, str]]:
        """All covered companies, with the sector the valuation engine assigns them."""
        return [
            {
                "symbol":  sym,
                "name":    name,
                "sector":  _SYM_SECTOR.get(sym, "DEFAULT"),
                "ticker":  f"{sym}.NS",
            }
            for sym, name in sorted(NIFTY50_TICKERS.items(), key=lambda kv: kv[1])
        ]

    @staticmethod
    def is_covered(symbol: str) -> bool:
        return normalise_symbol(symbol) in NIFTY50_TICKERS

    # ── Macro ─────────────────────────────────────────────────────────────────

    def rf_context(self):
        """The risk-free rate context, resolved once per service instance."""
        if self._rf_context is None:
            with self._lock:
                if self._rf_context is None:
                    self._rf_context = get_rf_context(quiet=True)
        return self._rf_context

    def macro(self) -> dict[str, Any]:
        ctx = self.rf_context()
        return {
            "rf_rate":    round(ctx.rate, 6),
            "rf_pct":     round(ctx.rate * 100, 3),
            "source":     ctx.source,
            "as_of":      ctx.as_of,
            "is_live":    ctx.is_live,
            "detail":     ctx.detail,
            "label":      ctx.label,
        }

    # ── Scores (cached across the whole universe) ──────────────────────────────

    def _fingerprint(self) -> str:
        """Newest mtime + file count across the Screener cache."""
        try:
            files = list(_SCREENER_DIR.glob("*.json"))
            newest = max((f.stat().st_mtime for f in files), default=0.0)
            return f"{len(files)}:{newest:.0f}"
        except Exception:
            return "unknown"

    def scores(self) -> dict[str, Any]:
        """
        Composite scores for every covered company, keyed by symbol.

        Recomputed only when the Screener cache changes.
        """
        fingerprint = self._fingerprint()
        with self._lock:
            if self._score_cache and self._score_fingerprint == fingerprint:
                return self._score_cache

        logger.info("[Service] computing composite scores for the full universe")
        results = CompositeScorer().score_all(list(NIFTY50_TICKERS.keys()))
        mapping = {r.symbol: r for r in results}

        with self._lock:
            self._score_cache = mapping
            self._score_fingerprint = fingerprint
        return mapping

    def score_for(self, symbol: str):
        return self.scores().get(normalise_symbol(symbol))

    def leaderboard(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Universe ranked by composite score — the table the web UI lands on."""
        ranked = sorted(
            self.scores().values(), key=lambda r: r.composite, reverse=True
        )
        if limit:
            ranked = ranked[:limit]
        return jsonable([
            {
                "symbol":    r.symbol,
                "name":      NIFTY50_TICKERS.get(r.symbol, r.symbol),
                "sector":    _SYM_SECTOR.get(r.symbol, "DEFAULT"),
                "composite": round(r.composite, 1),
                "grade":     r.grade,
                "quality":   round(r.quality, 1),
                "growth":    round(r.growth, 1),
                "piotroski": r.piotroski,
                "revenue_cagr_3y": round(r.revenue_cagr_3y, 1),
                "eps_cagr_3y":     round(r.eps_cagr_3y, 1),
                "de_ratio":  round(r.de_ratio, 2),
                "verdict":   r.verdict,
            }
            for r in ranked
        ])

    # ── Freshness ─────────────────────────────────────────────────────────────

    @staticmethod
    def freshness(symbol: str) -> dict[str, Any]:
        sym = normalise_symbol(symbol)
        return freshness_report([sym]).get(sym, {})

    def data_health(self) -> dict[str, Any]:
        """Fleet-wide data health, for the UI's status strip."""
        report = freshness_report(list(NIFTY50_TICKERS.keys()))
        ages   = [v["age_days"] for v in report.values() if v.get("age_days") is not None]
        stale  = [s for s, v in report.items() if v.get("stale")]
        ctx    = self.rf_context()
        return jsonable({
            "companies":          len(report),
            "stale_count":        len(stale),
            "stale_symbols":      sorted(stale),
            "oldest_age_days":    round(max(ages), 1) if ages else None,
            "newest_age_days":    round(min(ages), 1) if ages else None,
            "stale_threshold_days": 90,
            "rf_is_live":         ctx.is_live,
            "rf_label":           ctx.label,
            "fundamentals_ok":    not stale,
        })

    # ── Valuation ─────────────────────────────────────────────────────────────

    def valuation(self, symbol: str):
        """
        Run the valuation for one company.

        A fresh DCFEngine per call is deliberate: it is cheap (the expensive part
        is the Monte Carlo, which must run anyway) and it keeps the shared
        risk-free-rate context rather than re-fetching it.
        """
        sym = normalise_symbol(symbol)
        engine = DCFEngine(rf_context=self.rf_context(), quiet=True)
        return engine.value(sym)

    # ── Assembled report payload ──────────────────────────────────────────────

    def report(self, symbol: str, include_financials: bool = True) -> dict[str, Any]:
        """
        Everything the UI needs for one company, as plain JSON-safe types.

        Includes the caveats: `warnings` is non-empty whenever the numbers rest on
        stale fundamentals or a non-live discount rate.
        """
        sym = normalise_symbol(symbol)
        if sym not in NIFTY50_TICKERS:
            raise KeyError(f"{symbol} is not in the covered universe")

        name      = NIFTY50_TICKERS[sym]
        dcf       = self.valuation(sym)
        score     = self.score_for(sym)
        fresh     = self.freshness(sym)
        ok, warns = is_report_trustworthy(dcf, fresh)

        payload: dict[str, Any] = {
            "symbol":  sym,
            "name":    name,
            "ticker":  f"{sym}.NS",
            "sector":  _SYM_SECTOR.get(sym, "DEFAULT"),
            "valuation": {
                "model":            dcf.model,
                "cmp":              dcf.cmp,
                "intrinsic_value":  dcf.intrinsic_value,
                "iv_low":           dcf.iv_low,
                "iv_high":          dcf.iv_high,
                "margin_of_safety": dcf.margin_of_safety,
                "margin_of_safety_pct": (
                    round(dcf.margin_of_safety * 100, 1)
                    if dcf.margin_of_safety is not None else None
                ),
                "wacc":             round(dcf.wacc, 4),
                "wacc_pct":         round(dcf.wacc * 100, 2),
                "growth_input":     dcf.fcf_cagr,
                "terminal_value":   dcf.terminal_value,
                "pv_explicit":      dcf.pv_fcf,
                "shares_cr":        dcf.shares_cr,
                "verdict":          dcf.verdict,
                "notes":            list(dcf.notes or []),
            },
            "score": None,
            "macro": self.macro(),
            "freshness": {
                "fundamentals_as_of":  fresh.get("last_modified"),
                "fundamentals_age_days": fresh.get("age_days"),
                "stale":               fresh.get("stale", True),
            },
            "provenance": provenance_from_result(dcf, fresh),
            "trustworthy": ok,
            "warnings":    warns,
        }

        if score is not None:
            payload["score"] = {
                "composite":        round(score.composite, 1),
                "grade":            score.grade,
                "verdict":          score.verdict,
                "quality":          round(score.quality, 1),
                "growth":           round(score.growth, 1),
                "earnings_quality": round(score.earnings_quality, 1),
                "piotroski":        score.piotroski,
                "roce_percentile":  round(score.roce_pct, 1),
                "opm_percentile":   round(score.opm_pct, 1),
                "revenue_cagr_3y":  round(score.revenue_cagr_3y, 1),
                "eps_cagr_3y":      round(score.eps_cagr_3y, 1),
                "fcf_cagr_3y":      round(score.fcf_cagr_3y, 1),
                "de_ratio":         round(score.de_ratio, 2),
                "interest_coverage": round(score.interest_coverage, 2),
            }

        if include_financials:
            fin = load_report_financials(sym)
            payload["financials"] = {
                "years":       fin.years,
                "sales":       fin.sales,
                "net_profit":  fin.net_profit,
                "eps":         fin.eps,
                "opm":         fin.opm,
                "opm_label":   fin.opm_label,
                "roce":        fin.roce,
                "roce_label":  fin.roce_label,
                "cfo":         fin.cfo,
                "fcf":         fin.fcf,
                "is_bank":     fin.is_bank,
            }

        return jsonable(payload)


# Module-level singleton: the score cache is only useful if it is shared.
_service: Optional[ResearchService] = None
_service_lock = threading.Lock()


def get_service() -> ResearchService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ResearchService()
    return _service
