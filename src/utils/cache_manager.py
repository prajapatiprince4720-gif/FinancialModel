"""
Screener cache age tracker and optional refresh orchestrator.

Screener.in publishes annual reports, so data changes meaningfully only at
FY-end (March/June). We flag files older than MAX_DAYS_BEFORE_STALE as stale
and print a warning — but we never block a valuation run.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_CACHE_DIR   = Path("data/raw/screener")
_PRICES_FILE = Path("data/raw/nse/live_prices.json")
_STALE_DAYS  = 90       # quarterly threshold


def _age_days(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 86400


def freshness_report(symbols: list[str]) -> dict:
    """
    Return a dict with freshness metadata for each symbol.
    Keys: symbol → {age_days, stale, last_modified, path_exists}
    """
    report = {}
    for sym in symbols:
        path = _CACHE_DIR / f"{sym}.json"
        if not path.exists():
            report[sym] = {"age_days": None, "stale": True, "last_modified": "MISSING", "path_exists": False}
        else:
            age = _age_days(path)
            ts  = datetime.fromtimestamp(path.stat().st_mtime)
            report[sym] = {
                "age_days":      round(age, 1),
                "stale":         age > _STALE_DAYS,
                "last_modified": ts.strftime("%Y-%m-%d"),
                "path_exists":   True,
            }
    return report


def warn_if_stale(symbols: list[str]) -> None:
    """Print a one-line warning if any Screener data is stale. Called by dcf/score commands."""
    rep    = freshness_report(symbols)
    stale  = [s for s, v in rep.items() if v["stale"]]
    if stale:
        count = len(stale)
        print(
            f"\033[33m  ⚠  {count} stock(s) have Screener data older than {_STALE_DAYS} days: "
            f"{', '.join(stale[:5])}{'...' if count > 5 else ''}. "
            f"Run: python main.py cache --refresh\033[0m\n"
        )


def print_cache_table(symbols: list[str]) -> None:
    """Pretty-print data age for all symbols."""
    rep   = freshness_report(symbols)
    now   = datetime.now()

    # Colour helpers
    def _g(s): return f"\033[92m{s}\033[0m"
    def _y(s): return f"\033[93m{s}\033[0m"
    def _r(s): return f"\033[91m{s}\033[0m"
    def _b(s): return f"\033[1m{s}\033[0m"
    def _d(s): return f"\033[2m{s}\033[0m"

    # Also check live prices file
    prices_age = None
    if _PRICES_FILE.exists():
        prices_age = round(_age_days(_PRICES_FILE), 1)

    print()
    print(_b("═" * 70))
    print(_b("  SCREENER DATA CACHE — FRESHNESS AUDIT"))
    print(_b("═" * 70))
    print(f"  {'Symbol':<14}  {'Last updated':<14}  {'Age (days)':<12}  Status")
    print("  " + "─" * 66)

    stale_count   = 0
    missing_count = 0
    for sym in sorted(symbols):
        info = rep[sym]
        if not info["path_exists"]:
            status = _r("MISSING")
            age_s  = _r("—")
            date_s = _r("—")
            missing_count += 1
        elif info["stale"]:
            status = _y(f"STALE (>{_STALE_DAYS}d)")
            age_s  = _y(f"{info['age_days']:.0f}")
            date_s = _y(info["last_modified"])
            stale_count += 1
        else:
            status = _g("fresh")
            age_s  = _g(f"{info['age_days']:.0f}")
            date_s = _g(info["last_modified"])

        print(f"  {sym:<14}  {date_s:<14}  {age_s:<12}  {status}")

    print("  " + "─" * 66)
    print(f"  Screener JSONs : {len(symbols) - stale_count - missing_count} fresh  |  "
          f"{stale_count} stale  |  {missing_count} missing")
    if prices_age is not None:
        p_s = _g(f"{prices_age:.1f}d") if prices_age < 1 else _y(f"{prices_age:.1f}d")
        print(f"  Live prices    : last refreshed {p_s} ago")
    print(f"  Stale threshold: >{_STALE_DAYS} days  |  Today: {now.strftime('%Y-%m-%d')}")
    print()


def refresh_stale(symbols: list[str], max_age_days: int = _STALE_DAYS, force: bool = False) -> None:
    """Re-fetch stale or missing Screener data. Prints progress."""
    from src.data_pipeline.fetchers.screener_fetcher import ScreenerFetcher

    rep     = freshness_report(symbols)
    targets = [s for s, v in rep.items() if force or v["stale"]]

    if not targets:
        print("  All Screener data is fresh — nothing to refresh.")
        return

    print(f"\n  Refreshing {len(targets)} companies from Screener.in...\n")
    fetcher = ScreenerFetcher()
    ok, fail = 0, []

    for i, sym in enumerate(targets, 1):
        print(f"  [{i}/{len(targets)}] {sym} ...", end="", flush=True)
        try:
            data = fetcher.fetch(sym)
            out  = _CACHE_DIR / f"{sym}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(data, f, indent=2)
            print(" ✓")
            ok += 1
        except Exception as exc:
            print(f" ✗  ({exc})")
            fail.append(sym)
        time.sleep(1.5)   # polite rate-limit for Screener.in

    print(f"\n  Done: {ok}/{len(targets)} refreshed.")
    if fail:
        print(f"  Failed: {', '.join(fail)}")
    print()
