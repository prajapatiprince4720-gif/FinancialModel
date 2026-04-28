"""
FII/DII Institutional Flow Fetcher.

NSE publishes daily FII and DII buy/sell data for free.
This fetcher pulls the last 30 days of activity, computes net flows,
and identifies whether institutions are accumulating or distributing.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests


_NSE_FII_URL = (
    "https://www.nseindia.com/api/fiidiiTradeReact"
)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/fii-dii-activity",
    "X-Requested-With": "XMLHttpRequest",
}

_CACHE_FILE = Path("data/raw/nse/fii_dii.json")
_CACHE_TTL  = 3600 * 4   # refresh every 4 hours


@dataclass
class FlowDay:
    date:       str
    fii_buy:    float   # ₹ Crore
    fii_sell:   float
    fii_net:    float
    dii_buy:    float
    dii_sell:   float
    dii_net:    float
    total_net:  float   # FII net + DII net


@dataclass
class FlowSummary:
    days:            list[FlowDay]
    fii_30d_net:     float    # cumulative FII net (₹ Cr)
    dii_30d_net:     float
    total_30d_net:   float
    fii_trend:       str      # "BUYING" / "SELLING" / "NEUTRAL"
    dii_trend:       str
    market_signal:   str      # "RISK-ON" / "RISK-OFF" / "MIXED" / "NEUTRAL"
    last_updated:    str


class FIIDIIFetcher:

    def fetch(self, force: bool = False) -> FlowSummary:
        data = self._from_cache(force)
        if data is None:
            data = self._from_nse()
            if data:
                self._save_cache(data)
            else:
                return self._empty_summary()
        return self._build_summary(data)

    # ── NSE API ───────────────────────────────────────────────────────────────

    def _from_nse(self) -> Optional[list[dict]]:
        session = requests.Session()
        try:
            # NSE requires a cookie from the main page first
            session.get("https://www.nseindia.com", headers=_HEADERS, timeout=10)
            time.sleep(0.5)
            resp = session.get(_NSE_FII_URL, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _from_cache(self, force: bool) -> Optional[list[dict]]:
        if force or not _CACHE_FILE.exists():
            return None
        age = time.time() - _CACHE_FILE.stat().st_mtime
        if age > _CACHE_TTL:
            return None
        try:
            with open(_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, data: list[dict]) -> None:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump(data, f)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_cr(self, val) -> float:
        try:
            return float(str(val).replace(",", "").strip())
        except Exception:
            return 0.0

    def _build_summary(self, raw: list[dict]) -> FlowSummary:
        days: list[FlowDay] = []
        for rec in raw[:30]:   # last 30 trading days
            try:
                date     = rec.get("date", "")
                fb       = self._parse_cr(rec.get("fiiBuyValue",  rec.get("fii_buy",  0)))
                fs       = self._parse_cr(rec.get("fiiSellValue", rec.get("fii_sell", 0)))
                db       = self._parse_cr(rec.get("diiBuyValue",  rec.get("dii_buy",  0)))
                ds       = self._parse_cr(rec.get("diiSellValue", rec.get("dii_sell", 0)))
                fn, dn   = fb - fs, db - ds
                days.append(FlowDay(
                    date=date, fii_buy=fb, fii_sell=fs, fii_net=fn,
                    dii_buy=db, dii_sell=ds, dii_net=dn, total_net=fn+dn,
                ))
            except Exception:
                continue

        if not days:
            return self._empty_summary()

        fii_net = sum(d.fii_net for d in days)
        dii_net = sum(d.dii_net for d in days)
        total   = fii_net + dii_net

        # Trend: compare last 5 days vs prior 5 days
        def _trend(series: list[float]) -> str:
            if len(series) < 5:
                return "NEUTRAL"
            recent = sum(series[:5])
            prior  = sum(series[5:10]) if len(series) >= 10 else sum(series[5:])
            if recent > 500 and recent > prior * 0.8:
                return "BUYING"
            if recent < -500 and recent < prior * 0.8:
                return "SELLING"
            return "NEUTRAL"

        fii_nets  = [d.fii_net for d in days]
        dii_nets  = [d.dii_net for d in days]
        ft = _trend(fii_nets)
        dt = _trend(dii_nets)

        if ft == "BUYING" and total > 0:
            signal = "RISK-ON"
        elif ft == "SELLING" and total < 0:
            signal = "RISK-OFF"
        elif ft != dt:
            signal = "MIXED"
        else:
            signal = "NEUTRAL"

        return FlowSummary(
            days=days,
            fii_30d_net=round(fii_net, 1),
            dii_30d_net=round(dii_net, 1),
            total_30d_net=round(total, 1),
            fii_trend=ft,
            dii_trend=dt,
            market_signal=signal,
            last_updated=datetime.now().strftime("%d %b %Y %H:%M"),
        )

    def _empty_summary(self) -> FlowSummary:
        return FlowSummary(
            days=[], fii_30d_net=0, dii_30d_net=0, total_30d_net=0,
            fii_trend="UNKNOWN", dii_trend="UNKNOWN",
            market_signal="UNKNOWN", last_updated="N/A",
        )
