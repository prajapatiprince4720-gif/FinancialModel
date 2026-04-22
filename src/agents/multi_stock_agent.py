"""
Handles queries spanning multiple stocks:
  - "Give me P/E ratios for top 8 Nifty companies"
  - "Which banks have the lowest debt?"
  - "Top 10 Nifty stocks by market cap"
  - "Compare TCS, Infosys, Wipro"
"""

from typing import Any

import yfinance as yf

from config.nifty50_tickers import NIFTY50_TICKERS
from src.llm import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

MULTI_STOCK_SYSTEM = """You are a financial analyst specializing in Indian equities (Nifty 50).
You have been given real-time market data for multiple stocks.
Answer the user's question with:
- A clean formatted table where appropriate (use markdown)
- Key insights and comparisons
- Plain English explanation for retail investors
- Values in Indian format (₹, Crore abbreviated as Cr, percentages with %)
- Highlight the top performers and any red flags
Be concise, data-driven, and helpful."""


class MultiStockAgent:

    def __init__(self) -> None:
        self.llm = get_llm_client()

    def run(
        self,
        query: str,
        tickers: list[str] | None = None,
        top_n: int | None = None,
        metric: str | None = None,
    ) -> str:
        symbols = tickers if tickers else list(NIFTY50_TICKERS.keys())

        print(f"  Fetching live data for {len(symbols)} stocks...")
        data = self._fetch_market_data(symbols)

        # Sort by market cap and optionally slice to top N
        data_sorted = sorted(data, key=lambda x: x.get("market_cap_cr", 0), reverse=True)
        if top_n:
            data_sorted = data_sorted[:top_n]
        elif tickers:
            order = {t: i for i, t in enumerate(tickers)}
            data_sorted = sorted(data_sorted, key=lambda x: order.get(x["symbol"], 99))

        context = self._format_data(data_sorted)
        return self.llm.complete(
            system_prompt=MULTI_STOCK_SYSTEM,
            user_message=query,
            context=context,
            temperature=0.2,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_market_data(self, symbols: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for sym in symbols:
            try:
                ticker = yf.Ticker(f"{sym}.NS")
                fi = ticker.fast_info
                hist = ticker.history(period="2d")
                current_price = float(hist["Close"].iloc[-1]) if not hist.empty else None
                prev_price = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current_price
                change_pct = (
                    round((current_price - prev_price) / prev_price * 100, 2)
                    if current_price and prev_price
                    else None
                )
                market_cap_raw = getattr(fi, "market_cap", None)
                pe = getattr(fi, "pe_ratio", None)

                results.append({
                    "symbol": sym,
                    "company": NIFTY50_TICKERS.get(sym, sym),
                    "price": round(current_price, 2) if current_price else "N/A",
                    "change_pct": change_pct,
                    "market_cap_cr": round(market_cap_raw / 1e7, 0) if market_cap_raw else 0,
                    "pe_ratio": round(pe, 1) if pe else "N/A",
                    "52w_high": round(getattr(fi, "year_high", 0) or 0, 2) or "N/A",
                    "52w_low": round(getattr(fi, "year_low", 0) or 0, 2) or "N/A",
                })
            except Exception as exc:
                logger.warning(f"  Skipping {sym}: {exc}")
                results.append({
                    "symbol": sym,
                    "company": NIFTY50_TICKERS.get(sym, sym),
                    "market_cap_cr": 0,
                })
        return results

    def _format_data(self, data: list[dict[str, Any]]) -> str:
        lines = ["LIVE MARKET DATA (Nifty 50)\n"]
        for i, d in enumerate(data, 1):
            change = f"{d['change_pct']:+.2f}%" if isinstance(d.get("change_pct"), float) else "N/A"
            mcap = f"₹{int(d['market_cap_cr']):,} Cr" if d.get("market_cap_cr") else "N/A"
            lines.append(
                f"{i:2}. {d['company']} ({d['symbol']})"
                f" | Price: ₹{d.get('price', 'N/A')} ({change})"
                f" | Market Cap: {mcap}"
                f" | P/E: {d.get('pe_ratio', 'N/A')}"
                f" | 52W H/L: ₹{d.get('52w_high', 'N/A')} / ₹{d.get('52w_low', 'N/A')}"
            )
        return "\n".join(lines)
