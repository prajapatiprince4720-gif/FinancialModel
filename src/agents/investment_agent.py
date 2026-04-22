"""
Handles investment simulation queries:
  - "What if I invest ₹1000 in Reliance right now?"
  - "What happens if I put ₹5000 in TCS?"
  - "What are the alternatives to HDFC Bank?"
"""

from typing import Any

import yfinance as yf

from config.nifty50_tickers import NIFTY50_TICKERS
from src.llm import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

INVESTMENT_SYSTEM = """You are a financial advisor for Indian retail investors.
A user wants to know what could happen if they invest a specific amount in a stock.

Your response must include ALL of these sections:

## 📊 Investment Snapshot
- Current stock price and how many shares ₹X buys

## 🎯 3 Scenarios (1-Year Outlook)
| Scenario | Probability | Stock Move | Your ₹X becomes |
Show Bull, Base, Bear cases with realistic probabilities.

## ⚠️ Key Risks
2-3 specific risks for this stock right now

## 💡 Alternatives to Consider
3 other Nifty 50 stocks worth comparing — with a one-line reason each

## 🧠 Bottom Line
One honest paragraph for a first-time investor.

IMPORTANT:
- Use actual rupee amounts, not just percentages
- Be honest about risks — don't just be optimistic
- This is EDUCATIONAL only — not SEBI-registered advice
- Use ₹ symbol, Indian number format (lakhs/crores)"""


class InvestmentAgent:

    def __init__(self) -> None:
        self.llm = get_llm_client()

    def run(self, ticker: str, amount: float, query: str = "") -> str:
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        company = NIFTY50_TICKERS.get(symbol, symbol)

        print(f"  Fetching live data for {company}...")
        market_data = self._fetch_stock_data(ticker)

        context = self._build_context(company, ticker, amount, market_data)

        user_msg = query or f"What happens if I invest ₹{amount:,.0f} in {company} right now?"
        return self.llm.complete(
            system_prompt=INVESTMENT_SYSTEM,
            user_message=user_msg,
            context=context,
            temperature=0.3,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_stock_data(self, ticker: str) -> dict[str, Any]:
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info
            hist = t.history(period="1y")

            current_price = float(hist["Close"].iloc[-1]) if not hist.empty else None
            year_ago_price = float(hist["Close"].iloc[0]) if not hist.empty else None
            yearly_return = (
                round((current_price - year_ago_price) / year_ago_price * 100, 1)
                if current_price and year_ago_price
                else None
            )

            return {
                "current_price": round(current_price, 2) if current_price else None,
                "market_cap_cr": round(getattr(fi, "market_cap", 0) / 1e7, 0),
                "pe_ratio": round(getattr(fi, "pe_ratio", 0) or 0, 1) or None,
                "year_high": round(getattr(fi, "year_high", 0) or 0, 2),
                "year_low": round(getattr(fi, "year_low", 0) or 0, 2),
                "yearly_return_pct": yearly_return,
            }
        except Exception as exc:
            logger.warning(f"Could not fetch live data: {exc}")
            return {}

    def _build_context(
        self, company: str, ticker: str, amount: float, data: dict[str, Any]
    ) -> str:
        price = data.get("current_price")
        shares = round(amount / price, 4) if price else "unknown"
        near_high = (
            price and data.get("year_high") and price >= data["year_high"] * 0.95
        )

        lines = [
            f"INVESTMENT SIMULATION DATA",
            f"Company: {company} ({ticker})",
            f"Investment Amount: ₹{amount:,.0f}",
            f"Current Price: ₹{price}" if price else "Current Price: unavailable",
            f"Shares Purchasable: {shares}",
            f"Market Cap: ₹{int(data.get('market_cap_cr', 0)):,} Cr",
            f"P/E Ratio: {data.get('pe_ratio', 'N/A')}",
            f"52-Week High: ₹{data.get('year_high', 'N/A')}",
            f"52-Week Low: ₹{data.get('year_low', 'N/A')}",
            f"1-Year Return: {data.get('yearly_return_pct', 'N/A')}%",
        ]
        if near_high:
            lines.append("NOTE: Stock is currently trading near its 52-week high.")

        return "\n".join(lines)
