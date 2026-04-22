"""
Converts raw yfinance JSON into a structured, human-readable financial narrative
that can be chunked and embedded into the vector store.

The output is a list of text chunks like:
  "RELIANCE.NS — Income Statement FY2024: Revenue ₹9.7L Cr, Net Profit ₹79,020 Cr..."
"""

from typing import Any

from src.utils.logger import get_logger
from src.utils.helpers import safe_float

logger = get_logger(__name__)

# Crore = 10 million; convert raw Yahoo Finance numbers (in USD or INR raw)
CR = 1e7  # 1 Crore


class FinancialProcessor:

    def process(self, raw_data: dict[str, Any]) -> list[dict[str, str]]:
        """
        Returns a list of dicts with keys: 'text', 'ticker', 'section', 'period'.
        Each dict is one chunk ready for embedding.
        """
        ticker = raw_data.get("ticker", "UNKNOWN")
        info = raw_data.get("info", {})
        chunks: list[dict[str, str]] = []

        chunks.append(self._process_company_profile(ticker, info))
        chunks.append(self._process_key_ratios(ticker, raw_data.get("key_ratios", {})))
        chunks += self._process_income_statement(ticker, raw_data.get("income_statement", {}))
        chunks += self._process_balance_sheet(ticker, raw_data.get("balance_sheet", {}))
        chunks += self._process_cash_flow(ticker, raw_data.get("cash_flow", {}))

        # When yfinance returns no financial statements (known bug for some Indian stocks),
        # build a fallback chunk from the info dict so the stock isn't left empty
        if not raw_data.get("income_statement") and info:
            chunks.append(self._process_info_fallback(ticker, info))

        # Filter empty chunks
        return [c for c in chunks if c.get("text", "").strip()]

    def _process_info_fallback(self, ticker: str, info: dict[str, Any]) -> dict[str, str]:
        """Fallback chunk when yfinance financial statements are unavailable."""
        lines = [f"Available Financial Data — {ticker} (from Yahoo Finance info):"]
        fields = {
            "marketCap": ("Market Cap", lambda v: f"₹{v/1e7:,.0f} Cr"),
            "totalRevenue": ("Total Revenue (TTM)", lambda v: f"₹{v/1e7:,.0f} Cr"),
            "netIncomeToCommon": ("Net Income (TTM)", lambda v: f"₹{v/1e7:,.0f} Cr"),
            "ebitda": ("EBITDA", lambda v: f"₹{v/1e7:,.0f} Cr"),
            "totalDebt": ("Total Debt", lambda v: f"₹{v/1e7:,.0f} Cr"),
            "totalCash": ("Total Cash", lambda v: f"₹{v/1e7:,.0f} Cr"),
            "freeCashflow": ("Free Cash Flow", lambda v: f"₹{v/1e7:,.0f} Cr"),
            "trailingPE": ("P/E Ratio", lambda v: f"{v:.2f}x"),
            "returnOnEquity": ("ROE", lambda v: f"{v*100:.1f}%"),
            "revenueGrowth": ("Revenue Growth (YoY)", lambda v: f"{v*100:.1f}%"),
        }
        has_data = False
        for key, (label, fmt) in fields.items():
            val = safe_float(info.get(key))
            if val:
                lines.append(f"  {label}: {fmt(val)}")
                has_data = True
        if not has_data:
            return {"text": "", "ticker": ticker, "section": "info_fallback", "period": "current"}
        lines.append("  Note: Detailed annual statements unavailable from Yahoo Finance for this ticker.")
        return {"text": "\n".join(lines), "ticker": ticker, "section": "info_fallback", "period": "current"}

    # ──────────────────────────────────────────────────────────────────────────

    def _process_company_profile(self, ticker: str, info: dict[str, Any]) -> dict[str, str]:
        name = info.get("longName", ticker)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        desc = info.get("longBusinessSummary", "No description available.")
        employees = info.get("fullTimeEmployees", "N/A")
        website = info.get("website", "N/A")
        country = info.get("country", "India")

        text = (
            f"Company Profile: {name} ({ticker})\n"
            f"Sector: {sector} | Industry: {industry} | Country: {country}\n"
            f"Full-time Employees: {employees} | Website: {website}\n\n"
            f"Business Description:\n{desc}"
        )
        return {"text": text, "ticker": ticker, "section": "company_profile", "period": "current"}

    def _process_key_ratios(self, ticker: str, ratios: dict[str, Any]) -> dict[str, str]:
        lines = [f"Key Financial Ratios — {ticker}:"]
        mapping = {
            "pe_ratio": "P/E Ratio (Trailing)",
            "forward_pe": "P/E Ratio (Forward)",
            "pb_ratio": "Price-to-Book (P/B)",
            "ps_ratio": "Price-to-Sales (P/S)",
            "ev_ebitda": "EV/EBITDA",
            "debt_to_equity": "Debt-to-Equity",
            "current_ratio": "Current Ratio",
            "roe": "Return on Equity (ROE)",
            "roa": "Return on Assets (ROA)",
            "profit_margin": "Net Profit Margin",
            "operating_margin": "Operating Margin",
            "revenue_growth": "Revenue Growth (YoY)",
            "earnings_growth": "Earnings Growth (YoY)",
            "dividend_yield": "Dividend Yield",
            "beta": "Beta (Market Sensitivity)",
            "market_cap": "Market Capitalisation",
        }
        for key, label in mapping.items():
            val = ratios.get(key)
            if val is not None:
                if key in ("market_cap",):
                    val_str = f"₹{safe_float(val)/CR:,.0f} Cr"
                elif key in ("roe", "roa", "profit_margin", "operating_margin",
                             "revenue_growth", "earnings_growth", "dividend_yield"):
                    val_str = f"{safe_float(val)*100:.2f}%"
                else:
                    val_str = f"{safe_float(val):.2f}"
                lines.append(f"  {label}: {val_str}")

        return {"text": "\n".join(lines), "ticker": ticker, "section": "key_ratios", "period": "current"}

    def _process_income_statement(self, ticker: str, stmt: dict[str, Any]) -> list[dict[str, str]]:
        chunks = []
        for period, values in stmt.items():
            if not isinstance(values, dict):
                continue
            lines = [f"Income Statement — {ticker} — Period ending {period}:"]
            key_map = {
                "Total Revenue": "Total Revenue",
                "Gross Profit": "Gross Profit",
                "Operating Income": "Operating Income (EBIT)",
                "EBITDA": "EBITDA",
                "Net Income": "Net Income",
                "Basic EPS": "EPS (Basic)",
            }
            for raw_key, display_name in key_map.items():
                val = safe_float(values.get(raw_key, 0))
                if val:
                    lines.append(f"  {display_name}: ₹{val/CR:,.0f} Cr")
            chunks.append({
                "text": "\n".join(lines),
                "ticker": ticker,
                "section": "income_statement",
                "period": str(period),
            })
        return chunks

    def _process_balance_sheet(self, ticker: str, stmt: dict[str, Any]) -> list[dict[str, str]]:
        chunks = []
        for period, values in stmt.items():
            if not isinstance(values, dict):
                continue
            lines = [f"Balance Sheet — {ticker} — Period ending {period}:"]
            key_map = {
                "Total Assets": "Total Assets",
                "Total Liabilities Net Minority Interest": "Total Liabilities",
                "Total Equity Gross Minority Interest": "Total Equity",
                "Cash And Cash Equivalents": "Cash & Equivalents",
                "Long Term Debt": "Long-term Debt",
                "Current Debt": "Short-term Debt",
            }
            for raw_key, display_name in key_map.items():
                val = safe_float(values.get(raw_key, 0))
                if val:
                    lines.append(f"  {display_name}: ₹{val/CR:,.0f} Cr")
            chunks.append({
                "text": "\n".join(lines),
                "ticker": ticker,
                "section": "balance_sheet",
                "period": str(period),
            })
        return chunks

    def _process_cash_flow(self, ticker: str, stmt: dict[str, Any]) -> list[dict[str, str]]:
        chunks = []
        for period, values in stmt.items():
            if not isinstance(values, dict):
                continue
            lines = [f"Cash Flow Statement — {ticker} — Period ending {period}:"]
            key_map = {
                "Operating Cash Flow": "Operating Cash Flow (CFO)",
                "Investing Cash Flow": "Investing Cash Flow (CFI)",
                "Financing Cash Flow": "Financing Cash Flow (CFF)",
                "Free Cash Flow": "Free Cash Flow",
                "Capital Expenditure": "Capital Expenditure (Capex)",
            }
            for raw_key, display_name in key_map.items():
                val = safe_float(values.get(raw_key, 0))
                if val:
                    lines.append(f"  {display_name}: ₹{val/CR:,.0f} Cr")
            chunks.append({
                "text": "\n".join(lines),
                "ticker": ticker,
                "section": "cash_flow",
                "period": str(period),
            })
        return chunks
