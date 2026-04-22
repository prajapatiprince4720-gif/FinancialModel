"""
Converts raw Screener.in data into text chunks for embedding.

Produces one chunk per year per section, giving the AI 10 years of:
  - P&L narrative  (Sales, Operating Profit, Net Profit, EPS, OPM%)
  - Balance Sheet  (Assets, Liabilities, Equity, Debt, Investments)
  - Cash Flow      (CFO, CFI, CFF, Net Cash Flow)
  - Ratios         (ROCE, ROE, Debt/Equity, OPM%, Interest Coverage)
  - Peer summary   (top peers by market cap with key ratios)
"""

from typing import Any

from src.utils.logger import get_logger
from src.utils.helpers import safe_float

logger = get_logger(__name__)


class ScreenerProcessor:

    def process(self, data: dict[str, Any]) -> list[dict[str, str]]:
        if not data:
            return []

        symbol = data.get("symbol", "UNKNOWN")
        ticker = f"{symbol}.NS"
        chunks: list[dict[str, str]] = []

        chunks += self._process_section(
            ticker, data.get("profit_loss", {}), "P&L (Screener.in 10Y)",
            "profit_loss",
            row_map={
                "Sales": ("Revenue / Sales", "₹{:.0f} Cr"),
                "Expenses": ("Total Expenses", "₹{:.0f} Cr"),
                "Operating Profit": ("Operating Profit (EBITDA)", "₹{:.0f} Cr"),
                "OPM %": ("Operating Profit Margin", "{:.1f}%"),
                "Other Income": ("Other Income", "₹{:.0f} Cr"),
                "Interest": ("Interest Expense", "₹{:.0f} Cr"),
                "Depreciation": ("Depreciation", "₹{:.0f} Cr"),
                "Profit before tax": ("Profit Before Tax", "₹{:.0f} Cr"),
                "Net Profit": ("Net Profit", "₹{:.0f} Cr"),
                "EPS in Rs": ("EPS (₹)", "₹{:.2f}"),
            },
            scale_keys={"Sales", "Expenses", "Operating Profit", "Other Income",
                        "Interest", "Depreciation", "Profit before tax", "Net Profit"},
        )

        chunks += self._process_section(
            ticker, data.get("balance_sheet", {}), "Balance Sheet (Screener.in 10Y)",
            "balance_sheet",
            row_map={
                "Equity Capital": ("Equity Capital", "₹{:.0f} Cr"),
                "Reserves": ("Reserves & Surplus", "₹{:.0f} Cr"),
                "Borrowings": ("Total Borrowings (Debt)", "₹{:.0f} Cr"),
                "Other Liabilities": ("Other Liabilities", "₹{:.0f} Cr"),
                "Total Liabilities": ("Total Liabilities", "₹{:.0f} Cr"),
                "Fixed Assets": ("Fixed Assets (Net Block)", "₹{:.0f} Cr"),
                "CWIP": ("Capital Work in Progress", "₹{:.0f} Cr"),
                "Investments": ("Investments", "₹{:.0f} Cr"),
                "Other Assets": ("Other Assets", "₹{:.0f} Cr"),
                "Total Assets": ("Total Assets", "₹{:.0f} Cr"),
            },
            scale_keys={"Equity Capital", "Reserves", "Borrowings", "Other Liabilities",
                        "Total Liabilities", "Fixed Assets", "CWIP", "Investments",
                        "Other Assets", "Total Assets"},
        )

        chunks += self._process_section(
            ticker, data.get("cash_flow", {}), "Cash Flow (Screener.in 10Y)",
            "cash_flow",
            row_map={
                "Cash from Operating Activity": ("Cash from Operations (CFO)", "₹{:.0f} Cr"),
                "Cash from Investing Activity": ("Cash from Investing (CFI)", "₹{:.0f} Cr"),
                "Cash from Financing Activity": ("Cash from Financing (CFF)", "₹{:.0f} Cr"),
                "Net Cash Flow": ("Net Cash Flow", "₹{:.0f} Cr"),
            },
            scale_keys={"Cash from Operating Activity", "Cash from Investing Activity",
                        "Cash from Financing Activity", "Net Cash Flow"},
        )

        chunks += self._process_section(
            ticker, data.get("ratios", {}), "Key Ratios (Screener.in 10Y)",
            "ratios",
            row_map={
                "Debtor Days": ("Debtor Days", "{:.0f} days"),
                "Inventory Days": ("Inventory Days", "{:.0f} days"),
                "Days Payable": ("Days Payable", "{:.0f} days"),
                "Cash Conversion Cycle": ("Cash Conversion Cycle", "{:.0f} days"),
                "Working Capital Days": ("Working Capital Days", "{:.0f} days"),
                "ROCE %": ("Return on Capital Employed (ROCE)", "{:.1f}%"),
                "ROE %": ("Return on Equity (ROE)", "{:.1f}%"),
            },
            scale_keys=set(),
        )

        peers_chunk = self._process_peers(ticker, data.get("peers", []))
        if peers_chunk:
            chunks.append(peers_chunk)

        return [c for c in chunks if c.get("text", "").strip()]

    # ──────────────────────────────────────────────────────────────────────────

    def _process_section(
        self,
        ticker: str,
        section: dict[str, Any],
        title: str,
        section_key: str,
        row_map: dict[str, tuple[str, str]],
        scale_keys: set[str],
    ) -> list[dict[str, str]]:
        """
        Produce one chunk per year from a Screener section dict.
        Values from Screener are in Crores already (except ratios/EPS).
        """
        if not section:
            return []

        years: list[str] = section.get("years", [])
        rows: dict[str, list[Any]] = section.get("rows", {})

        if not years or not rows:
            return []

        chunks: list[dict[str, str]] = []
        for i, year in enumerate(years):
            lines = [f"{title} — {ticker} — {year}:"]
            has_data = False
            for raw_key, (display_name, fmt) in row_map.items():
                vals = rows.get(raw_key, [])
                if i < len(vals) and vals[i] is not None:
                    val = safe_float(vals[i])
                    if val is None:
                        continue
                    try:
                        val_str = fmt.format(val)
                    except Exception:
                        val_str = str(val)
                    lines.append(f"  {display_name}: {val_str}")
                    has_data = True

            if has_data:
                chunks.append({
                    "text": "\n".join(lines),
                    "ticker": ticker,
                    "section": f"screener_{section_key}",
                    "period": year,
                    "source": "screener.in",
                })

        return chunks

    def _process_peers(self, ticker: str, peers: list[dict[str, Any]]) -> dict[str, str] | None:
        if not peers:
            return None
        lines = [f"Peer Comparison (Screener.in) — {ticker}:"]
        for p in peers[:8]:
            name = p.get("Name", p.get("Company", ""))
            cmp = p.get("CMP", p.get("Price", ""))
            pe = p.get("P/E", "")
            mcap = p.get("Mar Cap", p.get("Market Cap", ""))
            roe = p.get("ROE %", p.get("ROE", ""))
            parts = [x for x in [name, f"Price:{cmp}", f"P/E:{pe}", f"MCap:{mcap}", f"ROE:{roe}"] if x and x != "–"]
            if parts:
                lines.append("  " + " | ".join(parts))

        return {
            "text": "\n".join(lines),
            "ticker": ticker,
            "section": "screener_peers",
            "period": "current",
            "source": "screener.in",
        }
