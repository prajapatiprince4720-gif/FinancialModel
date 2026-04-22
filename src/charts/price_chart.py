"""
Generates a 6-panel price history chart for a Nifty 50 stock.
Panels: 1D · 1W · 1M · 1Y · 5Y · 10Y

Saves as PNG to reports/ and auto-opens on Mac.
"""

import os
import subprocess
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import yfinance as yf
import pandas as pd

from config.nifty50_tickers import NIFTY50_TICKERS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# period label, yfinance period, yfinance interval, x-axis date format
PERIODS = [
    ("1 Day",    "1d",   "5m",  "%H:%M"),
    ("1 Week",   "5d",   "1h",  "%a %d"),
    ("1 Month",  "1mo",  "1d",  "%b %d"),
    ("1 Year",   "1y",   "1wk", "%b '%y"),
    ("5 Years",  "5y",   "1mo", "%Y"),
    ("10 Years", "10y",  "1mo", "%Y"),
]

BG       = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
SUBTEXT  = "#8b949e"
GREEN    = "#3fb950"
RED      = "#f85149"


class PriceChart:

    def plot(self, ticker: str, save_dir: str = "reports") -> str:
        """
        Generate and save the 6-panel chart. Returns the saved file path.
        Auto-opens the image on Mac.
        """
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        company = NIFTY50_TICKERS.get(symbol, symbol)

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 11), facecolor=BG)
        fig.suptitle(
            f"{company}  ({ticker})  —  Price History",
            color=TEXT, fontsize=17, fontweight="bold", y=0.98,
        )

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        for i, (label, period, interval, date_fmt) in enumerate(PERIODS):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            self._draw_panel(ax, ticker, period, interval, label, date_fmt)

        # Footer note
        fig.text(
            0.5, 0.01,
            "Data: Yahoo Finance  |  EquityLens AI  |  Not investment advice",
            ha="center", color=SUBTEXT, fontsize=8,
        )

        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
        path = os.path.join(save_dir, f"{symbol}_{ts}_chart.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info(f"Chart saved → {path}")

        # Auto-open on macOS
        subprocess.run(["open", path], check=False)
        return path

    # ──────────────────────────────────────────────────────────────────────────

    def _draw_panel(
        self,
        ax: Any,
        ticker: str,
        period: str,
        interval: str,
        label: str,
        date_fmt: str,
    ) -> None:
        # Styling
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=SUBTEXT, labelsize=7)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)

        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df.empty or len(df) < 2:
                self._no_data(ax, label)
                return

            # Handle MultiIndex columns (newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            prices = df["Close"].dropna()
            if len(prices) < 2:
                self._no_data(ax, label)
                return

            first = float(prices.iloc[0])
            last  = float(prices.iloc[-1])
            change_pct = (last - first) / first * 100
            color = GREEN if change_pct >= 0 else RED
            sign  = "+" if change_pct >= 0 else ""

            # Line + shaded fill
            ax.plot(prices.index, prices.values, color=color, linewidth=1.6, zorder=3)
            ax.fill_between(
                prices.index, prices.values, first,
                alpha=0.12, color=color, zorder=2,
            )
            ax.axhline(first, color=BORDER, linewidth=0.6, linestyle="--", zorder=1)

            # Title
            ax.set_title(
                f"{label}     ₹{last:,.1f}   ({sign}{change_pct:.1f}%)",
                color=color, fontsize=9, fontweight="bold", pad=6,
            )

            # Axes formatting
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6)

            # Grid
            ax.yaxis.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
            ax.set_axisbelow(True)

        except Exception as exc:
            logger.warning(f"Panel failed {ticker} {period}: {exc}")
            self._no_data(ax, label, str(exc)[:40])

    def _no_data(self, ax: Any, label: str, reason: str = "No data available") -> None:
        ax.set_title(label, color=SUBTEXT, fontsize=9)
        ax.text(
            0.5, 0.5, reason,
            ha="center", va="center", color=SUBTEXT, fontsize=8,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
