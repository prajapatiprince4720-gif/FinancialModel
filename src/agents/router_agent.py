"""
Routes natural language queries to the right agent.
Uses the LLM to classify intent and extract entities (tickers, amounts, etc.)
"""

import json
import re
from dataclasses import dataclass, field

from config.nifty50_tickers import NIFTY50_TICKERS
from src.llm import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROUTER_SYSTEM = """You are a financial query router for Indian stock market (Nifty 50).
Given a user message, return ONLY a JSON object with this exact structure:
{
  "intent": "<one of: concept|single_stock|multi_stock|market_overview|investment_sim|full_research|general>",
  "tickers": ["SYMBOL"],
  "top_n": null,
  "amount": null,
  "metric": null
}

Intent definitions:
- concept: explaining a financial concept (P/E ratio, what is EBITDA, how does SIP work, etc.)
- single_stock: question about one specific company
- multi_stock: comparing or listing data for specific multiple companies
- market_overview: top N / leading / best / largest stocks by some criteria
- investment_sim: "what if I invest X amount" or "if I put X in Y" scenarios
- full_research: wants complete research report on a stock
- general: greetings or unrelated questions

Rules:
- tickers: Nifty 50 symbols WITHOUT .NS suffix (e.g. ["RELIANCE", "TCS", "HDFCBANK"])
- top_n: integer if user asks for "top N" or "leading N" companies (default 10 if unspecified)
- amount: number in rupees if user mentions an investment amount (extract just the number)
- metric: specific metric if mentioned (pe_ratio, market_cap, revenue, debt_to_equity, roe, dividend_yield, etc.)

Return ONLY valid JSON. No explanation, no markdown."""


@dataclass
class RouteResult:
    intent: str
    tickers: list[str] = field(default_factory=list)
    top_n: int | None = None
    amount: float | None = None
    metric: str | None = None


class RouterAgent:

    def __init__(self) -> None:
        self.llm = get_llm_client()

    def route(self, user_message: str) -> RouteResult:
        try:
            raw = self.llm.complete(
                system_prompt=ROUTER_SYSTEM,
                user_message=user_message,
                temperature=0.0,
            )
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return RouteResult(intent="general")

            data = json.loads(match.group())
            tickers = [
                t.upper().replace(".NS", "").replace(".BO", "")
                for t in (data.get("tickers") or [])
            ]
            tickers = [t for t in tickers if t in NIFTY50_TICKERS]

            return RouteResult(
                intent=data.get("intent", "general"),
                tickers=tickers,
                top_n=data.get("top_n"),
                amount=data.get("amount"),
                metric=data.get("metric"),
            )
        except Exception as exc:
            logger.warning(f"Router error: {exc}")
            return RouteResult(intent="general")
