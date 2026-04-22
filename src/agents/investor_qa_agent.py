"""
InvestorQAAgent — answers basic to intermediate investor questions.

Handles two modes:
  1. Concept questions  ("What is P/E ratio?", "How does SIP work?")
     → Uses the rich system prompt + no retrieval needed
  2. Stock questions    ("Is Reliance debt heavy?", "What is TCS's ROE?")
     → Retrieves from vector store + answers with data

Usage:
    agent = InvestorQAAgent()
    print(agent.ask("What is a P/E ratio?"))
    print(agent.ask("How much debt does HDFC Bank have?", ticker="HDFCBANK.NS"))
"""

import re
from typing import Any

from config.nifty50_tickers import NIFTY50_TICKERS
from src.llm import get_llm_client
from src.llm.prompts.investor_qa import (
    INVESTOR_QA_SYSTEM_PROMPT,
    INVESTOR_FAQ,
    build_investor_qa_query,
)
from src.rag.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Map common name variations to NSE symbols
_NAME_TO_SYMBOL: dict[str, str] = {
    name.lower(): sym for sym, name in NIFTY50_TICKERS.items()
}
# Also add symbol → symbol mapping for direct lookups
_NAME_TO_SYMBOL.update({sym.lower(): sym for sym in NIFTY50_TICKERS})


class InvestorQAAgent:

    def __init__(self) -> None:
        self.llm = get_llm_client()
        self.retriever = Retriever()

    def ask(self, question: str, ticker: str = "") -> str:
        """
        Answer an investor question.

        Args:
            question: The investor's question in plain English
            ticker:   Optional yfinance ticker (e.g. 'RELIANCE.NS') for stock-specific questions.
                      If not provided, the agent tries to detect the stock from the question.
        """
        # Check FAQ cache first for instant answers
        faq_answer = self._check_faq(question)
        if faq_answer:
            return faq_answer

        # Detect ticker from question if not provided
        if not ticker:
            ticker = self._detect_ticker(question)

        # Fetch relevant context if a stock was mentioned
        context = ""
        if ticker:
            results = self.retriever.retrieve_multi_section(
                query=question,
                ticker=ticker,
                sections=[
                    "faq",                          # pre-trained Q&A pairs (highest relevance)
                    "company_profile", "key_ratios", "income_statement",
                    "screener_profit_loss", "screener_ratios", "screener_peers",
                    "screener_balance_sheet",
                ],
                top_k_per_section=3,
            )
            context = self.retriever.format_context(results)
            logger.info(f"[InvestorQA] Retrieved {len(results)} chunks for {ticker}")
        else:
            logger.info(f"[InvestorQA] Concept question — no stock retrieval")

        user_message = build_investor_qa_query(question, context=context, ticker=ticker)
        response = self.llm.complete(
            system_prompt=INVESTOR_QA_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.3,
        )
        return response

    def _check_faq(self, question: str) -> str:
        """Return a canned FAQ answer if the question is a close match."""
        q = question.lower().strip().rstrip("?").strip()
        for key, answer in INVESTOR_FAQ.items():
            if key in q or q in key:
                return answer
        return ""

    def _detect_ticker(self, question: str) -> str:
        """
        Try to find a Nifty 50 stock in the question.
        Matches on full company name or NSE symbol.
        Returns yfinance ticker like 'RELIANCE.NS', or '' if none found.
        """
        q_lower = question.lower()
        # Longest match wins (avoids "TCS" matching inside "TATACS")
        matches: list[tuple[int, str]] = []
        for name_or_sym, symbol in _NAME_TO_SYMBOL.items():
            if name_or_sym in q_lower:
                matches.append((len(name_or_sym), symbol))
        if matches:
            matches.sort(reverse=True)
            symbol = matches[0][1]
            return f"{symbol}.NS"
        return ""
