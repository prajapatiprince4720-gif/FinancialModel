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
from src.rag.financial_facts import FinancialFacts
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
        self.facts = FinancialFacts()

    def ask(self, question: str, ticker: str = "") -> str:
        """
        Answer an investor question.

        Args:
            question: The investor's question in plain English
            ticker:   Optional yfinance ticker (e.g. 'RELIANCE.NS') for stock-specific questions.
                      If not provided, the agent tries to detect the stock from the question.
        """
        return self.ask_detailed(question, ticker)["answer"]

    def ask_detailed(self, question: str, ticker: str = "") -> dict[str, Any]:
        """
        Answer a question and report how the answer was produced.

        Order of preference, cheapest and most reliable first:
          1. "concept"  — canned explanation of a general concept (no stock named)
          2. "facts"    — a lookup answered from the company's fact sheet, no LLM
          3. "llm"      — LLM reasoning, with the fact sheet and exact Screener
                          figures as its primary context
        """
        if not ticker:
            ticker = self._detect_ticker(question)
        if not ticker:
            symbols = self.facts.detect_symbols(question)
            ticker = f"{symbols[0]}.NS" if symbols else ""

        # Canned answers only for generic concept questions, never stock-specific ones
        if not ticker:
            faq_answer = self._check_faq(question)
            if faq_answer:
                return {"answer": faq_answer, "source": "concept", "symbol": None}

        symbol = ticker.replace(".NS", "").replace(".BO", "") if ticker else ""

        sheet = None
        if symbol:
            try:
                from src.services.research_service import get_service
                sheet = get_service().fact_sheet(symbol)
            except Exception as exc:
                logger.warning(f"[InvestorQA] fact sheet unavailable for {symbol}: {exc}")

        if sheet is not None:
            from src.facts.answerer import answer_from_facts
            direct = answer_from_facts(question, sheet)
            if direct is not None:
                logger.info(f"[InvestorQA] answered from fact sheet ({direct.metric})")
                return {"answer": direct.text, "source": "facts", "symbol": symbol}

        context = ""
        if sheet is not None:
            context = sheet.to_context()
        if ticker:
            # Raw statement chunks and news supplement the fact sheet. The old
            # LLM-pre-written "faq" section is deliberately not retrieved: those
            # answers go stale on every data refresh and could carry invented
            # numbers back in as if they were facts.
            results = self.retriever.retrieve_multi_section(
                query=question,
                ticker=ticker,
                sections=[
                    "company_profile", "news",
                    "screener_profit_loss", "screener_balance_sheet",
                    "screener_cash_flow", "screener_ratios",
                ],
                top_k_per_section=2,
            )
            if results:
                context += "\n\nSUPPORTING DOCUMENTS (lower priority than the fact sheet):\n"
                context += self.retriever.format_context(results)
            logger.info(f"[InvestorQA] Retrieved {len(results)} chunks for {ticker}")
        else:
            logger.info("[InvestorQA] Concept question — no stock retrieval")

        # Exact per-year figures straight from Screener data — search is unreliable for specific years
        facts = self.facts.for_question(question, ticker=ticker)
        if facts:
            context = f"{facts}\n\nBACKGROUND (figures above take precedence):\n{context}"

        user_message = build_investor_qa_query(question, context=context, ticker=ticker)
        response = self.llm.complete(
            system_prompt=INVESTOR_QA_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.3,
        )
        return {"answer": response, "source": "llm", "symbol": symbol or None}

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
        # Whole-word matches only: a bare substring test sent "result" to LT and
        # "below" to BEL. Longest match wins ("hdfc bank" over a shorter alias).
        matches: list[tuple[int, str]] = []
        for name_or_sym, symbol in _NAME_TO_SYMBOL.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(name_or_sym)}(?![a-z0-9])", q_lower):
                matches.append((len(name_or_sym), symbol))
        if matches:
            matches.sort(reverse=True)
            symbol = matches[0][1]
            return f"{symbol}.NS"
        return ""
