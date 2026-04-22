from typing import Any

from src.agents.base_agent import BaseAgent
from src.llm.prompts.fundamental_analysis import FUNDAMENTAL_SYSTEM_PROMPT, build_fundamental_query
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FundamentalAgent(BaseAgent):
    """
    Analyzes financial ratios, income statements, balance sheet, and cash flows.
    Produces an institutional-grade fundamental analysis.
    """

    name = "FundamentalAgent"
    system_prompt = FUNDAMENTAL_SYSTEM_PROMPT
    sections_to_retrieve = [
        "company_profile",
        "key_ratios",
        "income_statement",
        "balance_sheet",
        "cash_flow",
        "screener_profit_loss",
        "screener_balance_sheet",
        "screener_cash_flow",
        "screener_ratios",
        "screener_peers",
    ]
    top_k_per_section = 5

    def run(self, ticker: str, company_name: str, question: str = "", **kwargs: Any) -> str:
        logger.info(f"[FundamentalAgent] Running for {ticker}")

        query = f"financial analysis fundamentals ratios profitability debt cash flow {company_name}"
        context = self._retrieve_context(query, ticker)

        user_message = build_fundamental_query(ticker, company_name, question)
        result = self._call_llm(user_message, context, temperature=0.2)

        logger.info(f"[FundamentalAgent] Complete for {ticker}")
        return result
