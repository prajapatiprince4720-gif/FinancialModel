from typing import Any

from src.agents.base_agent import BaseAgent
from src.llm.prompts.sentiment_analysis import SENTIMENT_SYSTEM_PROMPT, build_sentiment_query
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAgent(BaseAgent):
    """
    Analyzes news articles and earnings call transcripts to produce
    a quantified sentiment scorecard and theme analysis.
    """

    name = "SentimentAgent"
    system_prompt = SENTIMENT_SYSTEM_PROMPT
    sections_to_retrieve = ["news", "earnings_transcript", "annual_report"]
    top_k_per_section = 5

    def run(
        self,
        ticker: str,
        company_name: str,
        text_type: str = "news",
        **kwargs: Any,
    ) -> str:
        logger.info(f"[SentimentAgent] Running for {ticker} (type={text_type})")

        query = f"news sentiment management commentary earnings outlook growth {company_name}"
        context = self._retrieve_context(query, ticker)

        user_message = build_sentiment_query(ticker, company_name, text_type)
        result = self._call_llm(user_message, context, temperature=0.2)

        logger.info(f"[SentimentAgent] Complete for {ticker}")
        return result
