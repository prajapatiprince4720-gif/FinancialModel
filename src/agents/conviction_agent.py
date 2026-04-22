from typing import Any

from src.llm.claude_client import ClaudeClient
from src.llm.prompts.conviction_summary import CONVICTION_SYSTEM_PROMPT, build_conviction_query
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConvictionAgent:
    """
    Takes the outputs of FundamentalAgent and ScenarioAgent and produces
    a plain-English conviction summary for retail investors.

    This agent does NOT query the vector store — it synthesizes prior outputs.
    """

    name = "ConvictionAgent"

    def __init__(self, llm: ClaudeClient | None = None) -> None:
        self.llm = llm or ClaudeClient()

    def run(
        self,
        ticker: str,
        company_name: str,
        fundamental_analysis: str,
        scenario_analysis: str,
        **kwargs: Any,
    ) -> str:
        logger.info(f"[ConvictionAgent] Running for {ticker}")

        user_message = build_conviction_query(
            ticker, company_name, fundamental_analysis, scenario_analysis
        )
        result = self.llm.complete(
            system_prompt=CONVICTION_SYSTEM_PROMPT,
            user_message=user_message,
            context="",
            temperature=0.5,
        )
        logger.info(f"[ConvictionAgent] Complete for {ticker}")
        return result
