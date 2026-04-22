from typing import Any

from src.agents.base_agent import BaseAgent
from src.llm.prompts.scenario_builder import SCENARIO_SYSTEM_PROMPT, build_scenario_query
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScenarioAgent(BaseAgent):
    """
    Builds probabilistic Bull / Base / Bear scenarios with price targets
    for 1-year and 3-year horizons.
    """

    name = "ScenarioAgent"
    system_prompt = SCENARIO_SYSTEM_PROMPT
    sections_to_retrieve = [
        "key_ratios",
        "income_statement",
        "cash_flow",
        "news",
        "screener_profit_loss",
        "screener_cash_flow",
        "screener_ratios",
    ]
    top_k_per_section = 5

    def run(
        self,
        ticker: str,
        company_name: str,
        horizon: str = "3 years",
        fundamental_context: str = "",
        **kwargs: Any,
    ) -> str:
        logger.info(f"[ScenarioAgent] Running for {ticker} (horizon={horizon})")

        query = f"growth projections revenue earnings expansion risks headwinds {company_name}"
        context = self._retrieve_context(query, ticker)

        if fundamental_context:
            context = f"=== FUNDAMENTAL ANALYSIS (from prior agent) ===\n{fundamental_context}\n\n=== RETRIEVED DATA ===\n{context}"

        user_message = build_scenario_query(ticker, company_name, horizon)
        result = self._call_llm(user_message, context, temperature=0.4)

        logger.info(f"[ScenarioAgent] Complete for {ticker}")
        return result
