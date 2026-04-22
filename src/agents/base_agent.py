"""
Abstract base class for all research agents.

Every agent follows the same pattern:
  1. Retrieve relevant context from the vector store
  2. Build a prompt with that context
  3. Call Claude
  4. Return structured output

Subclasses override: system_prompt, sections_to_retrieve, run()
"""

from abc import ABC, abstractmethod
from typing import Any

from src.llm import get_llm_client
from src.rag.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):

    # Subclasses set these
    name: str = "BaseAgent"
    system_prompt: str = ""
    sections_to_retrieve: list[str] = []
    top_k_per_section: int = 4

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: Any | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or get_llm_client()

    @abstractmethod
    def run(self, ticker: str, company_name: str, **kwargs: Any) -> str:
        """Execute the agent and return a Markdown-formatted analysis string."""
        ...

    def _retrieve_context(self, query: str, ticker: str) -> str:
        """Retrieve relevant chunks and format them as a context string."""
        if self.sections_to_retrieve:
            results = self.retriever.retrieve_multi_section(
                query=query,
                ticker=ticker,
                sections=self.sections_to_retrieve,
                top_k_per_section=self.top_k_per_section,
            )
        else:
            results = self.retriever.retrieve(query=query, ticker=ticker)

        context = self.retriever.format_context(results)
        logger.debug(f"[{self.name}] Retrieved {len(results)} chunks for '{ticker}'")
        return context

    def _call_llm(self, user_message: str, context: str, temperature: float = 0.3) -> str:
        return self.llm.complete(
            system_prompt=self.system_prompt,
            user_message=user_message,
            context=context,
            temperature=temperature,
        )
