from src.llm.claude_client import ClaudeClient
from src.llm.groq_client import GroqClient
from src.llm.gemini_client import GeminiClient


def get_llm_client():
    """Return the configured LLM client based on LLM_PROVIDER in .env (groq/gemini/claude)."""
    from config.settings import get_settings
    provider = get_settings().llm_provider.lower()
    if provider == "gemini":
        return GeminiClient()
    if provider == "groq":
        return GroqClient()
    return ClaudeClient()


__all__ = ["ClaudeClient", "GroqClient", "GeminiClient", "get_llm_client"]
