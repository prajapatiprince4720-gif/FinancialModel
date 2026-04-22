from src.llm.claude_client import ClaudeClient
from src.llm.groq_client import GroqClient


def get_llm_client():
    """Return the configured LLM client (groq or claude) based on LLM_PROVIDER in .env."""
    from config.settings import get_settings
    provider = get_settings().llm_provider.lower()
    if provider == "groq":
        return GroqClient()
    return ClaudeClient()


__all__ = ["ClaudeClient", "GroqClient", "get_llm_client"]
