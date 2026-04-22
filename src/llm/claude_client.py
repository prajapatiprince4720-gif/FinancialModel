"""
Anthropic Claude API wrapper with prompt caching.

Prompt caching is critical for this project:
  - System prompts and large context windows are reused across calls
  - Caching cuts token costs by ~90% on repeated queries for the same stock
  - Saves API credits as you iterate day after day

How caching works:
  1. Mark large, stable content with cache_control: {"type": "ephemeral"}
  2. Anthropic caches it for 5 minutes (extendable)
  3. Subsequent calls that hit the cache pay ~10% of normal input cost

Official docs: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""

from typing import Any, Iterator

import anthropic

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ClaudeClient:

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model
        self.max_tokens = settings.claude_max_tokens

    # ──────────────────────────────────────────────────────────────────────────
    # Core call
    # ──────────────────────────────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
        temperature: float = 0.3,
        use_cache: bool = True,
    ) -> str:
        """
        Single-turn completion with optional prompt caching.

        Args:
            system_prompt: Role + instructions for the agent
            user_message:  The user's question
            context:       Retrieved RAG chunks (large, benefits most from caching)
            temperature:   0.0 = deterministic, 1.0 = creative
            use_cache:     Whether to use Anthropic's prompt caching

        Returns:
            Model's text response
        """
        messages = self._build_messages(user_message, context, use_cache)

        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system_prompt}]
        if use_cache and len(system_prompt) > 1024:
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system_blocks,
            messages=messages,
        )

        self._log_usage(response)
        return response.content[0].text

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
    ) -> Iterator[str]:
        """Stream response tokens — useful for long reports in a CLI."""
        messages = self._build_messages(user_message, context, use_cache=True)
        system_blocks = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def multi_turn(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
    ) -> str:
        """
        Multi-turn conversation. conversation = [{"role": "user", "content": "..."}, ...]
        Useful for interactive research sessions.
        """
        system_blocks = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
            messages=conversation,
        )
        self._log_usage(response)
        return response.content[0].text

    # ──────────────────────────────────────────────────────────────────────────

    def _build_messages(
        self, user_message: str, context: str, use_cache: bool
    ) -> list[dict[str, Any]]:
        if context and use_cache:
            # Cache the large context block separately from the question
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"CONTEXT (retrieved financial data):\n\n{context}\n\n---",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": f"\nQUESTION: {user_message}"},
                    ],
                }
            ]
        full_text = f"CONTEXT:\n{context}\n\nQUESTION: {user_message}" if context else user_message
        return [{"role": "user", "content": full_text}]

    def _log_usage(self, response: Any) -> None:
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0
        logger.info(
            f"Tokens — input: {usage.input_tokens}, output: {usage.output_tokens}, "
            f"cache_read: {cache_read}, cache_created: {cache_created}"
        )
