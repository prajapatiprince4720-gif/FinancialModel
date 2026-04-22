"""
Groq API client — drop-in replacement for ClaudeClient.

Uses Llama 3.3 70B via Groq's free tier (no credit card needed).
Sign up at: https://console.groq.com

Same interface as ClaudeClient so all agents work without changes.
"""

from typing import Any, Iterator

from groq import Groq

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class GroqClient:

    def __init__(self) -> None:
        self._client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
        self.max_tokens = settings.claude_max_tokens

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
        temperature: float = 0.3,
        use_cache: bool = True,
    ) -> str:
        full_user = (
            f"CONTEXT (retrieved financial data):\n\n{context}\n\n---\n\nQUESTION: {user_message}"
            if context
            else user_message
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user},
            ],
            max_tokens=self.max_tokens,
            temperature=temperature,
        )
        self._log_usage(response)
        return response.choices[0].message.content

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
    ) -> Iterator[str]:
        full_user = (
            f"CONTEXT:\n{context}\n\nQUESTION: {user_message}" if context else user_message
        )
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user},
            ],
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def multi_turn(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}] + conversation
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        self._log_usage(response)
        return response.choices[0].message.content

    def _log_usage(self, response: Any) -> None:
        usage = response.usage
        logger.info(
            f"Tokens — input: {usage.prompt_tokens}, output: {usage.completion_tokens}"
        )
