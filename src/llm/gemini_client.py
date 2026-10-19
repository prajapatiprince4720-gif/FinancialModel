"""
Google Gemini API client — drop-in replacement for GroqClient/ClaudeClient.

Uses a Gemini Flash model via Google AI Studio's free tier (no credit card).
Get a key at: https://aistudio.google.com

Built on the `google-genai` SDK. The older `google-generativeai` package this
client used previously is end-of-life and no longer receives fixes.

Google retires Gemini model ids on a rolling basis — `gemini-2.0-flash`, which
this project pinned, now 404s outright. A pinned id that disappears would kill
an unattended overnight training run, so a 404 falls forward to the evergreen
`gemini-flash-latest` alias and logs loudly instead of aborting.

Same interface as GroqClient so every agent works without changes.
"""

from typing import Iterator, Optional

from google import genai
from google.genai import types

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# FAQ training sends 15 questions per call and expects 3-6 sentences each, which
# overruns a 4k budget and gets the tail of the batch silently truncated. Flash
# supports 8k output, so ask for it.
_MAX_OUTPUT_TOKENS = 8192

# Evergreen alias used when the configured model id has been retired.
_FALLBACK_MODEL = "gemini-flash-latest"


class GeminiClient:

    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env or switch LLM_PROVIDER."
            )
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_model
        self.max_tokens = _MAX_OUTPUT_TOKENS

    # ── Internals ─────────────────────────────────────────────────────────────

    def _is_retired_model(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return "404" in msg and ("no longer available" in msg or "not_found" in msg)

    def _fall_forward(self, exc: Exception) -> bool:
        """
        Switch to the evergreen alias after a retired-model 404.
        Returns True if the caller should retry, False if we are already on it.
        """
        if self.model_name == _FALLBACK_MODEL:
            return False
        logger.warning(
            f"Gemini model '{self.model_name}' has been retired by Google "
            f"({str(exc)[:120]}). Falling forward to '{_FALLBACK_MODEL}'. "
            f"Update GEMINI_MODEL in .env to pin a current id."
        )
        self.model_name = _FALLBACK_MODEL
        return True

    def _config(self, system_prompt: str, temperature: float) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            max_output_tokens=self.max_tokens,
            temperature=temperature,
        )

    @staticmethod
    def _merge(user_message: str, context: str) -> str:
        if not context:
            return user_message
        return (
            f"CONTEXT (retrieved financial data):\n\n{context}\n\n"
            f"---\n\nQUESTION: {user_message}"
        )

    @staticmethod
    def _log_usage(response) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            logger.info(
                f"Gemini tokens — input: {getattr(usage, 'prompt_token_count', '?')}, "
                f"output: {getattr(usage, 'candidates_token_count', '?')}"
            )

    @staticmethod
    def _text(response) -> str:
        """
        Extract text, tolerating a response that carries no usable candidate.

        `response.text` raises when the model returned only a safety block or an
        empty candidate, which would otherwise surface as an opaque crash in the
        middle of a long training run.
        """
        text: Optional[str] = None
        try:
            text = response.text
        except Exception as exc:
            logger.warning(f"Gemini returned no usable text: {exc}")
        if text:
            return text

        reason = None
        for candidate in (getattr(response, "candidates", None) or []):
            reason = getattr(candidate, "finish_reason", None) or reason
        raise RuntimeError(f"Gemini returned an empty response (finish_reason={reason})")

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
        temperature: float = 0.3,
        use_cache: bool = True,
    ) -> str:
        contents = self._merge(user_message, context)
        config   = self._config(system_prompt, temperature)
        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config,
            )
        except Exception as exc:
            if not (self._is_retired_model(exc) and self._fall_forward(exc)):
                raise
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config,
            )
        self._log_usage(response)
        return self._text(response)

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
    ) -> Iterator[str]:
        stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=self._merge(user_message, context),
            config=self._config(system_prompt, 0.3),
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def multi_turn(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
    ) -> str:
        if not conversation:
            raise ValueError("multi_turn requires at least one message")

        history = [
            types.Content(
                role="user" if msg.get("role") == "user" else "model",
                parts=[types.Part(text=msg.get("content", ""))],
            )
            for msg in conversation[:-1]
        ]
        chat = self.client.chats.create(
            model=self.model_name,
            config=self._config(system_prompt, 0.3),
            history=history,
        )
        response = chat.send_message(conversation[-1].get("content", ""))
        self._log_usage(response)
        return self._text(response)
