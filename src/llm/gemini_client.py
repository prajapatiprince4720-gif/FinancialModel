"""
Google Gemini API client — drop-in replacement for GroqClient/ClaudeClient.

Uses gemini-2.0-flash via Google AI Studio free tier (1,500 req/day, no credit card).
Get your key at: https://aistudio.google.com

Same interface as GroqClient so all agents work without changes.
"""

from typing import Iterator

import google.generativeai as genai

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class GeminiClient:

    def __init__(self) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_model
        self.max_tokens = 4096

    def _model(self, system_prompt: str):
        return genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=0.3,
            ),
        )

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
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=temperature,
            ),
        )
        response = model.generate_content(full_user)
        logger.info(f"Gemini tokens — input: {response.usage_metadata.prompt_token_count}, "
                    f"output: {response.usage_metadata.candidates_token_count}")
        return response.text

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
    ) -> Iterator[str]:
        full_user = (
            f"CONTEXT:\n{context}\n\nQUESTION: {user_message}" if context else user_message
        )
        model = self._model(system_prompt)
        response = model.generate_content(full_user, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def multi_turn(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
    ) -> str:
        model = self._model(system_prompt)
        # Convert conversation to Gemini format
        history = []
        for msg in conversation[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        chat = model.start_chat(history=history)
        response = chat.send_message(conversation[-1]["content"])
        logger.info(f"Gemini tokens — input: {response.usage_metadata.prompt_token_count}, "
                    f"output: {response.usage_metadata.candidates_token_count}")
        return response.text
