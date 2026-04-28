import os
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Anthropic
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 4096

    # LLM provider — "gemini" | "groq" | "claude"
    llm_provider: str = "gemini"
    groq_api_key: str = Field(default="", description="Groq API key (free at console.groq.com)")
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = Field(default="", description="Google Gemini key (free at aistudio.google.com)")
    gemini_model: str = "gemini-2.0-flash"

    # News (optional — falls back to free RSS feeds when blank)
    news_api_key: str = Field(default="", description="NewsAPI.org key")
    alpha_vantage_api_key: str = Field(default="", description="Alpha Vantage key")

    # Vector DB
    chroma_persist_dir: str = "data/vector_store"
    chroma_collection_name: str = "equity_research"

    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # App
    log_level: str = "INFO"
    environment: str = "development"
    stock_universe: str = "NIFTY50"

    # Embedding model (runs locally — no API key needed)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_chunk_size: int = 512
    embedding_chunk_overlap: int = 50

    # RAG retrieval
    retrieval_top_k: int = 8

    @field_validator("news_api_key", "alpha_vantage_api_key", "groq_api_key", mode="before")
    @classmethod
    def _strip_placeholders(cls, v: str) -> str:
        """Treat unfilled placeholder values as empty so optional features skip gracefully."""
        if not v:
            return ""
        v = str(v).strip()
        if v.startswith("your_") or v.endswith("_here") or v in ("sk-ant-...",):
            return ""
        return v

    @model_validator(mode="after")
    def _ensure_data_dirs(self) -> "Settings":
        """Create required data directories on first import so fetchers never hit FileNotFoundError."""
        for path in (
            self.processed_data_dir,
            os.path.join(self.raw_data_dir, "financials"),
            os.path.join(self.raw_data_dir, "news"),
            self.chroma_persist_dir,
            "reports",
        ):
            os.makedirs(path, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
