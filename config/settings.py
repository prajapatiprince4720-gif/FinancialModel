from functools import lru_cache
from pydantic import Field
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

    # News
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
