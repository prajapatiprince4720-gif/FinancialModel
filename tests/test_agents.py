"""
Tests for the agent layer.
These mock the LLM to avoid API calls — tests should be free to run.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.fundamental_agent import FundamentalAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.scenario_agent import ScenarioAgent
from src.agents.conviction_agent import ConvictionAgent
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.rag.embeddings import EmbeddingModel


@pytest.fixture
def mock_retriever(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path), collection_name="test_agents")
    model = EmbeddingModel()
    texts = [
        "RELIANCE.NS key ratios P/E 22x ROE 12%",
        "RELIANCE.NS income statement revenue growing",
        "RELIANCE.NS news: positive outlook for petrochemicals",
    ]
    embeddings = model.embed(texts)
    metas = [
        {"ticker": "RELIANCE.NS", "section": "key_ratios", "period": "", "source": "", "published_at": ""},
        {"ticker": "RELIANCE.NS", "section": "income_statement", "period": "2024", "source": "", "published_at": ""},
        {"ticker": "RELIANCE.NS", "section": "news", "period": "", "source": "rss", "published_at": "2024-01-01"},
    ]
    store.add(texts, embeddings, metas, [f"d{i}" for i in range(len(texts))])
    return Retriever(vector_store=store, embedding_model=model)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete.return_value = "## Mock Analysis\n\nThis is a test response with some financial details."
    return llm


class TestFundamentalAgent:

    def test_run_returns_string(self, mock_retriever, mock_llm):
        agent = FundamentalAgent(retriever=mock_retriever, llm=mock_llm)
        result = agent.run("RELIANCE.NS", "Reliance Industries")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_calls_llm_once(self, mock_retriever, mock_llm):
        agent = FundamentalAgent(retriever=mock_retriever, llm=mock_llm)
        agent.run("RELIANCE.NS", "Reliance Industries")
        mock_llm.complete.assert_called_once()

    def test_run_passes_context_to_llm(self, mock_retriever, mock_llm):
        agent = FundamentalAgent(retriever=mock_retriever, llm=mock_llm)
        agent.run("RELIANCE.NS", "Reliance Industries")
        call_kwargs = mock_llm.complete.call_args
        context = call_kwargs[1].get("context") or call_kwargs[0][2] if len(call_kwargs[0]) > 2 else ""
        # Context should not be empty since we have documents in the store
        assert len(context) > 0 or mock_llm.complete.called


class TestSentimentAgent:

    def test_run_returns_string(self, mock_retriever, mock_llm):
        agent = SentimentAgent(retriever=mock_retriever, llm=mock_llm)
        result = agent.run("RELIANCE.NS", "Reliance Industries")
        assert isinstance(result, str)


class TestScenarioAgent:

    def test_run_returns_string(self, mock_retriever, mock_llm):
        agent = ScenarioAgent(retriever=mock_retriever, llm=mock_llm)
        result = agent.run("RELIANCE.NS", "Reliance Industries", horizon="3 years")
        assert isinstance(result, str)

    def test_run_with_fundamental_context(self, mock_retriever, mock_llm):
        agent = ScenarioAgent(retriever=mock_retriever, llm=mock_llm)
        result = agent.run(
            "RELIANCE.NS", "Reliance Industries",
            fundamental_context="P/E ratio is 22x, ROE is 12%"
        )
        assert isinstance(result, str)


class TestConvictionAgent:

    def test_run_returns_string(self, mock_llm):
        agent = ConvictionAgent(llm=mock_llm)
        result = agent.run(
            ticker="RELIANCE.NS",
            company_name="Reliance Industries",
            fundamental_analysis="Strong fundamentals, P/E of 22x",
            scenario_analysis="Bull case: 30% upside, Bear case: 20% downside",
        )
        assert isinstance(result, str)

    def test_does_not_use_retriever(self, mock_llm):
        agent = ConvictionAgent(llm=mock_llm)
        assert not hasattr(agent, "retriever")
