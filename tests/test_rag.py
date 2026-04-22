"""
Tests for the RAG layer (embeddings, vector store, retriever).
These run locally — no API keys required.
"""

import pytest
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever


class TestEmbeddingModel:

    def test_embed_returns_correct_shape(self):
        model = EmbeddingModel()
        texts = ["Reliance revenue grew 12%", "HDFC Bank net interest margin 4.3%"]
        embeddings = model.embed(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == model.dimension

    def test_embed_one(self):
        model = EmbeddingModel()
        vec = model.embed_one("TCS quarterly results beat estimates")
        assert isinstance(vec, list)
        assert len(vec) > 0

    def test_empty_input(self):
        model = EmbeddingModel()
        result = model.embed([])
        assert result == []

    def test_similar_texts_closer_than_dissimilar(self):
        import numpy as np
        model = EmbeddingModel()

        v1 = np.array(model.embed_one("HDFC Bank net profit grew 20%"))
        v2 = np.array(model.embed_one("HDFC Bank earnings increased significantly"))
        v3 = np.array(model.embed_one("Monsoon season expected to be normal this year"))

        sim_12 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        sim_13 = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))

        # Financial texts should be more similar to each other than to unrelated text
        assert sim_12 > sim_13


class TestVectorStore:

    @pytest.fixture
    def store(self, tmp_path):
        return VectorStore(persist_dir=str(tmp_path), collection_name="test_collection")

    def test_add_and_count(self, store):
        texts = ["Reliance Q4 profit ₹20,000 Cr", "HDFC Bank NIM improved to 4.3%"]
        model = EmbeddingModel()
        embeddings = model.embed(texts)
        metadatas = [{"ticker": "RELIANCE.NS", "section": "income_statement", "period": "", "source": "", "published_at": ""}, {"ticker": "HDFCBANK.NS", "section": "key_ratios", "period": "", "source": "", "published_at": ""}]
        ids = ["doc_0", "doc_1"]
        store.add(texts, embeddings, metadatas, ids)
        assert store.count() == 2

    def test_query_returns_results(self, store):
        texts = ["Reliance Q4 profit ₹20,000 Cr"]
        model = EmbeddingModel()
        embeddings = model.embed(texts)
        store.add(texts, embeddings, [{"ticker": "RELIANCE.NS", "section": "income_statement", "period": "", "source": "", "published_at": ""}], ["doc_0"])

        query_vec = model.embed_one("What is Reliance profit?")
        results = store.query(query_vec, top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == texts[0]

    def test_delete_ticker(self, store):
        texts = ["Reliance profit", "TCS revenue"]
        model = EmbeddingModel()
        embeddings = model.embed(texts)
        metas = [
            {"ticker": "RELIANCE.NS", "section": "income_statement", "period": "", "source": "", "published_at": ""},
            {"ticker": "TCS.NS", "section": "income_statement", "period": "", "source": "", "published_at": ""},
        ]
        store.add(texts, embeddings, metas, ["r_0", "t_0"])
        assert store.count() == 2
        store.delete_ticker("RELIANCE.NS")
        assert store.count() == 1

    def test_metadata_filter(self, store):
        texts = ["Reliance news article today", "Reliance annual income statement FY24"]
        model = EmbeddingModel()
        embeddings = model.embed(texts)
        metas = [
            {"ticker": "RELIANCE.NS", "section": "news", "period": "", "source": "", "published_at": ""},
            {"ticker": "RELIANCE.NS", "section": "income_statement", "period": "", "source": "", "published_at": ""},
        ]
        store.add(texts, embeddings, metas, ["n_0", "i_0"])

        query_vec = model.embed_one("Reliance financial data")
        results = store.query(query_vec, top_k=5, where={"section": "news"})
        assert all(r["metadata"]["section"] == "news" for r in results)


class TestRetriever:

    @pytest.fixture
    def retriever(self, tmp_path):
        store = VectorStore(persist_dir=str(tmp_path), collection_name="test_retriever")
        model = EmbeddingModel()
        texts = [
            "RELIANCE.NS income statement FY24 revenue ₹9.67 lakh crore",
            "RELIANCE.NS balance sheet total debt ₹3.2 lakh crore",
            "RELIANCE.NS news: Jio launches new 5G plans",
            "TCS.NS income statement FY24 revenue ₹2.4 lakh crore",
        ]
        embeddings = model.embed(texts)
        metas = [
            {"ticker": "RELIANCE.NS", "section": "income_statement", "period": "2024", "source": "", "published_at": ""},
            {"ticker": "RELIANCE.NS", "section": "balance_sheet", "period": "2024", "source": "", "published_at": ""},
            {"ticker": "RELIANCE.NS", "section": "news", "period": "", "source": "rss", "published_at": "2024-04-01"},
            {"ticker": "TCS.NS", "section": "income_statement", "period": "2024", "source": "", "published_at": ""},
        ]
        ids = [f"doc_{i}" for i in range(len(texts))]
        store.add(texts, embeddings, metas, ids)
        return Retriever(vector_store=store, embedding_model=model)

    def test_retrieve_filters_by_ticker(self, retriever):
        results = retriever.retrieve("revenue", ticker="TCS.NS")
        assert all(r["metadata"]["ticker"] == "TCS.NS" for r in results)

    def test_retrieve_returns_scores(self, retriever):
        results = retriever.retrieve("Reliance revenue", ticker="RELIANCE.NS")
        assert all("score" in r for r in results)
        assert all(0 <= r["score"] <= 1 for r in results)

    def test_format_context_not_empty(self, retriever):
        results = retriever.retrieve("financial data", ticker="RELIANCE.NS")
        ctx = retriever.format_context(results)
        assert len(ctx) > 0
        assert "RELIANCE.NS" in ctx
