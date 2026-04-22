# EquityLens — Deep-Dive Architecture

## Why RAG Instead of Fine-tuning

### The Misconception
Many beginners think building an AI for a domain means:
1. Collect domain data
2. Train (or fine-tune) a model on it
3. Deploy it

This works at Google or Meta. For a solo developer, it's wrong for three reasons:

| | Fine-tuning | RAG |
|---|---|---|
| **Cost** | $10,000–$1M in GPU compute | ~$0 (local embeddings) |
| **Data staleness** | Knowledge frozen at training time | Update by re-indexing tonight |
| **Debuggability** | Model is a black box | You can read every retrieved chunk |
| **Time to first result** | Weeks | Hours |
| **What it's good for** | Style/format adaptation | Grounding on fresh facts |

**The correct mental model:** Claude is already a brilliant analyst. You don't need to teach it finance — it knows it. What it lacks is the *specific numbers and events* for a given company on a given date. RAG solves exactly that.

---

## System Architecture (Detailed)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                     │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  YFinanceFetcher│  │   NewsFetcher   │  │      NSEFetcher         │  │
│  │  yfinance lib   │  │  NewsAPI + RSS  │  │  NSE public JSON API    │  │
│  │  • P&L          │  │  • Headlines    │  │  • Corporate actions    │  │
│  │  • Balance sheet│  │  • Descriptions │  │  • Shareholding pattern │  │
│  │  • Cash flows   │  │  • Full text    │  │  • FII/DII activity     │  │
│  │  • Key ratios   │  │                 │  │                         │  │
│  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │
│           │                    │                         │               │
│           ▼                    ▼                         ▼               │
│  ┌─────────────────┐  ┌─────────────────┐                               │
│  │FinancialProcessor│  │  TextProcessor  │                               │
│  │ Ratio computation│  │  Chunking       │                               │
│  │ Narrative text  │  │  Cleaning       │                               │
│  │ Structured chunks│  │  Deduplication  │                               │
│  └────────┬────────┘  └────────┬────────┘                               │
└───────────┼────────────────────┼──────────────────────────────────────── ┘
            │                    │
            └──────────┬─────────┘
                       │ List[{text, ticker, section, period}]
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          RAG LAYER                                       │
│                                                                          │
│   RAGIngestion                                                           │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  chunks ──► EmbeddingModel ──► float vectors                     │  │
│   │               (sentence-transformers, all-MiniLM-L6-v2)          │  │
│   │               Runs locally. No API. ~80MB.                       │  │
│   │                                                                  │  │
│   │  vectors + texts + metadata ──► VectorStore (ChromaDB)          │  │
│   │               Persisted to data/vector_store/                    │  │
│   │               HNSW index for fast approximate nearest neighbour  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   Retriever (at query time)                                              │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  query_text ──► embed ──► ChromaDB cosine search                 │  │
│   │  Optional: filter by {ticker, section}                           │  │
│   │  Returns: Top-K {text, metadata, similarity_score} dicts         │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────── ┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        AGENT LAYER                                       │
│                                                                          │
│   ResearchOrchestrator                                                   │
│   │                                                                      │
│   ├─► FundamentalAgent                                                   │
│   │     sections: company_profile, key_ratios, income_stmt,             │
│   │               balance_sheet, cash_flow                              │
│   │     output: financial health analysis (Markdown)                    │
│   │                                                                      │
│   ├─► SentimentAgent                                                     │
│   │     sections: news, earnings_transcript, annual_report              │
│   │     output: sentiment scorecard + themes (Markdown)                 │
│   │                                                                      │
│   ├─► ScenarioAgent                                                      │
│   │     sections: key_ratios, income_stmt, cash_flow, news              │
│   │     input: also receives FundamentalAgent output as context         │
│   │     output: Bull/Base/Bear with probability + price targets         │
│   │                                                                      │
│   └─► ConvictionAgent                                                    │
│         input: outputs of Fundamental + Scenario agents                 │
│         output: plain-English brief for retail investors                │
└───────────────────────────────────────────────────────────────────────── ┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      LLM LAYER (Claude API)                              │
│                                                                          │
│   ClaudeClient                                                           │
│   • Model: claude-sonnet-4-6                                             │
│   • Prompt caching: system prompt + context block cached                 │
│   • Cache TTL: 5 minutes (extends on each read)                          │
│   • Streaming: available for long reports                                │
│                                                                          │
│   Cost example (per full research report):                               │
│   • Without cache: ~8,000 input tokens × 4 agents ≈ 32K tokens          │
│   • With cache: 32K first call + ~3.2K subsequent = ~80% cost reduction  │
└───────────────────────────────────────────────────────────────────────── ┘
```

---

## Data Flow: One Research Report

```
User: "Research Reliance Industries for 3-year investment"
           │
           ▼
ResearchOrchestrator.research("RELIANCE.NS", horizon="3 years")
           │
    ┌──────┴──────────────────────────────────────────────┐
    │                                                      │
    ▼ Step 1                                              │
FundamentalAgent.run("RELIANCE.NS")                       │
  • Embeds query: "financial analysis ratios profitability..."
  • Retrieves 15 chunks (3 per section × 5 sections)
  • Passes to Claude with FUNDAMENTAL_SYSTEM_PROMPT
  • Output: ~800 word financial analysis
    │                                                      │
    ▼ Step 2                                              │
SentimentAgent.run("RELIANCE.NS")                         │
  • Embeds query: "news sentiment management commentary..."│
  • Retrieves 15 chunks (5 per section × 3 sections)      │
  • Output: sentiment scorecard + themes                  │
    │                                                      │
    ▼ Step 3                                              │
ScenarioAgent.run("RELIANCE.NS", fundamental_context=...) │
  • Embeds query: "growth projections risks headwinds..."  │
  • Also passes Step 1 output as context prefix          │
  • Output: Bull/Base/Bear probability table + narrative  │
    │                                                      │
    ▼ Step 4                                              │
ConvictionAgent.run(fundamental=..., scenarios=...)       │
  • NO vector search — synthesises prior outputs          │
  • Output: plain-English 500-word brief                  │
    │                                                      │
    └──────────────────────────────────────────────────────┘
           │
           ▼
ResearchReport.save() → reports/RELIANCE_NS_20260421_1430.md
```

---

## Vector Database Schema

Each document in ChromaDB has:

```
{
  "id":       "RELIANCE_NS_00042_a3f2b1c4",   // deterministic, enables upsert
  "document": "Income Statement — RELIANCE.NS — Period ending 2024-03-31:\n  Total Revenue: ₹9,67,108 Cr\n  Net Income: ₹79,020 Cr",
  "embedding": [0.023, -0.441, 0.118, ...],   // 384-dimensional float vector
  "metadata": {
    "ticker":       "RELIANCE.NS",
    "section":      "income_statement",
    "period":       "2024-03-31",
    "source":       "yfinance",
    "published_at": ""
  }
}
```

### Metadata filtering strategy
- Always filter by `ticker` to avoid cross-contamination between stocks
- Use `section` filters when you want targeted retrieval (e.g. only news for sentiment)
- The `$and` operator in ChromaDB enables compound filters

---

## Embedding Model Choice

| Model | Dim | Size | Speed | Quality | Use case |
|-------|-----|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast | Good | **Start here (Q1)** |
| `all-mpnet-base-v2` | 768 | 420MB | 2× slower | Better | Q2 upgrade |
| `BAAI/bge-small-en-v1.5` | 384 | 130MB | Fast | Better retrieval | Q2 alternative |
| `intfloat/multilingual-e5-small` | 384 | 470MB | Medium | Hindi support | Q3 if adding Hindi |

---

## Why This Architecture Scales

1. **Modular**: swap embedding model without changing retriever or agents
2. **Idempotent ingestion**: upsert by deterministic ID — safe to re-run daily
3. **Section-aware retrieval**: agents only pull relevant sections, reducing noise
4. **Prompt caching**: the large context block is cached, not re-processed each call
5. **Separation of concerns**: data pipeline, vector store, and LLM are fully decoupled
