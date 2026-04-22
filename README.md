# EquityLens — AI-Driven Equity Research for Indian Retail Investors

> Institutional-grade analysis of Nifty 50 stocks, powered by RAG + Agentic AI.  
> Built for users of Zerodha, Groww, and other Indian brokerages.

---

## The Most Important Thing to Understand First

**You do NOT need to train a model from scratch.** This is the #1 misconception among beginners.

Training a large language model from scratch requires:
- Millions of dollars in GPU compute
- Terabytes of curated training data
- A team of ML researchers

Instead, this project uses two modern techniques that let a single developer build something far more powerful:

### Retrieval-Augmented Generation (RAG)

Think of RAG as giving a genius (Claude) a perfectly organized filing cabinet (your vector database) to consult before answering. Instead of "training" it on financial data, you:

1. **Collect** financial documents (annual reports, news, earnings call transcripts)
2. **Chunk** them into small pieces and convert to vector embeddings (numerical representations)
3. **Store** embeddings in a vector database (ChromaDB)
4. At query time, **retrieve** the 5–10 most relevant chunks for the user's question
5. **Pass** those chunks as context to Claude, which synthesizes a research report

The model's weights never change. You are just giving it better, fresher, domain-specific context.

```
User Query → Embedding → Vector DB Search → Top-K Chunks → Claude → Research Report
```

### Agentic Workflows

Instead of one monolithic prompt, you build specialized **agents** that collaborate:

```
Orchestrator Agent
    ├── Fundamental Agent   → Analyzes PE, debt, cash flow ratios
    ├── Sentiment Agent     → Analyzes news and earnings call tone
    ├── Scenario Agent      → Builds Bull / Base / Bear cases for 1–5 years
    └── Conviction Agent    → Writes plain-English summary for retail investor
```

Each agent has its own tools (fetch data, query vector DB, calculate ratios) and its own focused prompt. The orchestrator routes the query and assembles the final report.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│  yfinance ──► Financial Processor ──► Structured JSON           │
│  NewsAPI  ──► Text Processor      ──► Cleaned Chunks            │
│  NSE/BSE  ──► NSE Fetcher         ──► Corporate Actions         │
└───────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  RAG INGESTION  │
                    │  Chunking       │
                    │  Embedding      │  ← sentence-transformers
                    │  ChromaDB       │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │       AGENT LAYER           │
              │  Orchestrator               │
              │  ├── Fundamental Agent      │
              │  ├── Sentiment Agent        │
              │  ├── Scenario Agent         │
              │  └── Conviction Agent       │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │   CLAUDE API    │  ← Anthropic claude-sonnet-4-6
                    │  Prompt Cache   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   OUTPUT        │
                    │  Research PDF   │
                    │  JSON Report    │
                    │  (REST API Q3)  │
                    └─────────────────┘
```

---

## Technical Approaches — Choose Your Path

Before you start coding, consider these three approaches and their trade-offs:

| Approach | Description | Pros | Cons | Best for |
|----------|-------------|------|------|----------|
| **A: LangChain** | High-level framework with prebuilt chains, retrievers, agents | Fast prototyping, huge ecosystem, lots of tutorials | Magic abstractions hide what's happening, hard to debug | Q1 learning |
| **B: LlamaIndex** | Document-centric RAG framework, excellent for ingestion pipelines | Best-in-class document parsing, good eval tools | Steeper learning curve than LangChain | Q2 production RAG |
| **C: Raw Anthropic SDK** | Direct API calls, manual RAG plumbing | Full control, cheapest, easiest to debug | More boilerplate | Understanding fundamentals |

**Recommendation**: Start with Approach C for the first 4–6 weeks to deeply understand RAG. Then layer LangChain on top for agent orchestration.

---

## Repository Structure

```
FinancialModel/
├── .env.example              # Copy to .env and fill in API keys
├── .gitignore
├── requirements.txt
├── README.md
│
├── config/
│   ├── __init__.py
│   ├── settings.py           # Pydantic settings (reads from .env)
│   └── nifty50_tickers.py    # Master list of all 50 tickers
│
├── data/                     # Gitignored — raw/processed data lives here
│   ├── raw/
│   │   ├── financials/       # yfinance JSON dumps
│   │   ├── news/             # NewsAPI JSON dumps
│   │   └── transcripts/      # Earnings call transcripts (manually sourced)
│   ├── processed/            # Cleaned, chunked text
│   └── vector_store/         # ChromaDB persistent store
│
├── src/
│   ├── data_pipeline/
│   │   ├── fetchers/
│   │   │   ├── yfinance_fetcher.py   # Fundamentals via yfinance
│   │   │   ├── news_fetcher.py       # News via NewsAPI / RSS
│   │   │   └── nse_fetcher.py        # NSE-specific: corporate actions, FII data
│   │   ├── processors/
│   │   │   ├── financial_processor.py  # Ratio computation, normalization
│   │   │   └── text_processor.py       # Chunking, cleaning, deduplication
│   │   └── pipeline.py               # Orchestrates full ingestion run
│   │
│   ├── rag/
│   │   ├── embeddings.py      # Embedding model wrapper
│   │   ├── vector_store.py    # ChromaDB client
│   │   ├── retriever.py       # Hybrid retrieval (dense + keyword)
│   │   └── ingestion.py       # End-to-end: fetch → chunk → embed → store
│   │
│   ├── agents/
│   │   ├── base_agent.py          # Abstract base with tool-use interface
│   │   ├── fundamental_agent.py   # Analyzes financial ratios
│   │   ├── sentiment_agent.py     # Analyzes news & transcript tone
│   │   ├── scenario_agent.py      # Builds 1–5 year scenarios
│   │   └── orchestrator.py        # Routes and assembles final report
│   │
│   ├── llm/
│   │   ├── claude_client.py       # Anthropic SDK wrapper with prompt caching
│   │   └── prompts/
│   │       ├── fundamental_analysis.py
│   │       ├── scenario_builder.py
│   │       └── conviction_summary.py
│   │
│   └── utils/
│       ├── logger.py
│       └── helpers.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Explore yfinance data for one ticker
│   ├── 02_rag_prototype.ipynb      # Build minimal RAG pipeline end-to-end
│   └── 03_agent_testing.ipynb      # Test individual agents interactively
│
├── tests/
│   ├── test_fetchers.py
│   ├── test_rag.py
│   └── test_agents.py
│
└── docs/
    ├── ARCHITECTURE.md        # Deep-dive system design
    └── QUARTERLY_ROADMAP.md   # Detailed 4-quarter plan
```

---

## Data Sources (Free / Cheap)

| Source | What you get | Cost | Library |
|--------|-------------|------|---------|
| **yfinance** | Income statement, balance sheet, cash flow, PE/PB/EPS, dividends | Free | `yfinance` |
| **NSE India** | EOD prices, FII/DII data, corporate actions | Free (scrape) | `requests` + `bs4` |
| **NewsAPI.org** | 100 req/day free, news headlines + snippets | Free tier | `requests` |
| **Alpha Vantage** | Fundamentals + news sentiment | Free (25 req/day) | `requests` |
| **MoneyControl / Screener.in** | Qualitative commentary (scrape responsibly) | Free | `bs4` |
| **NSE Annual Reports** | Full PDFs of annual reports | Free (PDF download) | `pypdf` |
| **Concall.in / Earnings Whispers** | Earnings call transcripts | Free (some paywalled) | Manual + `requests` |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/prajapatiprince4720-gif/FinancialModel.git
cd FinancialModel

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Run the data pipeline for a single ticker
python -m src.data_pipeline.pipeline --ticker RELIANCE.NS

# 6. Run the full research report
python -m src.agents.orchestrator --ticker RELIANCE.NS --query "Investment thesis for next 3 years"
```

---

## Quarter-by-Quarter Roadmap

See [docs/QUARTERLY_ROADMAP.md](docs/QUARTERLY_ROADMAP.md) for the full plan.

| Quarter | Focus | Deliverable |
|---------|-------|-------------|
| **Q1** | Data pipeline + minimal RAG | CLI: ask questions about one Nifty 50 stock |
| **Q2** | Multi-agent system + all 50 stocks | Multi-stock comparison, bull/bear scenarios |
| **Q3** | REST API + quality evaluation | Public API endpoint, RAG eval metrics |
| **Q4** | UI dashboard + portfolio analytics | Web app, portfolio-level insights |

---

## VS Code Workflow

### Recommended Extensions
- `Python` (ms-python.python)
- `Pylance` (ms-python.vscode-pylance)
- `Jupyter` (ms-toolsai.jupyter)
- `GitLens` (eamodio.gitlens)
- `Ruff` (charliermarsh.ruff) — linter/formatter
- `Thunder Client` — test REST APIs without Postman

### Git Branch Strategy
```
main          ← stable, deployable
dev           ← integration branch
feature/q1-data-pipeline
feature/q2-rag-ingestion
feature/q2-agents
feature/q3-api
```

Never commit directly to `main`. Always open a PR from a feature branch.

---

## License

MIT — build on this freely, but give credit.
