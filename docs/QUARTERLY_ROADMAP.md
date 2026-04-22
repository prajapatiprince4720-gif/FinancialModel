# EquityLens — 1-Year Quarterly Roadmap

## Overview

| Quarter | Theme | Key Deliverable |
|---------|-------|-----------------|
| Q1 (Months 1–3) | Foundation & First Working RAG | CLI: research one Nifty 50 stock end-to-end |
| Q2 (Months 4–6) | All 50 Stocks + Multi-Agent | Full Nifty 50 coverage, scenario builder live |
| Q3 (Months 7–9) | API + Evaluation + Quality | Public REST API, RAG eval metrics, daily refresh |
| Q4 (Months 10–12) | UI Dashboard + Portfolio | Web app, portfolio analytics, production-ready |

---

## Q1 — Foundation (Months 1–3)

**Goal:** Get one stock working end-to-end. Don't over-engineer.

### Month 1: Data Pipeline
- [ ] Set up Python environment + Git repo (done ✓)
- [ ] Get `yfinance` fetching data for `RELIANCE.NS`
- [ ] Run `FinancialProcessor` and inspect the chunks
- [ ] Get `NewsFetcher` returning articles (free RSS at minimum)
- [ ] Understand what data you have and what's missing
- [ ] **Checkpoint:** `python -m src.data_pipeline.pipeline --ticker RELIANCE.NS` works

### Month 2: RAG Pipeline
- [ ] Install and test ChromaDB locally
- [ ] Run `RAGIngestion.ingest("RELIANCE.NS")`
- [ ] Verify vector store has documents: `store.count()`
- [ ] Test retrieval: does `Retriever.retrieve("debt levels RELIANCE")` return relevant chunks?
- [ ] Experiment with top_k values (try 3, 5, 8, 10)
- [ ] **Checkpoint:** You can ask a question and get relevant financial data back

### Month 3: LLM Integration
- [ ] Get Anthropic API key and test a basic Claude call
- [ ] Wire up `FundamentalAgent` only (skip other agents)
- [ ] Read the output critically — is it using the retrieved data?
- [ ] Experiment with temperature settings
- [ ] Add `ConvictionAgent` to simplify the output
- [ ] **Checkpoint:** `python main.py --ticker RELIANCE.NS --mode quick` produces a real report

### Q1 Learning Goals
- Understand the RAG loop: embed → store → retrieve → prompt → response
- Understand why prompt caching matters (check your Anthropic usage dashboard)
- Learn to read ChromaDB output and debug retrieval misses
- Git habit: commit every working feature with a clear message

---

## Q2 — Scale & Intelligence (Months 4–6)

**Goal:** Cover all 50 stocks, add scenarios, improve quality.

### Month 4: All Nifty 50
- [ ] Run `RAGIngestion.ingest_all()` — this will take a while, let it run overnight
- [ ] Fix any tickers that fail (some NSE symbols differ in yfinance)
- [ ] Build a simple health-check: which tickers have < 20 chunks?
- [ ] Add earnings call transcript ingestion for top 10 stocks (manual PDF + text extraction)
- [ ] **Checkpoint:** `store.get_all_tickers()` returns 45+ tickers

### Month 5: Full Agent Pipeline
- [ ] Activate `SentimentAgent` and `ScenarioAgent`
- [ ] Test `ResearchOrchestrator.research()` for 5 different stocks
- [ ] Evaluate: are the scenarios grounded in the retrieved data?
- [ ] Add a daily data refresh script (cron-style runner)
- [ ] **Checkpoint:** Full 4-part research report for any Nifty 50 stock

### Month 6: Quality Evaluation
- [ ] Build a simple eval: generate 20 questions you know the answer to (e.g. "What was Reliance FY24 revenue?")
- [ ] Score RAG retrieval: did the right chunk get retrieved?
- [ ] Score LLM output: did Claude cite the right number?
- [ ] Identify the top failure modes and fix them
- [ ] Add hybrid retrieval (dense + keyword) for financial term precision
- [ ] **Checkpoint:** RAG retrieval accuracy > 80% on your eval set

### Q2 Learning Goals
- Multi-agent coordination: how to pass outputs between agents
- Evaluation mindset: never trust AI output without measuring it
- Data quality awareness: garbage in, garbage out
- Cost management: understand your Anthropic bill

---

## Q3 — API & Production Patterns (Months 7–9)

**Goal:** Make the system accessible as an API. Add production-grade patterns.

### Month 7: FastAPI REST Layer
```
GET  /api/v1/research/{ticker}          → Full research report
GET  /api/v1/quick/{ticker}?q={question} → Quick Q&A
GET  /api/v1/tickers                    → List all ingested tickers
POST /api/v1/ingest/{ticker}            → Trigger ingestion for a ticker
GET  /api/v1/health                     → System health check
```

### Month 8: Daily Data Refresh
- [ ] Cron job: run pipeline for all 50 tickers every morning at 7 AM IST
- [ ] Smart refresh: only re-ingest if new data detected
- [ ] Add earnings calendar awareness (don't miss quarterly results)
- [ ] Alert system: flag tickers where sentiment shifted significantly

### Month 9: Observability
- [ ] Log all API calls with response times
- [ ] Track RAG retrieval quality metrics over time
- [ ] Monitor Claude API costs per ticker
- [ ] Add rate limiting to the API
- [ ] **Checkpoint:** API deployed on a free cloud instance (Railway / Render / Fly.io)

---

## Q4 — UI & Portfolio Analytics (Months 10–12)

**Goal:** Build a web dashboard. Add portfolio-level thinking.

### Month 10: Frontend (React or Streamlit)
- [ ] Streamlit app for rapid prototyping (Python, no JS needed)
- [ ] Stock search + research report display
- [ ] Sentiment timeline chart (plot news sentiment over 90 days)
- [ ] Side-by-side comparison of 2 stocks

### Month 11: Portfolio Analytics
- [ ] Input: user's holdings (ticker + quantity + buy price)
- [ ] Output: portfolio-level sentiment, sector concentration, risk assessment
- [ ] Alert: which holdings have high news sentiment shifts?
- [ ] "Portfolio health score" using conviction summaries

### Month 12: Polish & Launch
- [ ] Write comprehensive tests
- [ ] Performance optimization (caching at API layer)
- [ ] Documentation cleanup
- [ ] Share on LinkedIn/Twitter — this is portfolio-grade work
- [ ] **Final deliverable:** Working web app, public GitHub with 100+ commits

---

## Day-by-Day Habit (Most Important)

The project compounds when you treat it like a gym habit:

```
Monday:    Run fresh ingestion. Check if any stocks have new earnings.
Tuesday:   Test one new query. Document what worked/failed.
Wednesday: Read one paper or blog post on RAG/LLMs. Apply one idea.
Thursday:  Improve one component based on test failures.
Friday:    Commit, push, write 2-sentence journal entry about what you learned.
Weekend:   Stretch goal — add one new data source or feature.
```

**Rule of thumb:** 1 hour/day × 365 days = 365 hours. That's more focused time than most ML internships.

---

## Skill Progression

| Month | Primary Skill | Secondary Skill |
|-------|--------------|-----------------|
| 1–2 | Python data pipelines | API consumption |
| 3–4 | Vector databases, embeddings | Prompt engineering |
| 5–6 | Multi-agent systems | Evaluation methodology |
| 7–8 | REST API design | Cloud deployment |
| 9–10 | Frontend basics | Data visualization |
| 11–12 | System design | Production ML patterns |

By the end of Q4, you will have touched every layer of a production AI system. That is a stronger portfolio than most CS graduates have at graduation.
