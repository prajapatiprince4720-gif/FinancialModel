"""
FAQ Trainer — enriches the vector store with pre-answered investor Q&A pairs.

For each company it:
  1. Generates ~170 investor questions from the question bank
  2. Groups them into thematic batches of ~15 (so one LLM call answers many)
  3. Answers each batch using the retrieved financial context + LLM
  4. Stores every Q&A pair as a vector chunk under section='faq'

When a user later asks a similar question, the retriever surfaces the pre-computed
Q&A as high-quality context, making answers faster and more consistent.

Usage:
    python main.py train --ticker RELIANCE.NS
    python main.py train --all
    python main.py train --all --workers 3
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from config.nifty50_tickers import NIFTY50_TICKERS
from src.llm import get_llm_client
from src.rag.embeddings import get_embedding_model
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.training.question_bank import get_questions
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)

# How many questions to answer in a single LLM call
BATCH_SIZE = 15

# Sections to retrieve context from for each question batch
CONTEXT_SECTIONS = [
    "company_profile", "key_ratios", "income_statement",
    "balance_sheet", "cash_flow",
    "screener_profit_loss", "screener_balance_sheet",
    "screener_ratios", "screener_peers",
]

BATCH_SYSTEM_PROMPT = """You are a senior Indian equity analyst answering a batch of investor questions
about a specific company. Use the provided financial data to give accurate, specific answers.

Rules:
- Answer every question — number your answers to match the question numbers
- Cite real numbers from the context (₹ in Crores, ratios to 2 decimal places)
- Keep each answer to 3-6 sentences — clear and direct
- If data for a question is not in the context, give your best general answer and flag it
- Use Indian format: Crore, Lakh, NSE/BSE, SEBI
- Never say "I cannot answer" — give the best answer possible
- End investment-recommendation answers with: "Not SEBI-registered advice."

Output format — strictly follow this structure:
Q1: [exact question]
A1: [answer]

Q2: [exact question]
A2: [answer]
...and so on for all questions in the batch."""


class FAQTrainer:

    def __init__(self) -> None:
        self.llm = get_llm_client()
        self.retriever = Retriever()
        self.embedder = get_embedding_model()
        self.store = VectorStore()
        ensure_dir("data/faq_cache")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def train_ticker(self, ticker_yf: str, refresh: bool = False) -> int:
        """
        Generate and store FAQ pairs for one ticker.
        Returns number of Q&A chunks stored.
        """
        nse_symbol = ticker_yf.replace(".NS", "").replace(".BO", "")
        company_name = NIFTY50_TICKERS.get(nse_symbol, nse_symbol)

        cache_path = f"data/faq_cache/{nse_symbol}_faq.json"
        if os.path.exists(cache_path) and not refresh:
            logger.info(f"[FAQ] {ticker_yf}: loading from cache ({cache_path})")
            with open(cache_path) as f:
                qa_pairs = json.load(f)
        else:
            questions = get_questions(nse_symbol, company_name)
            logger.info(f"[FAQ] {ticker_yf}: {len(questions)} questions → answering in batches of {BATCH_SIZE}")
            qa_pairs = self._answer_all_batches(ticker_yf, company_name, questions)
            if qa_pairs:
                with open(cache_path, "w") as f:
                    json.dump(qa_pairs, f, indent=2)
                logger.info(f"[FAQ] {ticker_yf}: cached {len(qa_pairs)} Q&A pairs")
            else:
                logger.warning(f"[FAQ] {ticker_yf}: no Q&A pairs generated — skipping cache write")

        # Remove old FAQ chunks for this ticker before re-inserting
        if refresh:
            self._delete_faq_chunks(ticker_yf)

        stored = self._embed_and_store(ticker_yf, qa_pairs)
        logger.info(f"[FAQ] {ticker_yf}: stored {stored} Q&A chunks in vector store")
        return stored

    def train_all(self, refresh: bool = False, workers: int = 1) -> dict[str, int]:
        """Train FAQ for all 50 Nifty stocks."""
        tickers = [f"{sym}.NS" for sym in NIFTY50_TICKERS]
        results: dict[str, int] = {}

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self.train_ticker, t, refresh): t for t in tickers}
                for fut in as_completed(futures):
                    t = futures[fut]
                    try:
                        results[t] = fut.result()
                    except Exception as exc:
                        logger.error(f"[FAQ] {t} failed: {exc}")
                        results[t] = -1
        else:
            for t in tickers:
                try:
                    results[t] = self.train_ticker(t, refresh=refresh)
                except Exception as exc:
                    logger.error(f"[FAQ] {t} failed: {exc}")
                    results[t] = -1

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _answer_all_batches(
        self,
        ticker: str,
        company_name: str,
        questions: list[str],
    ) -> list[dict[str, str]]:
        """Split questions into batches, answer each with one LLM call."""
        all_qa: list[dict[str, str]] = []

        # Retrieve rich context once — covers all questions for this company
        context = self._build_context(ticker, company_name)

        batches = [questions[i: i + BATCH_SIZE] for i in range(0, len(questions), BATCH_SIZE)]
        for batch_idx, batch in enumerate(batches):
            logger.info(f"[FAQ] {ticker}: batch {batch_idx + 1}/{len(batches)} ({len(batch)} questions)")
            try:
                qa_pairs = self._answer_batch(ticker, company_name, batch, context)
                all_qa.extend(qa_pairs)
                time.sleep(2.5)  # respect Groq free-tier rate limits
            except Exception as exc:
                err = str(exc)
                if "tokens per day" in err or "TPD" in err:
                    # Daily quota exhausted — stop cleanly, save what we have
                    logger.warning(f"[FAQ] Daily token quota reached at batch {batch_idx + 1}. Saving {len(all_qa)} answers so far.")
                    break
                elif "rate_limit" in err or "429" in err:
                    # Per-minute limit — wait and retry once
                    logger.warning(f"[FAQ] Per-minute rate limit, waiting 65s...")
                    time.sleep(65)
                    try:
                        qa_pairs = self._answer_batch(ticker, company_name, batch, context)
                        all_qa.extend(qa_pairs)
                        time.sleep(2.5)
                    except Exception as exc2:
                        logger.warning(f"[FAQ] Retry failed for batch {batch_idx + 1}: {exc2}")
                else:
                    logger.warning(f"[FAQ] {ticker} batch {batch_idx + 1} failed: {exc}")
                    time.sleep(5)

        return all_qa

    def _build_context(self, ticker: str, company_name: str) -> str:
        """Retrieve a comprehensive context block covering all financial sections."""
        results = self.retriever.retrieve_multi_section(
            query=f"{company_name} financial analysis fundamentals valuation risks",
            ticker=ticker,
            sections=CONTEXT_SECTIONS,
            top_k_per_section=4,
        )
        return self.retriever.format_context(results)

    def _answer_batch(
        self,
        ticker: str,
        company_name: str,
        questions: list[str],
        context: str,
    ) -> list[dict[str, str]]:
        """Send one LLM call to answer a batch of questions. Parse Q/A pairs from response."""
        numbered = "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(questions))
        user_msg = (
            f"Company: {company_name} ({ticker})\n\n"
            f"Answer ALL of the following {len(questions)} investor questions:\n\n"
            f"{numbered}"
        )

        raw = self.llm.complete(
            system_prompt=BATCH_SYSTEM_PROMPT,
            user_message=user_msg,
            context=context,
            temperature=0.2,
        )

        return self._parse_qa_response(raw, questions)

    def _parse_qa_response(
        self,
        raw: str,
        questions: list[str],
    ) -> list[dict[str, str]]:
        """
        Parse LLM output into Q/A pairs.
        Expects format: Q1: ... \\nA1: ... \\nQ2: ...
        Falls back to pairing by index if parsing fails.
        """
        import re
        pairs: list[dict[str, str]] = []

        # Try structured parse: find all A{n}: blocks
        answers_raw = re.split(r"\nQ\d+:", "\n" + raw)
        # answers_raw[0] is empty, [1] = "Q1 text\nA1: answer", etc.
        for i, block in enumerate(answers_raw[1:]):
            a_match = re.search(r"A\d+:\s*(.*)", block, re.DOTALL)
            q_match = re.match(r"(.*?)\nA\d+:", block, re.DOTALL)
            question = questions[i] if i < len(questions) else f"Question {i+1}"
            answer = a_match.group(1).strip() if a_match else block.strip()
            if answer:
                pairs.append({"question": question, "answer": answer})

        # If parsing yielded nothing, store the raw batch as one chunk
        if not pairs:
            pairs.append({
                "question": questions[0] if questions else "FAQ batch",
                "answer": raw.strip(),
            })

        return pairs

    def _embed_and_store(self, ticker: str, qa_pairs: list[dict[str, str]]) -> int:
        """Embed Q&A pairs and upsert into vector store."""
        if not qa_pairs:
            return 0

        texts, metadatas, ids = [], [], []
        for i, pair in enumerate(qa_pairs):
            q = pair.get("question", "")
            a = pair.get("answer", "")
            if not q or not a:
                continue
            # Store as "Q: ...\nA: ..." so embedding captures the question intent
            # and the answer text is what the retriever returns
            text = f"FAQ — {ticker}\nQ: {q}\nA: {a}"
            doc_id = self._make_id(ticker, q)
            texts.append(text)
            metadatas.append({
                "ticker": ticker,
                "section": "faq",
                "period": "current",
                "source": "faq_training",
                "question": q[:200],
            })
            ids.append(doc_id)

        # Embed in batches of 64
        BATCH = 64
        total = 0
        for start in range(0, len(texts), BATCH):
            batch_texts = texts[start: start + BATCH]
            batch_meta = metadatas[start: start + BATCH]
            batch_ids = ids[start: start + BATCH]
            embeddings = self.embedder.embed(batch_texts)
            self.store.add(batch_texts, embeddings, batch_meta, batch_ids)
            total += len(batch_texts)

        return total

    def _delete_faq_chunks(self, ticker: str) -> None:
        """Remove existing FAQ chunks for a ticker before refreshing."""
        try:
            self.store._collection.delete(
                where={"$and": [{"ticker": ticker}, {"section": "faq"}]}
            )
            logger.info(f"[FAQ] Deleted existing FAQ chunks for {ticker}")
        except Exception as exc:
            logger.warning(f"[FAQ] Could not delete FAQ chunks for {ticker}: {exc}")

    @staticmethod
    def _make_id(ticker: str, question: str) -> str:
        content_hash = hashlib.md5(question.encode()).hexdigest()[:10]
        clean = ticker.replace(".", "_")
        return f"{clean}_faq_{content_hash}"
