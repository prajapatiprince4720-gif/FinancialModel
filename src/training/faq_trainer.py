"""
FAQ Trainer — enriches the vector store with pre-answered investor Q&A pairs.

For each company it:
  1. Generates ~180 investor questions from the question bank
  2. Groups them into thematic batches of ~15 (so one LLM call answers many)
  3. Answers each batch using the retrieved financial context + LLM
  4. Stores every Q&A pair as a vector chunk under section='faq'

When a user later asks a similar question, the retriever surfaces the pre-computed
Q&A as high-quality context, making answers faster and more consistent.

Training is RESUMABLE. Free LLM tiers impose a daily token cap, so a company's
run is routinely cut off part-way through its 12 batches. Rather than treating
the partial cache as final, every run diffs the cached questions against the
full question bank and answers only what is still missing. A company interrupted
at 15/180 answers therefore heals on the next run instead of being frozen
forever, which is what previously stalled the fleet at 4 of 50 companies.

Usage:
    python main.py train --ticker RELIANCE.NS
    python main.py train --all
    python main.py train --all --workers 3
"""

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
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

# Where per-company answer caches live.
FAQ_CACHE_DIR = "data/faq_cache"

# A company counts as trained once this share of its question bank is answered.
# Expressed as a ratio, not a magic number, so it stays correct if the bank grows.
# Single source of truth: main.py and scripts/daily_train.sh both read it through
# training_status() rather than hardcoding their own thresholds, which previously
# disagreed (150 vs. "any answer at all") and let the nightly job declare victory
# over companies that were 8% trained.
TRAINED_RATIO = 0.85

# An echoed question must look at least this similar to the one we asked before
# we trust the pairing. Guards against the LLM renumbering or skipping an item
# and silently welding every later answer to the wrong question.
_QUESTION_MATCH_FLOOR = 0.60

# ── Retry policy for provider errors ──────────────────────────────────────────
# The daily cap is terminal for the run: stop and resume tomorrow.
_QUOTA_SIGNATURES = (
    "tokens per day", "tpd", "quota exceeded", "exceeded your current quota",
    "daily limit",
)
# Congestion and per-minute limits are worth waiting out.
_TRANSIENT_SIGNATURES = (
    "503", "unavailable", "high demand", "overloaded",
    "429", "rate limit", "rate_limit", "resource_exhausted",
    "500", "internal", "deadline", "timeout", "temporarily",
)
_MAX_BATCH_ATTEMPTS = 4      # 1 initial try + 3 backoff retries
_RETRY_BASE_SLEEP   = 20     # seconds; doubles each retry (20 → 40 → 80)
_INTER_BATCH_SLEEP  = 2.5    # polite gap between successful batches

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


# ── Shared progress accounting ────────────────────────────────────────────────
# Every caller (CLI, nightly script, web API) must measure "trained" the same way.

def cache_path(nse_symbol: str) -> str:
    return os.path.join(FAQ_CACHE_DIR, f"{nse_symbol}_faq.json")


def load_cached_pairs(nse_symbol: str) -> list[dict[str, str]]:
    """Return cached Q&A pairs for a symbol, or [] if absent/corrupt."""
    path = cache_path(nse_symbol)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        logger.warning(f"[FAQ] {nse_symbol}: cache unreadable, treating as empty")
        return []


def _normalise(question: str) -> str:
    """Collapse a question to a comparison key (case/punctuation/spacing-insensitive)."""
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def company_status(nse_symbol: str) -> dict[str, Any]:
    """
    Progress for one company: how many of its questions are answered.

    Returns answered/expected/missing/ratio/trained so callers never have to
    invent their own completeness rule.
    """
    company_name = NIFTY50_TICKERS.get(nse_symbol, nse_symbol)
    expected     = get_questions(nse_symbol, company_name)
    cached       = load_cached_pairs(nse_symbol)

    answered_keys = {
        _normalise(p.get("question", ""))
        for p in cached
        if p.get("question") and p.get("answer")
    }
    missing = [q for q in expected if _normalise(q) not in answered_keys]
    answered = len(expected) - len(missing)
    ratio    = answered / len(expected) if expected else 0.0

    return {
        "symbol":   nse_symbol,
        "company":  company_name,
        "answered": answered,
        "expected": len(expected),
        "missing":  missing,
        "ratio":    ratio,
        "trained":  ratio >= TRAINED_RATIO,
    }


def training_status(symbols: list[str] | None = None) -> dict[str, Any]:
    """Fleet-wide training progress. The single source of truth for 'how far along are we'."""
    symbols = symbols or list(NIFTY50_TICKERS.keys())
    per     = {sym: company_status(sym) for sym in symbols}
    trained = [s for s, v in per.items() if v["trained"]]
    partial = [s for s, v in per.items() if not v["trained"] and v["answered"] > 0]
    missing = [s for s, v in per.items() if v["answered"] == 0]
    return {
        "per_company": per,
        "trained":     trained,
        "partial":     partial,
        "missing":     missing,
        "pending":     partial + missing,
        "total":       len(symbols),
    }


class FAQTrainer:

    def __init__(self) -> None:
        self.llm = get_llm_client()
        self.retriever = Retriever()
        self.embedder = get_embedding_model()
        self.store = VectorStore()
        ensure_dir(FAQ_CACHE_DIR)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def train_ticker(self, ticker_yf: str, refresh: bool = False) -> int:
        """
        Generate and store FAQ pairs for one ticker, resuming a partial run.

        Only the questions still missing from the cache are sent to the LLM, so an
        interrupted company picks up where it left off instead of returning its
        truncated cache forever. Returns the number of Q&A chunks stored.
        """
        nse_symbol   = ticker_yf.replace(".NS", "").replace(".BO", "")
        company_name = NIFTY50_TICKERS.get(nse_symbol, nse_symbol)
        path         = cache_path(nse_symbol)

        if refresh:
            # Start clean: drop the cache and the stored vectors, re-answer everything.
            existing_pairs = []
            todo = get_questions(nse_symbol, company_name)
            self._delete_faq_chunks(ticker_yf)
            logger.info(f"[FAQ] {ticker_yf}: refresh — re-answering all {len(todo)} questions")
        else:
            status         = company_status(nse_symbol)
            existing_pairs = load_cached_pairs(nse_symbol)
            todo           = status["missing"]
            if todo:
                logger.info(
                    f"[FAQ] {ticker_yf}: resuming — {status['answered']}/{status['expected']} "
                    f"cached, {len(todo)} still to answer"
                )
            else:
                logger.info(
                    f"[FAQ] {ticker_yf}: cache complete "
                    f"({status['answered']}/{status['expected']}) — embedding only"
                )

        new_pairs: list[dict[str, str]] = []
        if todo:
            new_pairs = self._answer_all_batches(ticker_yf, company_name, todo)

        # Merge, letting a freshly generated answer win over a stale cached one.
        qa_pairs = self._merge_pairs(existing_pairs, new_pairs)

        if new_pairs:
            try:
                with open(path, "w") as f:
                    json.dump(qa_pairs, f, indent=2)
                logger.info(
                    f"[FAQ] {ticker_yf}: cached {len(qa_pairs)} Q&A pairs "
                    f"(+{len(new_pairs)} this run)"
                )
            except Exception as exc:
                logger.error(f"[FAQ] {ticker_yf}: could not write cache: {exc}")
        elif todo:
            logger.warning(
                f"[FAQ] {ticker_yf}: no new answers generated this run "
                f"(quota exhausted or LLM error) — {len(todo)} still pending"
            )

        stored = self._embed_and_store(ticker_yf, qa_pairs)
        final  = company_status(nse_symbol)
        logger.info(
            f"[FAQ] {ticker_yf}: stored {stored} chunks | "
            f"{final['answered']}/{final['expected']} answered "
            f"({final['ratio']*100:.0f}%) | trained={final['trained']}"
        )
        return stored

    @staticmethod
    def _merge_pairs(
        existing: list[dict[str, str]],
        fresh: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Union of cached and new pairs, de-duplicated by normalised question."""
        merged: dict[str, dict[str, str]] = {}
        for pair in list(existing) + list(fresh):
            q, a = pair.get("question", ""), pair.get("answer", "")
            if q and a:
                merged[_normalise(q)] = {"question": q, "answer": a}
        return list(merged.values())

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
        """
        Split questions into batches and answer each with one LLM call.

        Each batch is retried with exponential backoff on transient provider
        errors. Gemini Flash in particular returns 503 "high demand" on a large
        fraction of calls, and the previous handler only retried 429s — so most
        503s were logged once and their 15 questions abandoned for the run.
        """
        all_qa: list[dict[str, str]] = []

        batches = [questions[i: i + BATCH_SIZE] for i in range(0, len(questions), BATCH_SIZE)]
        for batch_idx, batch in enumerate(batches):
            label = f"batch {batch_idx + 1}/{len(batches)}"
            logger.info(f"[FAQ] {ticker}: {label} ({len(batch)} questions)")
            # Context is retrieved per batch, keyed on that batch's own questions.
            # A single company-wide query fed dividend questions the same chunks as
            # debt questions, which is the main reason cached answers drifted generic.
            context = self._build_context(ticker, company_name, batch)

            pairs, quota_exhausted = self._answer_batch_with_retry(
                ticker, company_name, batch, context, label,
            )
            all_qa.extend(pairs)

            if quota_exhausted:
                logger.warning(
                    f"[FAQ] {ticker}: daily quota reached at {label}. "
                    f"Keeping {len(all_qa)} answers; the rest resume on the next run."
                )
                break

            time.sleep(_INTER_BATCH_SLEEP)

        return all_qa

    def _answer_batch_with_retry(
        self,
        ticker: str,
        company_name: str,
        batch: list[str],
        context: str,
        label: str,
    ) -> tuple[list[dict[str, str]], bool]:
        """
        Answer one batch, retrying transient provider errors.

        Returns (pairs, quota_exhausted). `quota_exhausted` tells the caller to
        stop the whole company rather than burn through remaining batches that
        will all fail the same way.
        """
        for attempt in range(1, _MAX_BATCH_ATTEMPTS + 1):
            try:
                return self._answer_batch(ticker, company_name, batch, context), False
            except Exception as exc:
                err = str(exc).lower()

                if any(sig in err for sig in _QUOTA_SIGNATURES):
                    return [], True

                if any(sig in err for sig in _TRANSIENT_SIGNATURES):
                    if attempt == _MAX_BATCH_ATTEMPTS:
                        logger.warning(
                            f"[FAQ] {ticker} {label}: still failing after "
                            f"{attempt} attempts ({str(exc)[:90]}). Left pending."
                        )
                        return [], False
                    # Exponential backoff with a floor, so a congested model gets
                    # time to recover instead of being hammered.
                    delay = _RETRY_BASE_SLEEP * (2 ** (attempt - 1))
                    logger.warning(
                        f"[FAQ] {ticker} {label}: transient error "
                        f"({str(exc)[:70]}) — retry {attempt}/{_MAX_BATCH_ATTEMPTS - 1} in {delay}s"
                    )
                    time.sleep(delay)
                    continue

                logger.warning(f"[FAQ] {ticker} {label} failed: {str(exc)[:140]}")
                time.sleep(5)
                return [], False

        return [], False

    def _build_context(
        self,
        ticker: str,
        company_name: str,
        questions: list[str] | None = None,
    ) -> str:
        """
        Retrieve a context block covering the financial sections.

        When `questions` is supplied the retrieval query is built from those
        questions, so each batch gets chunks relevant to what it actually asks.
        """
        if questions:
            query = f"{company_name} " + " ".join(questions)
        else:
            query = f"{company_name} financial analysis fundamentals valuation risks"

        results = self.retriever.retrieve_multi_section(
            query=query,
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
        Parse LLM output into Q/A pairs, keyed on the model's OWN numbering.

        Expected format per item:  Qn: <echoed question>\\nAn: <answer>

        The previous implementation paired answers to questions by list position.
        If the model skipped or reordered one item, every later answer was welded
        to the wrong question and then embedded as authoritative FAQ — silently
        poisoning retrieval. We instead read the integer the model emitted, use it
        to index the batch, and verify the echoed question text actually resembles
        the question we asked. Pairs that fail that check are dropped, because a
        wrong answer in the knowledge base is worse than a missing one (a missing
        one is simply retried on the next resumable run).
        """
        pairs: dict[int, dict[str, str]] = {}

        # Split on "Qn:" markers, capturing n so we can trust the model's numbering.
        blocks = re.split(r"(?:^|\n)\s*Q\s*(\d+)\s*[:.]", raw)
        # blocks = [preamble, "1", body1, "2", body2, ...]
        for idx in range(1, len(blocks) - 1, 2):
            try:
                number = int(blocks[idx])
            except ValueError:
                continue
            body = blocks[idx + 1]

            a_match = re.search(r"(?:^|\n)\s*A\s*\d*\s*[:.]\s*(.*)", body, re.DOTALL)
            if not a_match:
                continue
            answer = a_match.group(1).strip()
            # Trailing content of the last item may include the model's sign-off;
            # everything up to the next Qn marker already got split off above.
            if not answer:
                continue

            position = number - 1
            if not (0 <= position < len(questions)):
                logger.debug(f"[FAQ] dropping answer {number}: outside batch of {len(questions)}")
                continue

            expected = questions[position]
            echoed   = body[: a_match.start()].strip()
            if echoed and not self._questions_match(echoed, expected):
                logger.warning(
                    f"[FAQ] dropping Q{number}: echoed question does not match "
                    f"the one asked (asked {expected!r}, got {echoed[:70]!r})"
                )
                continue

            pairs[position] = {"question": expected, "answer": answer}

        if not pairs:
            # Never store an unparsed blob under questions[0] — that mislabels a
            # whole batch of answers as the answer to one question. Returning empty
            # leaves the questions marked missing so the next run retries them.
            logger.warning(
                f"[FAQ] could not parse any Q/A pair from a {len(questions)}-question "
                f"batch response ({len(raw)} chars); leaving them pending for retry"
            )
            return []

        return [pairs[k] for k in sorted(pairs)]

    @staticmethod
    def _questions_match(echoed: str, expected: str) -> bool:
        """
        True when the model's echoed question is recognisably the one we asked.

        Models lightly reword prompts (contractions, punctuation, a dropped
        company suffix), so this is a fuzzy ratio rather than equality.
        """
        a, b = _normalise(echoed), _normalise(expected)
        if not a or not b:
            return False
        if a == b or a in b or b in a:
            return True
        return SequenceMatcher(None, a, b).ratio() >= _QUESTION_MATCH_FLOOR

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
