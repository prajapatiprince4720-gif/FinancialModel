"""
Runs a system over the benchmark and reports accuracy per category.

Systems:
  equitylens   InvestorQAAgent — retrieval over the knowledge base + LLM
  llm_only     the same LLM and system prompt with no retrieved context (baseline)

Results are appended per question to data/eval/results/<system>_<provider>.jsonl,
so an interrupted run (rate limit, daily quota) resumes where it stopped.

Usage:
    python main.py eval run --system equitylens --provider groq
    python main.py eval report
"""

import json
import os
import time
from collections import defaultdict
from typing import Any, Callable

from src.eval import benchmark, scorer
from src.utils.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = "data/eval/results"
CATEGORIES = ["lookup", "derived", "compare", "unanswerable"]


def _client(provider: str):
    from src.llm import ClaudeClient, GeminiClient, GroqClient
    return {"groq": GroqClient, "gemini": GeminiClient, "claude": ClaudeClient}[provider]()


def _make_system(system: str, provider: str) -> Callable[[dict[str, Any]], str]:
    llm = _client(provider)
    if system == "equitylens":
        from src.agents.investor_qa_agent import InvestorQAAgent
        agent = InvestorQAAgent()
        agent.llm = llm
        return lambda q: agent.ask(q["question"], ticker=q["ticker"])
    if system == "llm_only":
        from src.llm.prompts.investor_qa import INVESTOR_QA_SYSTEM_PROMPT, build_investor_qa_query
        return lambda q: llm.complete(
            system_prompt=INVESTOR_QA_SYSTEM_PROMPT,
            user_message=build_investor_qa_query(q["question"]),
            temperature=0.3,
        )
    raise ValueError(f"Unknown system: {system}")


def _is_daily_quota(err: str) -> bool:
    e = err.lower()
    return "per day" in e or "tpd" in e or "perday" in e or "rpd" in e


def _is_rate_limit(err: str) -> bool:
    e = err.lower()
    return "429" in e or "rate limit" in e or "rate_limit" in e or "resource" in e or "quota" in e


def _model_name(provider: str) -> str:
    from config.settings import get_settings
    settings = get_settings()
    model = {"groq": settings.groq_model, "gemini": settings.gemini_model, "claude": settings.claude_model}[provider]
    return model.split("/")[-1]


def results_path(system: str, provider: str) -> str:
    return os.path.join(RESULTS_DIR, f"{system}_{provider}-{_model_name(provider)}.jsonl")


def _load_results(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


def run(system: str, provider: str, limit: int | None = None, delay: float = 2.0) -> dict[str, dict[str, Any]]:
    questions = benchmark.load()
    path = results_path(system, provider)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = _load_results(path)
    todo = [q for q in questions if q["id"] not in done][:limit]
    logger.info(f"[eval] {system}/{provider}: {len(done)} done, {len(todo)} to run")

    ask = _make_system(system, provider)
    with open(path, "a") as f:
        for i, q in enumerate(todo, 1):
            answer = None
            for attempt in range(4):
                try:
                    answer = ask(q)
                    break
                except Exception as exc:
                    err = str(exc)
                    if _is_daily_quota(err):
                        logger.warning(f"[eval] Daily quota reached after {i - 1} questions — rerun later to resume: {err[:200]}")
                        return done
                    if _is_rate_limit(err):
                        wait = 30 * (attempt + 1)
                        logger.warning(f"[eval] Rate limited, waiting {wait}s: {err[:160]}")
                        time.sleep(wait)
                        continue
                    logger.warning(f"[eval] {q['id']} failed: {err[:300]}")
                    break
            if answer is None:
                continue
            result = {"id": q["id"], "category": q["category"], "answer": answer, **scorer.score(q, answer)}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            done[q["id"]] = result
            mark = "✓" if result["correct"] else "✗"
            print(f"  [{i}/{len(todo)}] {mark} {q['id']}")
            time.sleep(delay)
    return done


def report() -> None:
    """Print an accuracy table for every system/provider that has results."""
    files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith(".jsonl")) if os.path.isdir(RESULTS_DIR) else []
    if not files:
        print("No results yet — run: python main.py eval run --system equitylens")
        return
    questions = {q["id"]: q for q in benchmark.load()}
    total_questions = len(questions)
    header = f"{"system":36s}" + "".join(f"{c:>14s}" for c in CATEGORIES) + f"{'overall':>11s}{'answered':>11s}"
    print("\n" + header)
    print("─" * len(header))
    for fname in files:
        rows = _load_results(os.path.join(RESULTS_DIR, fname)).values()
        by_cat: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            # rescore with the current scorer so scorer fixes apply to earlier runs
            q = questions.get(r["id"])
            correct = scorer.score(q, r["answer"])["correct"] if q else r["correct"]
            by_cat[r["category"]].append(correct)
        cells = []
        for c in CATEGORIES:
            v = by_cat.get(c, [])
            cells.append(f"{sum(v)}/{len(v)} {100 * sum(v) / len(v):3.0f}%" if v else "—")
        allv = [x for v in by_cat.values() for x in v]
        overall = f"{100 * sum(allv) / len(allv):.1f}%" if allv else "—"
        print(f"{fname[:-6]:36s}" + "".join(f"{c:>14s}" for c in cells)
              + f"{overall:>11s}{f'{len(allv)}/{total_questions}':>11s}")
    print()
