#!/usr/bin/env python3
"""
EquityLens — AI-Driven Equity Research CLI

Usage examples:
    # Fetch + embed data for one stock
    python main.py ingest --ticker RELIANCE.NS

    # Ingest all 50 Nifty stocks (takes ~10-15 minutes)
    python main.py ingest --all

    # Generate full research report
    python main.py research --ticker RELIANCE.NS

    # Quick Q&A
    python main.py ask --ticker TCS.NS --question "What is the debt-to-equity ratio?"

    # List all ingested tickers
    python main.py status
"""

import argparse
import sys
import os

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.utils.logger import get_logger
logger = get_logger("main")


def cmd_ingest(args):
    from src.rag.ingestion import RAGIngestion
    rag = RAGIngestion()

    if args.all:
        print("Ingesting all Nifty 50 tickers — this will take 10–20 minutes...\n")
        results = rag.ingest_all(refresh=args.refresh)
        print("\n=== Ingestion Summary ===")
        success = sum(1 for v in results.values() if v > 0)
        for ticker, count in results.items():
            status = f"{count:4d} chunks" if count > 0 else " FAILED"
            print(f"  {ticker:25s} {status}")
        print(f"\nSuccess: {success}/{len(results)} tickers")
        print(f"Total docs in store: {rag.store.count()}")
    elif args.ticker:
        print(f"Ingesting {args.ticker}...")
        n = rag.ingest(args.ticker, refresh=args.refresh)
        print(f"Done — {n} chunks stored")
        print(f"Total in store: {rag.store.count()}")
    else:
        print("Provide --ticker TICKER or --all")


def cmd_research(args):
    from src.agents.orchestrator import ResearchOrchestrator
    from config.nifty50_tickers import NIFTY50_TICKERS

    orch = ResearchOrchestrator()

    if not getattr(args, "all", False) and not args.ticker:
        print("Error: provide --ticker TICKER or --all")
        return

    if getattr(args, "all", False):
        tickers = [f"{sym}.NS" for sym in NIFTY50_TICKERS]
        total = len(tickers)
        failed = []
        print(f"\nGenerating research reports for all {total} Nifty 50 stocks...\n")
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{total}] {ticker} ...")
            try:
                report = orch.research(ticker_yf=ticker, horizon=args.horizon, save=True)
                saved = report.metadata.get("saved_to", "")
                print(f"       Saved → {saved}\n")
            except Exception as exc:
                print(f"       FAILED: {exc}\n")
                failed.append(ticker)
        print("="*60)
        print(f"Done. {total - len(failed)}/{total} reports generated.")
        if failed:
            print(f"Failed: {', '.join(failed)}")
        return

    print(f"\nGenerating research report for {args.ticker}")
    print("Running 4 AI agents... (this takes 2-5 minutes)\n")

    report = orch.research(
        ticker_yf=args.ticker,
        horizon=args.horizon,
        question=args.question,
        save=True,
    )

    if args.full:
        print(report.to_markdown())
    else:
        print("\n" + "="*60)
        print("CONVICTION SUMMARY (Plain English)")
        print("="*60)
        print(report.conviction_summary)
        print("\n" + "="*60)
        print("SCENARIO ANALYSIS")
        print("="*60)
        print(report.scenario_analysis)

    if report.metadata.get("saved_to"):
        print(f"\nFull report saved to: {report.metadata['saved_to']}")


def cmd_ask(args):
    from src.agents.orchestrator import ResearchOrchestrator
    orch = ResearchOrchestrator()
    print(f"\nQuerying: {args.question}\n")
    answer = orch.quick_answer(args.ticker, args.question)
    print(answer)


def cmd_chart(args):
    from src.charts.price_chart import PriceChart
    ticker = args.ticker
    print(f"\nFetching price data for {ticker} across 6 timeframes...")
    print("This takes about 15 seconds — downloading 1D, 1W, 1M, 1Y, 5Y, 10Y data\n")
    path = PriceChart().plot(ticker)
    print(f"\nChart saved → {path}")
    print("Opening chart...")


def cmd_dataset(args):
    from src.reports.dataset_report import (
        build_dataset, print_valuation_table,
        print_pl_income_table, print_pl_cost_table,
        print_cashflow_table, print_pl_trend, save_csv,
    )

    print("\nLoading data for all 50 Nifty 50 companies from cache...")
    rows = build_dataset()

    if args.symbol:
        # Single company 5-year trend
        print_pl_trend(rows, args.symbol.upper().replace(".NS", ""))
    elif args.pl:
        year = args.year or "2025"
        print_pl_income_table(rows, year)
        print_pl_cost_table(rows, year)
    elif args.valuation:
        print_valuation_table(rows)
    else:
        # Default: all four tables
        print_valuation_table(rows)
        year = args.year or "2025"
        print_pl_income_table(rows, year)
        print_pl_cost_table(rows, year)
        print_cashflow_table(rows)

    # Always save CSV
    csv_path = save_csv(rows)
    print(f"  Dataset saved → {csv_path}\n")


def cmd_train(args):
    import os
    from src.training.faq_trainer import FAQTrainer
    from config.nifty50_tickers import NIFTY50_TICKERS

    trainer = FAQTrainer()

    # ── progress command: show what's done / pending ──────────────────────────
    if getattr(args, "progress", False):
        import json
        done, pending = [], []
        for sym, name in NIFTY50_TICKERS.items():
            cache = f"data/faq_cache/{sym}_faq.json"
            if os.path.exists(cache):
                with open(cache) as f:
                    count = len(json.load(f))
                if count > 0:
                    done.append((sym, name, count))
                else:
                    pending.append((sym, name))
            else:
                pending.append((sym, name))

        print(f"\n=== FAQ Training Progress: {len(done)}/50 companies done ===\n")
        if done:
            print("✓ Done:")
            for sym, name, count in done:
                print(f"    {sym:15s}  {name:35s}  {count} Q&A pairs")
        if pending:
            print(f"\n⏳ Pending ({len(pending)} companies):")
            for sym, name in pending:
                print(f"    {sym:15s}  {name}")
        days_left = -(-len(pending) // args.batch)  # ceiling division
        print(f"\n  At {args.batch} companies/day → {days_left} day(s) remaining")
        return

    # ── train a single ticker ─────────────────────────────────────────────────
    if args.ticker:
        print(f"\nTraining FAQ for {args.ticker}...")
        print(f"Generating ~180 questions and answering in batches of 15\n")
        n = trainer.train_ticker(args.ticker, refresh=args.refresh)
        print(f"\nDone — {n} Q&A chunks stored for {args.ticker}")
        print(f"Total docs in store: {trainer.store.count()}")
        return

    # ── batch / --all mode ────────────────────────────────────────────────────
    all_symbols = list(NIFTY50_TICKERS.keys())

    # Find untrained companies (no cache or empty cache)
    import json as _json
    def _is_trained(sym: str) -> bool:
        path = f"data/faq_cache/{sym}_faq.json"
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                return len(_json.load(f)) > 0
        except Exception:
            return False

    untrained = [sym for sym in all_symbols if not _is_trained(sym)]

    if not untrained:
        print("\nAll 50 companies are already trained!")
        print(f"Total docs in store: {trainer.store.count()}")
        print("Use --refresh to regenerate answers.")
        return

    # Pick the next batch
    batch_size = args.batch if not getattr(args, "all", False) else len(untrained)
    to_train = untrained[:batch_size]
    remaining_after = untrained[batch_size:]

    print(f"\n{'='*60}")
    print(f"  FAQ Training — Daily Batch")
    print(f"{'='*60}")
    print(f"  Already trained : {len(all_symbols) - len(untrained)}/50 companies")
    print(f"  Training today  : {len(to_train)} companies")
    print(f"  Still pending   : {len(remaining_after)} after today")
    days_left = -(-len(remaining_after) // args.batch)
    if remaining_after:
        print(f"  Days remaining  : ~{days_left} more day(s) at {args.batch}/day")
    print(f"{'='*60}\n")

    results: dict[str, int] = {}
    for i, sym in enumerate(to_train, 1):
        ticker_yf = f"{sym}.NS"
        company = NIFTY50_TICKERS[sym]
        print(f"[{i}/{len(to_train)}] {company} ({ticker_yf})")
        try:
            n = trainer.train_ticker(ticker_yf, refresh=args.refresh)
            results[ticker_yf] = n
            print(f"       ✓ {n} Q&A pairs stored\n")
        except Exception as exc:
            print(f"       ✗ Failed: {exc}\n")
            results[ticker_yf] = -1

    success = sum(1 for v in results.values() if v > 0)
    total_done = len(all_symbols) - len(untrained) + success
    print(f"{'='*60}")
    print(f"  Today: {success}/{len(to_train)} companies trained successfully")
    print(f"  Overall progress: {total_done}/50 companies complete")
    print(f"  Total docs in store: {trainer.store.count()}")
    if remaining_after:
        next_up = [NIFTY50_TICKERS[s] for s in remaining_after[:3]]
        print(f"  Next up tomorrow: {', '.join(next_up)}{' ...' if len(remaining_after) > 3 else ''}")
    else:
        print(f"  All 50 companies trained!")
    print(f"{'='*60}")


def cmd_qa(args):
    from src.agents.investor_qa_agent import InvestorQAAgent
    agent = InvestorQAAgent()

    if args.question:
        answer = agent.ask(args.question, ticker=args.ticker or "")
        print(f"\n{answer}\n")
        return

    # Interactive Q&A mode
    print("\n" + "="*60)
    print("  EquityLens — Investor Q&A")
    print("  Ask anything: concepts, ratios, strategies, stock questions")
    print("  Examples:")
    print("    > What is P/E ratio?")
    print("    > How does SIP work?")
    print("    > What is Reliance's debt level?")
    print("    > Explain ROE vs ROCE")
    print("    > Is HDFC Bank's valuation fair?")
    print("    > What are the risks of investing in small-caps?")
    print("  Type 'exit' to quit.")
    print("="*60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break
        print("\nEquityLens: thinking...\n")
        try:
            answer = agent.ask(question)
            print(f"EquityLens:\n{answer}\n")
            print("-" * 60 + "\n")
        except Exception as exc:
            print(f"  Error: {exc}\n")


def cmd_chat(args):
    from src.agents.chat_agent import ChatAgent
    agent = ChatAgent()

    print("\n" + "="*60)
    print("  EquityLens AI — Interactive Financial Assistant")
    print("  Nifty 50 | Powered by Llama 3 via Groq")
    print("="*60)
    print("  Ask anything about Indian stocks, investing, or finance.")
    print("  Examples:")
    print("    > What is P/E ratio?")
    print("    > Give me top 8 Nifty companies by market cap")
    print("    > What if I invest ₹1000 in Reliance right now?")
    print("    > Compare TCS and Infosys")
    print("    > Full research report on HDFC Bank")
    print("    > Which Nifty stocks have the lowest P/E?")
    print("  Type 'exit' or 'quit' to stop. Type 'clear' to reset history.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            agent.clear_history()
            print("  [Conversation history cleared]\n")
            continue

        print("\nEquityLens: thinking...\n")
        try:
            response = agent.ask(user_input)
            print(f"EquityLens:\n{response}\n")
            print("-" * 60 + "\n")
        except Exception as exc:
            print(f"  Error: {exc}\n")


def cmd_status(args):
    from src.rag.vector_store import VectorStore
    store = VectorStore()
    tickers = store.get_all_tickers()
    print(f"\n=== EquityLens Knowledge Base Status ===")
    print(f"Total documents: {store.count()}")
    print(f"Tickers ingested: {len(tickers)}\n")
    if tickers:
        for t in tickers:
            print(f"  ✓ {t}")
    else:
        print("  (empty — run: python main.py ingest --ticker RELIANCE.NS)")


def main():
    parser = argparse.ArgumentParser(
        description="EquityLens — AI Equity Research for Nifty 50",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── ingest ──
    p_ingest = subparsers.add_parser("ingest", help="Fetch and embed financial data")
    p_ingest.add_argument("--ticker", type=str, help="e.g. RELIANCE.NS")
    p_ingest.add_argument("--all", action="store_true", help="Ingest all Nifty 50 tickers")
    p_ingest.add_argument("--refresh", action="store_true", help="Re-ingest even if already present")
    p_ingest.set_defaults(func=cmd_ingest)

    # ── research ──
    p_research = subparsers.add_parser("research", help="Generate full research report")
    p_research.add_argument("--ticker", type=str, help="e.g. RELIANCE.NS")
    p_research.add_argument("--all", action="store_true", help="Generate reports for all Nifty 50 stocks")
    p_research.add_argument("--horizon", type=str, default="3 years", help="Scenario horizon")
    p_research.add_argument("--question", type=str, default="", help="Optional focus question")
    p_research.add_argument("--full", action="store_true", help="Print all 4 sections")
    p_research.set_defaults(func=cmd_research)

    # ── ask ──
    p_ask = subparsers.add_parser("ask", help="Quick Q&A on a stock")
    p_ask.add_argument("--ticker", type=str, required=True)
    p_ask.add_argument("--question", type=str, required=True)
    p_ask.set_defaults(func=cmd_ask)

    # ── chart ──
    p_chart = subparsers.add_parser("chart", help="Generate 6-panel price chart (1D/1W/1M/1Y/5Y/10Y)")
    p_chart.add_argument("--ticker", type=str, required=True, help="e.g. RELIANCE.NS")
    p_chart.set_defaults(func=cmd_chart)

    # ── dataset ──
    p_ds = subparsers.add_parser("dataset", help="Generate P/E + P&L dataset for all 50 Nifty companies")
    p_ds.add_argument("--valuation", action="store_true", help="Show valuation table only (PE, PB, EV/EBITDA, ROE...)")
    p_ds.add_argument("--pl", action="store_true", help="Show P&L table only")
    p_ds.add_argument("--year", type=str, default="2025", help="FY year for P&L table (default: 2025)")
    p_ds.add_argument("--symbol", type=str, default="", help="5-year P&L trend for one company e.g. TCS")
    p_ds.set_defaults(func=cmd_dataset)

    # ── train ──
    p_train = subparsers.add_parser("train", help="Pre-answer 180 investor questions per company and store in knowledge base")
    p_train.add_argument("--ticker", type=str, help="Train a single stock e.g. RELIANCE.NS")
    p_train.add_argument("--all", action="store_true", help="Train ALL remaining untrained companies (ignores --batch)")
    p_train.add_argument("--batch", type=int, default=8, help="How many companies to train today (default 8)")
    p_train.add_argument("--refresh", action="store_true", help="Re-generate answers even if cached")
    p_train.add_argument("--progress", action="store_true", help="Show training progress across all 50 companies")
    p_train.set_defaults(func=cmd_train)

    # ── qa ──
    p_qa = subparsers.add_parser("qa", help="Investor Q&A — ask any investing or finance question")
    p_qa.add_argument("--question", "-q", type=str, default="", help="Question to answer (omit for interactive mode)")
    p_qa.add_argument("--ticker", type=str, default="", help="Optional: focus on a specific stock e.g. RELIANCE.NS")
    p_qa.set_defaults(func=cmd_qa)

    # ── chat ──
    p_chat = subparsers.add_parser("chat", help="Interactive financial assistant (ask anything)")
    p_chat.set_defaults(func=cmd_chat)

    # ── status ──
    p_status = subparsers.add_parser("status", help="Show knowledge base status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
