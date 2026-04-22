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
    orch = ResearchOrchestrator()

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
    p_research.add_argument("--ticker", type=str, required=True)
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
