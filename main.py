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
        print_cashflow_table, print_pl_trend,
        print_profit_mcap_table, save_csv,
    )

    print("\nLoading data for all 50 Nifty 50 companies from cache...")
    rows = build_dataset()

    if args.symbol:
        print_pl_trend(rows, args.symbol.upper().replace(".NS", ""))
    elif args.profit:
        print_profit_mcap_table(rows, args.year or "2025")
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


def cmd_cache(args):
    """Show Screener data age and optionally refresh stale files."""
    from src.utils.cache_manager import print_cache_table, refresh_stale
    from config.nifty50_tickers import NIFTY50_TICKERS

    symbols = list(NIFTY50_TICKERS.keys())
    print_cache_table(symbols)

    if args.refresh or args.force:
        refresh_stale(symbols, force=args.force)
    elif args.auto:
        refresh_stale(symbols)


def cmd_dcf(args):
    """Run DCF valuation engine on all 50 companies."""
    from src.valuation.dcf_engine import DCFEngine
    from src.reports.advanced_report import print_dcf_table, print_dcf_sensitivity
    from src.utils.cache_manager import warn_if_stale
    from config.nifty50_tickers import NIFTY50_TICKERS

    symbols = list(NIFTY50_TICKERS.keys())
    single_mode = False
    if args.ticker:
        sym = args.ticker.upper().replace(".NS", "")
        if sym in NIFTY50_TICKERS:
            symbols    = [sym]
            single_mode = True

    warn_if_stale(symbols)

    engine = DCFEngine()
    print(f"\nRunning DCF/RIM valuation on {len(symbols)} companies (10,000 Monte Carlo simulations each)...")

    results = []
    for sym in symbols:
        r = engine.value(sym)
        results.append(r)
        if args.verbose and r.intrinsic_value > 0:
            mos = f"{r.margin_of_safety*100:+.1f}%" if r.margin_of_safety is not None else "N/A"
            print(f"  {sym:15s}  [{r.model}]  IV: ₹{r.intrinsic_value:,.0f}  MoS: {mos}  [{r.verdict}]")

    print_dcf_table(results)

    # Sensitivity table: show for single-stock runs, or for every DCF stock in --verbose
    for r in results:
        if r.intrinsic_value > 0 and getattr(r, "model", "DCF") == "DCF":
            if single_mode or args.verbose:
                print_dcf_sensitivity(r)
                if not single_mode:
                    break   # verbose all-stock: show only first DCF stock to avoid flood


def cmd_score(args):
    """Multi-factor composite score for all Nifty 50 companies."""
    from src.scoring.composite_score import CompositeScorer
    from src.reports.advanced_report import print_score_table, print_sector_ranking
    from src.utils.cache_manager import warn_if_stale
    from config.nifty50_tickers import NIFTY50_TICKERS

    symbols = list(NIFTY50_TICKERS.keys())
    warn_if_stale(symbols)
    print(f"\nComputing multi-factor composite scores for {len(symbols)} companies...")

    scorer  = CompositeScorer()
    results = scorer.score_all(symbols)

    if args.sector:
        print_sector_ranking(results)
    else:
        print_score_table(results)


def cmd_pdf(args):
    """Generate a professional PDF equity research report for one company."""
    from src.valuation.dcf_engine      import DCFEngine
    from src.scoring.composite_score   import CompositeScorer
    from src.reports.pdf_report        import generate_pdf
    from config.nifty50_tickers        import NIFTY50_TICKERS

    sym = args.ticker.upper().replace(".NS", "")
    company = NIFTY50_TICKERS.get(sym, sym)

    print(f"\nGenerating PDF research report for {company} ({sym})...")

    dcf_result   = DCFEngine().value(sym)
    scorer       = CompositeScorer()
    score_list   = scorer.score_all(list(NIFTY50_TICKERS.keys()))
    score_result = next((r for r in score_list if r.symbol == sym), None)

    ai_thesis = ""
    if not args.no_ai:
        try:
            from src.llm import get_llm_client
            llm = get_llm_client()
            prompt = (
                f"Write a concise 3-paragraph investment thesis for {company} ({sym}), "
                f"an Indian Nifty 50 company. "
                f"Composite score: {score_result.composite:.1f}/100 ({score_result.grade}). "
                f"Revenue CAGR 3Y: {score_result.revenue_cagr_3y:+.1f}%, "
                f"EPS CAGR 3Y: {score_result.eps_cagr_3y:+.1f}%, "
                f"ROCE percentile: {score_result.roce_pct:.0f}th, "
                f"D/E: {score_result.de_ratio:.2f}x. "
                + (f"DCF IV: ₹{dcf_result.intrinsic_value:,.0f}, MoS: {dcf_result.margin_of_safety*100:+.1f}%. "
                   if dcf_result and dcf_result.margin_of_safety else "") +
                "Cover: (1) Business model & competitive moat, (2) Financial quality & growth drivers, "
                "(3) Valuation assessment & key risks. Keep it factual and data-driven."
            )
            ai_thesis = llm.complete(prompt)
            print("  AI thesis generated.")
        except Exception as e:
            print(f"  AI thesis skipped ({e})")

    out = generate_pdf(
        symbol=sym,
        company_name=company,
        dcf_result=dcf_result,
        score_result=score_result,
        ai_thesis=ai_thesis,
        output_dir="reports",
    )
    print(f"\n  PDF saved → {out}")
    print(f"  Open with: open '{out}'\n")


def cmd_flows(args):
    """FII/DII institutional money flow tracker."""
    from src.data_pipeline.fetchers.fii_dii_fetcher import FIIDIIFetcher
    from src.reports.advanced_report import print_fii_dii_table

    print("\nFetching FII/DII flow data from NSE...")
    fetcher = FIIDIIFetcher()
    summary = fetcher.fetch(force=args.refresh)
    print_fii_dii_table(summary)


def cmd_portfolio(args):
    """Markowitz portfolio optimiser using composite scores as quality tilt."""
    from src.scoring.composite_score import CompositeScorer
    from src.valuation.portfolio_optimizer import PortfolioOptimizer
    from src.reports.advanced_report import print_portfolio_table
    from config.nifty50_tickers import NIFTY50_TICKERS

    symbols = list(NIFTY50_TICKERS.keys())
    capital = args.capital

    print(f"\nScoring all {len(symbols)} companies...")
    scorer  = CompositeScorer()
    results = scorer.score_all(symbols)
    scores  = {r.symbol: r.composite for r in results}

    print(f"Optimising portfolio for ₹{capital:,.0f} across top {args.stocks} stocks (5,000 random portfolios)...")
    optimizer = PortfolioOptimizer()
    portfolio = optimizer.optimize(
        symbols=symbols,
        scores=scores,
        capital=capital,
        max_stocks=args.stocks,
        min_stocks=min(args.stocks, 5),
    )
    print_portfolio_table(portfolio)


def cmd_train(args):
    """
    Pre-answer the investor question bank per company and store it in the KB.

    Training is resumable: a company interrupted part-way through resumes from
    its cache instead of being frozen at a partial answer count. Progress is
    measured by src.training.faq_trainer.training_status(), the single source of
    truth shared with scripts/daily_train.sh.
    """
    from src.training.faq_trainer import (
        FAQTrainer, TRAINED_RATIO, company_status, training_status,
    )

    # ── progress command: show what's done / pending ──────────────────────────
    if getattr(args, "progress", False):
        st = training_status()
        per = st["per_company"]
        print(f"\n=== FAQ Training Progress: {len(st['trained'])}/{st['total']} "
              f"companies trained (>= {TRAINED_RATIO*100:.0f}% of question bank) ===\n")

        if st["trained"]:
            print("  Trained:")
            for sym in st["trained"]:
                c = per[sym]
                print(f"    {sym:15s}  {c['company']:35s}  {c['answered']:3d}/{c['expected']:3d}")
        if st["partial"]:
            print(f"\n  Partial ({len(st['partial'])}) — will resume from cache:")
            for sym in st["partial"]:
                c = per[sym]
                print(f"    {sym:15s}  {c['company']:35s}  {c['answered']:3d}/{c['expected']:3d}"
                      f"  ({c['ratio']*100:4.0f}%)  {len(c['missing'])} to go")
        if st["missing"]:
            print(f"\n  Not started ({len(st['missing'])}):")
            for sym in st["missing"]:
                print(f"    {sym:15s}  {per[sym]['company']}")

        questions_left = sum(len(per[s]["missing"]) for s in st["pending"])
        print(f"\n  Questions still to answer: {questions_left:,} "
              f"across {len(st['pending'])} companies")
        if st["pending"]:
            days_left = -(-len(st["pending"]) // max(args.batch, 1))
            print(f"  At {args.batch} companies/day -> ~{days_left} day(s) remaining")
        return

    trainer = FAQTrainer()

    # ── train a single ticker ─────────────────────────────────────────────────
    if args.ticker:
        sym = args.ticker.upper().replace(".NS", "").replace(".BO", "")
        before = company_status(sym)
        print(f"\nTraining FAQ for {args.ticker}...")
        print(f"  Cached: {before['answered']}/{before['expected']} "
              f"— {len(before['missing'])} questions to answer "
              f"in batches of 15\n")
        n = trainer.train_ticker(args.ticker, refresh=args.refresh)
        after = company_status(sym)
        print(f"\nDone — {n} Q&A chunks stored for {args.ticker}")
        print(f"  Now at {after['answered']}/{after['expected']} "
              f"({after['ratio']*100:.0f}%) — trained={after['trained']}")
        print(f"  Total docs in store: {trainer.store.count()}")
        return

    # ── batch / --all mode ────────────────────────────────────────────────────
    st  = training_status()
    per = st["per_company"]
    pending = st["pending"]

    if not pending:
        print(f"\nAll {st['total']} companies are already trained!")
        print(f"Total docs in store: {trainer.store.count()}")
        print("Use --refresh to regenerate answers.")
        return

    # Resume the closest-to-done companies first: they need the fewest calls, so
    # a quota-limited run converts tokens into completed companies fastest.
    pending.sort(key=lambda s: len(per[s]["missing"]))

    batch_size = len(pending) if getattr(args, "all", False) else args.batch
    to_train   = pending[:batch_size]
    remaining  = pending[batch_size:]
    workers    = max(1, getattr(args, "workers", 1) or 1)

    print(f"\n{'='*64}")
    print(f"  FAQ Training — {'full run' if getattr(args, 'all', False) else 'daily batch'}")
    print(f"{'='*64}")
    print(f"  Already trained : {len(st['trained'])}/{st['total']} companies")
    print(f"  Training now    : {len(to_train)} companies "
          f"({sum(len(per[s]['missing']) for s in to_train):,} questions)")
    print(f"  Still pending   : {len(remaining)} after this run")
    if workers > 1:
        print(f"  Workers         : {workers} (parallel)")
    print(f"{'='*64}\n")

    results: dict[str, int] = {}

    def _train_one(sym: str) -> int:
        return trainer.train_ticker(f"{sym}.NS", refresh=args.refresh)

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_train_one, sym): sym for sym in to_train}
            for done_count, fut in enumerate(as_completed(futures), 1):
                sym = futures[fut]
                try:
                    results[sym] = fut.result()
                    c = company_status(sym)
                    print(f"[{done_count}/{len(to_train)}] {per[sym]['company']}: "
                          f"{c['answered']}/{c['expected']} ({c['ratio']*100:.0f}%) "
                          f"trained={c['trained']}")
                except Exception as exc:
                    results[sym] = -1
                    print(f"[{done_count}/{len(to_train)}] {per[sym]['company']}: FAILED — {exc}")
    else:
        for i, sym in enumerate(to_train, 1):
            print(f"[{i}/{len(to_train)}] {per[sym]['company']} ({sym}.NS) — "
                  f"{len(per[sym]['missing'])} questions")
            try:
                results[sym] = _train_one(sym)
                c = company_status(sym)
                print(f"       {c['answered']}/{c['expected']} "
                      f"({c['ratio']*100:.0f}%) trained={c['trained']}\n")
            except Exception as exc:
                print(f"       FAILED: {exc}\n")
                results[sym] = -1

    final = training_status()
    print(f"{'='*64}")
    print(f"  Overall progress : {len(final['trained'])}/{final['total']} companies trained")
    print(f"  Total docs       : {trainer.store.count()}")
    if final["pending"]:
        left = sum(len(final['per_company'][s]['missing']) for s in final['pending'])
        print(f"  Still pending    : {len(final['pending'])} companies, {left:,} questions")
        next_up = [final["per_company"][s]["company"] for s in final["pending"][:3]]
        print(f"  Next up          : {', '.join(next_up)}"
              f"{' ...' if len(final['pending']) > 3 else ''}")
        print(f"  Re-run the same command to resume — nothing is re-answered twice.")
    else:
        print(f"  All {final['total']} companies trained!")
    print(f"{'='*64}")


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


def cmd_macro(args):
    from src.utils.macro_rates import get_rf_rate, refresh_rates, set_rf_rate
    from src.valuation.dcf_engine import _ERP, _TERM_G

    if getattr(args, "set_rate", None):
        rate = set_rf_rate(args.set_rate)
        print(f"\n  \033[92mRF rate manually set to {rate*100:.2f}%\033[0m  (cached for 24h)")
    elif getattr(args, "refresh", False):
        rate = refresh_rates()
    else:
        rate = get_rf_rate()

    coe_example = rate + 1.0 * _ERP   # beta=1.0 stock
    print()
    print("\033[1m  INDIA MACRO RATES — DCF / CAPM INPUTS\033[0m")
    print("  " + "─" * 50)
    print(f"  India 10Y GSEC yield (RF rate)  : \033[92m{rate*100:.2f}%\033[0m")
    print(f"  Equity Risk Premium (ERP)        : {_ERP*100:.1f}%  (Damodaran India estimate)")
    print(f"  Terminal Growth Rate             : {_TERM_G*100:.1f}%  (long-run nominal GDP)")
    print(f"  Example CoE (β=1.0 stock)        : {coe_example*100:.2f}%  (RF + β × ERP)")
    print()
    print("  Run  python main.py macro --refresh  to force a fresh fetch.")
    print()


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
    p_ds.add_argument("--profit", action="store_true", help="Show net profit + market cap table")
    p_ds.add_argument("--year", type=str, default="2025", help="FY year for P&L table (default: 2025)")
    p_ds.add_argument("--symbol", type=str, default="", help="5-year P&L trend for one company e.g. TCS")
    p_ds.set_defaults(func=cmd_dataset)

    # ── train ──
    p_train = subparsers.add_parser("train", help="Pre-answer 180 investor questions per company and store in knowledge base")
    p_train.add_argument("--ticker", type=str, help="Train a single stock e.g. RELIANCE.NS")
    p_train.add_argument("--all", action="store_true", help="Train ALL remaining untrained companies (ignores --batch)")
    p_train.add_argument("--batch", type=int, default=8, help="How many companies to train today (default 8)")
    p_train.add_argument("--workers", type=int, default=1, help="Train this many companies in parallel (default 1)")
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

    # ── dcf ──
    p_dcf = subparsers.add_parser("dcf", help="DCF intrinsic value + Monte Carlo for all Nifty 50")
    p_dcf.add_argument("--ticker", type=str, default="", help="Single stock e.g. RELIANCE.NS (default: all 50)")
    p_dcf.add_argument("--verbose", action="store_true", help="Print per-company progress")
    p_dcf.set_defaults(func=cmd_dcf)

    # ── score ──
    p_score = subparsers.add_parser("score", help="Multi-factor composite quality score (0-100) for all Nifty 50")
    p_score.add_argument("--sector", action="store_true", help="Show sector peer ranking view instead of full leaderboard")
    p_score.set_defaults(func=cmd_score)

    # ── flows ──
    p_flows = subparsers.add_parser("flows", help="FII/DII institutional money flow tracker (NSE live data)")
    p_flows.add_argument("--refresh", action="store_true", help="Force re-fetch from NSE (ignore cache)")
    p_flows.set_defaults(func=cmd_flows)

    # ── portfolio ──
    p_port = subparsers.add_parser("portfolio", help="Markowitz portfolio optimizer — max Sharpe from Nifty 50")
    p_port.add_argument("--capital", type=float, default=100_000, help="Capital to invest in ₹ (default: 1,00,000)")
    p_port.add_argument("--stocks", type=int, default=10, help="Max number of stocks in portfolio (default: 10)")
    p_port.set_defaults(func=cmd_portfolio)

    # ── pdf ──
    p_pdf = subparsers.add_parser("pdf", help="Generate professional PDF equity research report for a stock")
    p_pdf.add_argument("--ticker", type=str, required=True, help="Stock symbol e.g. TCS or RELIANCE.NS")
    p_pdf.add_argument("--no-ai", action="store_true", help="Skip AI thesis generation (faster, offline)")
    p_pdf.set_defaults(func=cmd_pdf)

    # ── cache ──
    p_cache = subparsers.add_parser("cache", help="Show Screener data age and refresh stale files")
    p_cache.add_argument("--refresh", action="store_true", help="Re-fetch Screener data older than 90 days")
    p_cache.add_argument("--force",   action="store_true", help="Re-fetch ALL Screener data (regardless of age)")
    p_cache.add_argument("--auto",    action="store_true", help="Auto-refresh stale files only (same as --refresh)")
    p_cache.set_defaults(func=cmd_cache)

    p_macro = subparsers.add_parser("macro", help="Show / refresh live India macro rates (10Y GSEC yield)")
    p_macro.add_argument("--refresh", action="store_true", help="Force-fetch fresh India 10Y GSEC yield from market data")
    p_macro.add_argument("--set", dest="set_rate", type=float, metavar="PCT",
                         help="Manually set the 10Y GSEC yield in %% (e.g. --set 6.75). "
                              "Check https://www.nseindia.com/market-data/bonds-trading-information")
    p_macro.set_defaults(func=cmd_macro)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
