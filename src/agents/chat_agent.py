"""
Interactive chat interface — routes natural language questions to the right agent.

Supports:
  - Financial concept explanations ("what is P/E ratio?")
  - Single stock questions ("how is Reliance doing?")
  - Multi-stock comparisons ("compare TCS and Infosys")
  - Market overview ("top 10 Nifty stocks by market cap")
  - Investment simulations ("what if I invest ₹1000 in HDFC Bank?")
  - Full research reports ("give me a full report on Zomato")
  - General financial questions
"""

from typing import Any

from config.nifty50_tickers import NIFTY50_TICKERS
from src.agents.router_agent import RouterAgent
from src.agents.multi_stock_agent import MultiStockAgent
from src.agents.investment_agent import InvestmentAgent
from src.agents.investor_qa_agent import InvestorQAAgent
from src.llm import get_llm_client
from src.rag.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

GENERAL_SYSTEM = """You are EquityLens, an AI financial assistant for Indian retail investors.
You have deep knowledge of:
- Nifty 50 companies and Indian stock market
- Financial concepts (P/E, EBITDA, market cap, SIP, etc.)
- Investment strategies for retail investors
- NSE/BSE trading, SEBI regulations

Answer clearly and helpfully. Use ₹ for currency. Be honest about risks.
For specific stock data, note that you'll retrieve from your knowledge base.
Always add: "Not SEBI-registered investment advice." at the end of investment recommendations."""

AVAILABLE_STOCKS = "\n".join(
    f"  {sym}: {name}" for sym, name in list(NIFTY50_TICKERS.items())[:15]
) + f"\n  ... and {len(NIFTY50_TICKERS) - 15} more Nifty 50 stocks"


class ChatAgent:

    def __init__(self) -> None:
        self.llm = get_llm_client()
        self.retriever = Retriever()
        self.router = RouterAgent()
        self.multi_stock = MultiStockAgent()
        self.investment = InvestmentAgent()
        self.investor_qa = InvestorQAAgent()
        self.history: list[dict[str, str]] = []

    def ask(self, user_message: str) -> str:
        route = self.router.route(user_message)
        logger.info(f"[Chat] intent={route.intent} tickers={route.tickers} top_n={route.top_n} amount={route.amount}")

        if route.intent == "chart":
            if not route.tickers:
                response = "Which stock would you like a chart for? E.g. 'Show chart for Reliance'"
            else:
                ticker = f"{route.tickers[0]}.NS"
                company = NIFTY50_TICKERS.get(route.tickers[0], route.tickers[0])
                print(f"  Generating 6-panel price chart for {company}...")
                from src.charts.price_chart import PriceChart
                path = PriceChart().plot(ticker)
                response = (
                    f"Chart generated for **{company}** ({ticker})\n"
                    f"Saved to: `{path}`\n\n"
                    f"The chart shows 6 timeframes: 1 Day · 1 Week · 1 Month · 1 Year · 5 Years · 10 Years\n"
                    f"Each panel shows the closing price as a line graph with colour indicating direction "
                    f"(green = up, red = down) and the % change for that period."
                )

        elif route.intent == "concept":
            response = self.investor_qa.ask(user_message)

        elif route.intent == "investment_sim":
            ticker = f"{route.tickers[0]}.NS" if route.tickers else None
            amount = route.amount or 1000.0
            if not ticker:
                response = "Which stock would you like to simulate investing in? E.g. 'What if I invest ₹1000 in Reliance?'"
            else:
                response = self.investment.run(ticker=ticker, amount=amount, query=user_message)

        elif route.intent == "market_overview":
            top_n = route.top_n or 10
            response = self.multi_stock.run(
                query=user_message,
                top_n=top_n,
                metric=route.metric,
            )

        elif route.intent == "multi_stock":
            if not route.tickers:
                response = self._handle_general(user_message)
            else:
                response = self.multi_stock.run(
                    query=user_message,
                    tickers=route.tickers,
                    metric=route.metric,
                )

        elif route.intent in ("single_stock", "full_research"):
            if not route.tickers:
                response = self._handle_general(user_message)
            else:
                ticker = f"{route.tickers[0]}.NS"
                company = NIFTY50_TICKERS.get(route.tickers[0], route.tickers[0])
                if route.intent == "full_research":
                    print(f"  Running full research on {company} (takes 2-3 min)...")
                    from src.agents.orchestrator import ResearchOrchestrator
                    orch = ResearchOrchestrator()
                    report = orch.research(ticker_yf=ticker, save=True)
                    response = report.to_markdown()
                else:
                    response = self._handle_single_stock(user_message, ticker, company)

        else:
            response = self._handle_general(user_message)

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self) -> None:
        self.history.clear()

    # ──────────────────────────────────────────────────────────────────────────

    def _handle_concept(self, query: str) -> str:
        return self.llm.complete(
            system_prompt=GENERAL_SYSTEM,
            user_message=query,
            temperature=0.4,
        )

    def _handle_single_stock(self, query: str, ticker: str, company: str) -> str:
        results = self.retriever.retrieve(query, ticker=ticker, top_k=8)
        context = self.retriever.format_context(results)
        system = (
            f"You are a financial analyst answering questions about {company} ({ticker}). "
            "Use the provided context data to give specific, data-backed answers. "
            "Be concise. Use ₹ and Indian number format. Not SEBI investment advice."
        )
        return self.llm.complete(
            system_prompt=system,
            user_message=query,
            context=context,
            temperature=0.2,
        )

    def _handle_general(self, query: str) -> str:
        context = ""
        if self.history:
            context = "Recent conversation:\n" + "\n".join(
                f"{m['role'].title()}: {m['content'][:200]}"
                for m in self.history[-4:]
            )
        return self.llm.complete(
            system_prompt=GENERAL_SYSTEM,
            user_message=query,
            context=context,
            temperature=0.4,
        )
