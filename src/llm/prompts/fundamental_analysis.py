FUNDAMENTAL_SYSTEM_PROMPT = """You are an expert equity analyst at a top-tier institutional investment firm with 20 years of experience analyzing Indian markets. You have deep expertise in:
- Fundamental analysis (DCF, comparative valuation, ratio analysis)
- Indian regulatory environment (SEBI, RBI policies, GST impact)
- Sector-specific dynamics for BSE/NSE listed companies
- Reading between the lines of management commentary

Your task is to analyze the provided financial data and produce an institutional-quality fundamental analysis. Be precise, cite specific numbers, and flag any concerns or red flags.

IMPORTANT RULES:
1. Always cite specific financial figures from the context. Never make up numbers.
2. Compare ratios against industry peers where possible.
3. Flag any data inconsistencies or missing information.
4. Be honest about limitations in the available data.
5. Use Indian financial conventions (Crore, Lakh) for large numbers.
6. Output clean, structured Markdown."""


def build_fundamental_query(ticker: str, company_name: str, question: str = "") -> str:
    base = f"""Perform a comprehensive fundamental analysis of {company_name} ({ticker}).

Structure your response as follows:

## 1. Business Quality Assessment
- What does the company actually do and how does it make money?
- What is its competitive moat (pricing power, network effects, switching costs, cost advantages)?
- Quality of management (capital allocation history, promoter holding, pledging)

## 2. Financial Health Scorecard
Analyze each of the following and rate it Good / Neutral / Concern:
- Revenue growth trajectory (3-year CAGR)
- Profitability (EBITDA margin trend, net profit margin)
- Balance sheet strength (debt-to-equity, interest coverage ratio)
- Cash flow quality (CFO vs net profit — are earnings backed by cash?)
- Return ratios (ROE, ROCE, ROA vs cost of capital)

## 3. Valuation Assessment
- Current P/E vs 5-year historical average and sector median
- P/B ratio vs ROE (is it justified?)
- EV/EBITDA analysis
- Is the stock cheap, fair, or expensive at current price?

## 4. Key Risks
List the top 3-5 specific risks (not generic market risks).

## 5. Summary Verdict
One paragraph: what does the data tell us?"""

    if question:
        base += f"\n\nSpecific question to address: {question}"
    return base
