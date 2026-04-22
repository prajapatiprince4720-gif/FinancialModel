"""
Question bank: 170+ templates across 9 categories, covering what investors
actually search on Google, ChatGPT, and Claude about Indian stocks.

Each template uses {company} and {ticker} placeholders.
Sector-specific extras are keyed by sector string.
"""

from typing import Any

# ── Category 1: Business & Company Basics (20 questions) ─────────────────────
BUSINESS_BASICS: list[str] = [
    "What does {company} do?",
    "What business is {company} in?",
    "What products and services does {company} offer?",
    "How does {company} make money?",
    "What is the main revenue source for {company}?",
    "What sector does {company} belong to?",
    "Who is the CEO of {company}?",
    "Who are the promoters of {company}?",
    "What is the promoter holding percentage in {company}?",
    "How many employees does {company} have?",
    "Where is {company} headquartered?",
    "When was {company} founded?",
    "Is {company} a public or private company?",
    "Is {company} listed on NSE or BSE?",
    "What is {company}'s NSE ticker symbol?",
    "What is {company}'s market position in its industry?",
    "Who are the main competitors of {company}?",
    "What is {company}'s competitive advantage or moat?",
    "Does {company} operate internationally?",
    "What is the business model of {company}?",
]

# ── Category 2: Financial Performance (25 questions) ─────────────────────────
FINANCIAL_PERFORMANCE: list[str] = [
    "What is {company}'s annual revenue?",
    "What is {company}'s net profit?",
    "What is {company}'s EBITDA?",
    "What is {company}'s EBITDA margin?",
    "What is {company}'s net profit margin?",
    "What is {company}'s revenue growth rate?",
    "How has {company}'s profit grown over the last 5 years?",
    "What is {company}'s earnings per share (EPS)?",
    "Has {company}'s revenue been growing or declining?",
    "What is {company}'s operating profit?",
    "What is {company}'s free cash flow?",
    "How much cash does {company} have on its balance sheet?",
    "What is {company}'s operating cash flow?",
    "What was {company}'s revenue in the last quarter?",
    "Did {company} beat or miss earnings estimates last quarter?",
    "What is {company}'s 3-year revenue CAGR?",
    "What is {company}'s 5-year profit CAGR?",
    "Is {company} profitable?",
    "What is {company}'s gross margin?",
    "How does {company}'s profitability compare to its peers?",
    "What are {company}'s biggest expenses?",
    "What is {company}'s working capital?",
    "Does {company} generate consistent free cash flow?",
    "What is {company}'s capital expenditure?",
    "How efficiently does {company} use its assets?",
]

# ── Category 3: Balance Sheet & Debt (20 questions) ──────────────────────────
BALANCE_SHEET: list[str] = [
    "What is {company}'s total debt?",
    "What is {company}'s debt-to-equity ratio?",
    "Is {company} debt-heavy or debt-free?",
    "Does {company} have high or low debt?",
    "What is {company}'s interest coverage ratio?",
    "How much has {company} borrowed from banks?",
    "What are {company}'s total liabilities?",
    "What is {company}'s net worth or book value?",
    "What is {company}'s total assets?",
    "Has {company}'s debt been increasing or decreasing?",
    "Is {company} financially stable?",
    "What is {company}'s current ratio?",
    "Can {company} repay its debt comfortably?",
    "What is {company}'s long-term debt?",
    "Is {company} overleveraged?",
    "What is {company}'s debt level compared to its earnings?",
    "Are promoters pledging shares in {company}?",
    "What is {company}'s cash-to-debt ratio?",
    "What is {company}'s total equity?",
    "How strong is {company}'s balance sheet?",
]

# ── Category 4: Valuation (20 questions) ─────────────────────────────────────
VALUATION: list[str] = [
    "What is {company}'s P/E ratio?",
    "What is {company}'s current share price?",
    "Is {company} stock overvalued or undervalued?",
    "What is {company}'s Price-to-Book (P/B) ratio?",
    "What is {company}'s EV/EBITDA ratio?",
    "Is {company} expensive compared to peers?",
    "What is a fair value for {company} stock?",
    "What is {company}'s market capitalization?",
    "What is {company}'s 52-week high and low?",
    "How does {company}'s valuation compare to its historical average?",
    "What is {company}'s Price-to-Sales ratio?",
    "Is {company} trading at a premium or discount to intrinsic value?",
    "What is the target price for {company} according to analysts?",
    "What is {company}'s forward P/E ratio?",
    "Is now a good time to buy {company} stock?",
    "What is {company}'s enterprise value?",
    "How does {company}'s P/E compare to Nifty 50 average?",
    "Is {company} cheap at current levels?",
    "What multiple does {company} deserve?",
    "Has {company}'s stock been fairly valued historically?",
]

# ── Category 5: Return Ratios & Efficiency (15 questions) ────────────────────
RETURN_RATIOS: list[str] = [
    "What is {company}'s Return on Equity (ROE)?",
    "What is {company}'s Return on Capital Employed (ROCE)?",
    "What is {company}'s Return on Assets (ROA)?",
    "Is {company}'s ROE improving or declining?",
    "Does {company} create value for shareholders?",
    "How efficiently does {company} manage its equity capital?",
    "What is {company}'s asset turnover ratio?",
    "What is {company}'s inventory turnover?",
    "What is {company}'s debtor days (receivables turnover)?",
    "Is {company}'s ROCE higher than its cost of capital?",
    "Has {company}'s ROE been consistent over 5 years?",
    "How does {company}'s ROE compare to industry peers?",
    "Is {company} a capital-efficient business?",
    "What is {company}'s return on invested capital (ROIC)?",
    "Does {company} have improving operating leverage?",
]

# ── Category 6: Dividends & Shareholder Returns (15 questions) ───────────────
DIVIDENDS: list[str] = [
    "Does {company} pay dividends?",
    "What is {company}'s dividend yield?",
    "What is {company}'s dividend per share?",
    "Is {company} a good dividend stock?",
    "Has {company} consistently paid dividends?",
    "What is {company}'s dividend payout ratio?",
    "Does {company} do share buybacks?",
    "How much has {company} returned to shareholders?",
    "Is {company}'s dividend safe and sustainable?",
    "What is {company}'s dividend growth history?",
    "Will {company} increase its dividend?",
    "Is {company} a good stock for passive income?",
    "What is the total shareholder return from {company}?",
    "Has {company} announced any special dividend recently?",
    "Is {company} better for growth or income investing?",
]

# ── Category 7: Risks & Concerns (20 questions) ──────────────────────────────
RISKS: list[str] = [
    "What are the main risks of investing in {company}?",
    "What could go wrong with {company}?",
    "Is {company} stock risky?",
    "What are the red flags in {company}'s financials?",
    "Is {company} facing any regulatory risks?",
    "Does {company} face competition risk?",
    "What are the threats to {company}'s business?",
    "Is {company}'s management trustworthy?",
    "Has {company} been involved in any controversy or scandal?",
    "Is {company} vulnerable to commodity price changes?",
    "How does inflation affect {company}?",
    "How does interest rate hikes affect {company}?",
    "Is {company} exposed to global recession risk?",
    "Is {company} over-dependent on a single product or customer?",
    "What happened to {company}'s stock during COVID?",
    "Is {company} at risk of losing market share?",
    "What are the macro risks for {company}?",
    "Does {company} have any litigation or legal risks?",
    "Is {company}'s sector facing disruption?",
    "Are there any governance concerns with {company}?",
]

# ── Category 8: Investment Suitability (20 questions) ────────────────────────
INVESTMENT_SUITABILITY: list[str] = [
    "Should I invest in {company}?",
    "Is {company} a good long-term investment?",
    "Is {company} a good stock for beginners?",
    "Is {company} a buy, hold, or sell?",
    "Is {company} worth investing in for 5 years?",
    "Can {company} give multibagger returns?",
    "Is {company} a safe investment?",
    "Is {company} good for a SIP?",
    "Is {company} suitable for conservative investors?",
    "Is {company} a good investment for 2025?",
    "What is the expected return from {company} in 3 years?",
    "Is {company} better than a mutual fund?",
    "Should I hold or sell {company}?",
    "Is {company} a value pick at current price?",
    "At what price should I buy {company}?",
    "Is {company} worth adding to my Nifty 50 portfolio?",
    "What is the downside risk if I invest in {company}?",
    "Can {company} double my investment?",
    "Is {company} good for a 10-year investment horizon?",
    "What do analysts say about {company}?",
]

# ── Category 9: Comparison & Industry (15 questions) ─────────────────────────
COMPARISON: list[str] = [
    "How does {company} compare to its competitors?",
    "Is {company} the market leader in its sector?",
    "What is {company}'s market share?",
    "Is {company} better than its peers?",
    "Who are the top 3 competitors of {company}?",
    "How does {company} perform vs the Nifty 50 index?",
    "Is {company} a Nifty 50 constituent?",
    "What is {company}'s industry outlook?",
    "How will the sector {company} operates in grow in the next 5 years?",
    "What tailwinds does {company}'s industry have?",
    "What headwinds does {company}'s industry face?",
    "Is {company}'s sector in a bull or bear phase?",
    "How does {company} rank in its sector by revenue?",
    "What is the biggest threat to {company} from a new entrant?",
    "Which Nifty 50 stock is better: {company} or its closest peer?",
]

# ── Sector-specific extras ────────────────────────────────────────────────────
SECTOR_EXTRAS: dict[str, list[str]] = {
    "banking": [
        "What is {company}'s Net Interest Margin (NIM)?",
        "What is {company}'s gross NPA ratio?",
        "What is {company}'s net NPA ratio?",
        "What is {company}'s CASA ratio?",
        "What is {company}'s Capital Adequacy Ratio (CAR)?",
        "How safe are deposits in {company}?",
        "Is {company} a private or public sector bank?",
        "What is {company}'s loan book size?",
        "How is {company}'s asset quality?",
        "What is {company}'s credit growth rate?",
    ],
    "it": [
        "What is {company}'s revenue in USD terms?",
        "What is {company}'s attrition rate?",
        "How does rupee depreciation affect {company}?",
        "What are {company}'s largest client geographies?",
        "What is {company}'s deal win rate?",
        "How is {company} positioned in AI and cloud?",
        "What is {company}'s employee headcount?",
        "What is {company}'s revenue per employee?",
        "Is {company} growing faster than industry?",
        "What is {company}'s order book pipeline?",
    ],
    "pharma": [
        "What drugs does {company} manufacture?",
        "Does {company} have any USFDA issues?",
        "What is {company}'s R&D spend as a % of revenue?",
        "How dependent is {company} on US generics?",
        "Does {company} have any patent-protected products?",
        "What is {company}'s domestic vs export revenue split?",
        "Is {company} facing any drug recall issues?",
        "What is {company}'s ANDA pipeline?",
        "How does API price fluctuation affect {company}?",
        "Is {company} growing in the Indian domestic formulations market?",
    ],
    "fmcg": [
        "What are {company}'s best-selling products?",
        "How does {company} manage rural vs urban sales?",
        "What is {company}'s volume growth vs price growth?",
        "How does inflation affect {company}'s margins?",
        "What is {company}'s distribution reach in India?",
        "Is {company} gaining or losing market share in its categories?",
        "How strong is {company}'s brand portfolio?",
        "What is {company}'s advertising spend as % of revenue?",
        "How is {company}'s premium product strategy performing?",
        "Is {company} affected by private label competition?",
    ],
    "energy": [
        "How does crude oil price affect {company}?",
        "What is {company}'s refining capacity?",
        "Is {company} investing in renewable energy?",
        "What is {company}'s upstream vs downstream revenue split?",
        "How does government policy affect {company}'s pricing?",
        "What is {company}'s proven reserves?",
        "Is {company} transitioning to clean energy?",
        "How does the energy transition affect {company}'s long-term outlook?",
        "What is {company}'s capex plan for the next 3 years?",
        "Is {company} benefiting from India's energy security push?",
    ],
    "auto": [
        "What is {company}'s monthly vehicle sales volume?",
        "How is {company}'s EV strategy developing?",
        "What is {company}'s market share in its vehicle segment?",
        "How does commodity price inflation affect {company}?",
        "Is {company} gaining or losing market share?",
        "What is {company}'s export revenue contribution?",
        "How is {company} positioned for the EV transition?",
        "What is {company}'s capacity utilisation rate?",
        "How does rural demand affect {company}'s sales?",
        "Does {company} have a strong dealer network in India?",
    ],
    "telecom": [
        "What is {company}'s ARPU (Average Revenue Per User)?",
        "How many subscribers does {company} have?",
        "What is {company}'s 5G rollout status?",
        "Is {company} gaining or losing subscribers?",
        "What is {company}'s spectrum holdings?",
        "How does competition from rivals affect {company}?",
        "What is {company}'s data revenue growth?",
        "How is {company}'s fibre broadband business growing?",
        "What is {company}'s capex guidance for 5G?",
        "Is {company}'s ARPU improving or declining?",
    ],
    "cement": [
        "What is {company}'s cement production capacity?",
        "What is {company}'s cost per tonne?",
        "How does coal and pet coke price affect {company}?",
        "What is {company}'s realisation per tonne?",
        "Is {company} expanding capacity?",
        "How does infrastructure spending affect {company}?",
        "What is {company}'s market share in its region?",
        "Is {company} a north India or south India focused company?",
        "How does {company}'s EBITDA per tonne compare to peers?",
        "What is {company}'s logistics and freight cost?",
    ],
    "default": [
        "What are the growth drivers for {company}?",
        "Is {company} expanding into new markets?",
        "Does {company} have any upcoming product launches?",
        "What is {company}'s capex guidance?",
        "Is {company} gaining customers?",
        "What is management's revenue guidance for next year?",
        "Is {company} a capital-light or capital-intensive business?",
        "Does {company} benefit from government policy tailwinds?",
        "What are the regulatory tailwinds or headwinds for {company}?",
        "Is {company} a domestic consumption play?",
    ],
}

# Map NSE sector tags to our sector extras keys
SECTOR_MAP: dict[str, str] = {
    "HDFCBANK": "banking", "ICICIBANK": "banking", "AXISBANK": "banking",
    "KOTAKBANK": "banking", "SBIN": "banking", "INDUSINDBK": "banking",
    "HDFCLIFE": "banking", "SBILIFE": "banking",
    "TCS": "it", "INFY": "it", "HCLTECH": "it", "WIPRO": "it", "TECHM": "it",
    "SUNPHARMA": "pharma", "CIPLA": "pharma", "DRREDDY": "pharma",
    "HINDUNILVR": "fmcg", "BRITANNIA": "fmcg", "NESTLEIND": "fmcg",
    "ITC": "fmcg", "TATACONSUM": "fmcg",
    "RELIANCE": "energy", "ONGC": "energy", "BPCL": "energy",
    "NTPC": "energy", "POWERGRID": "energy",
    "TATAMOTORS": "auto", "MARUTI": "auto", "HEROMOTOCO": "auto",
    "BAJAJ-AUTO": "auto", "EICHERMOT": "auto", "M&M": "auto",
    "BHARTIARTL": "telecom",
    "ULTRACEMCO": "cement", "GRASIM": "cement",
    "JSWSTEEL": "default", "TATASTEEL": "default", "HINDALCO": "default",
    "COALINDIA": "default", "ADANIENT": "default", "ADANIPORTS": "default",
    "APOLLOHOSP": "default", "ASIANPAINT": "default",
    "BAJFINANCE": "default", "BAJAJFINSV": "default", "SHRIRAMFIN": "default",
    "BEL": "default", "LT": "default", "TITAN": "default",
    "ZOMATO": "default",
}


def get_questions(nse_symbol: str, company_name: str) -> list[str]:
    """
    Return ~170 investor questions for a given company.
    Fills {company} and {ticker} placeholders.
    """
    ticker = f"{nse_symbol}.NS"
    sector_key = SECTOR_MAP.get(nse_symbol, "default")
    sector_extras = SECTOR_EXTRAS.get(sector_key, SECTOR_EXTRAS["default"])

    all_templates = (
        BUSINESS_BASICS
        + FINANCIAL_PERFORMANCE
        + BALANCE_SHEET
        + VALUATION
        + RETURN_RATIOS
        + DIVIDENDS
        + RISKS
        + INVESTMENT_SUITABILITY
        + COMPARISON
        + sector_extras
    )

    questions: list[str] = []
    for tmpl in all_templates:
        questions.append(
            tmpl.replace("{company}", company_name).replace("{ticker}", ticker)
        )
    return questions
