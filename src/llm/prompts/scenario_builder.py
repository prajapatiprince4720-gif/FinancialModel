SCENARIO_SYSTEM_PROMPT = """You are a senior equity strategist who specializes in building probabilistic investment scenarios for emerging market stocks. You combine:
- Quantitative modeling (revenue projections, margin assumptions, valuation multiples)
- Qualitative assessment (management quality, industry tailwinds/headwinds, macro factors)
- Risk-adjusted thinking (what can go wrong, what can exceed expectations)

You build THREE scenarios — Bull, Base, and Bear — each with a probability, a 1-year and 3-year price target, and a clear narrative. You avoid vague platitudes and always tie scenarios to specific, measurable triggers.

Output clean Markdown. Always include a probability table."""


def build_scenario_query(ticker: str, company_name: str, horizon: str = "3 years") -> str:
    return f"""Build detailed Bull / Base / Bear investment scenarios for {company_name} ({ticker}) over a {horizon} horizon.

## Scenario Framework

### Probability Distribution
| Scenario | Probability | 1-Year Target | 3-Year Target |
|----------|-------------|---------------|---------------|
| Bull     | XX%         | ₹XXX          | ₹XXX          |
| Base     | XX%         | ₹XXX          | ₹XXX          |
| Bear     | XX%         | ₹XXX          | ₹XXX          |

### 🐂 BULL CASE (XX% probability)
**Narrative:** [What has to go RIGHT]
**Key Assumptions:**
- Revenue growth: X% CAGR
- EBITDA margin: X%
- Exit P/E multiple: Xx
**Catalysts:** [Specific triggers that would cause this]
**Timeline:** When would we know this scenario is playing out?

### 📊 BASE CASE (XX% probability)
**Narrative:** [Most likely outcome based on current trends]
**Key Assumptions:**
- Revenue growth: X% CAGR
- EBITDA margin: X%
- Exit P/E multiple: Xx
**Thesis:** [Why this is the most probable path]

### 🐻 BEAR CASE (XX% probability)
**Narrative:** [What has to go WRONG]
**Key Assumptions:**
- Revenue growth: X% CAGR
- EBITDA margin: X%
- Exit P/E multiple: Xx
**Risk Triggers:** [Specific events that would cause downside]
**Circuit Breaker:** At what price/event should an investor exit?

### Key Monitorables
List 5 metrics/events to watch quarterly that will tell you which scenario is unfolding.

Use the financial data provided to anchor all projections. State clearly where you are extrapolating vs. where data supports the assumption."""
