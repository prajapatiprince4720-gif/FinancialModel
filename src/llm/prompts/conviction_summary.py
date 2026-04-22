CONVICTION_SYSTEM_PROMPT = """You are a financial educator who translates complex institutional equity research into plain language for first-time retail investors in India. Your users are 22–35-year-olds using Zerodha or Groww who are smart but don't have finance backgrounds.

Your golden rule: If a 20-year-old college student can't understand it, rewrite it.

Never use jargon without explaining it. Never give explicit buy/sell advice (regulatory). Frame everything as "here's what the data suggests" and "here's what to watch."

Be honest about uncertainty. Not every stock has a clear thesis."""


def build_conviction_query(
    ticker: str,
    company_name: str,
    fundamental_analysis: str,
    scenario_analysis: str,
) -> str:
    return f"""Based on the following institutional research on {company_name} ({ticker}), write a plain-English conviction summary for a retail investor.

=== FUNDAMENTAL ANALYSIS ===
{fundamental_analysis[:3000]}

=== SCENARIO ANALYSIS ===
{scenario_analysis[:2000]}

=== YOUR TASK ===

Write a 400–600 word plain-English research brief with these sections:

## {company_name} — What You Need to Know

### What Does This Company Do? (2-3 sentences)
Explain the business like you're telling a friend.

### The Numbers in Plain English
Pick the 3 most important financial facts and explain what they mean for a normal investor. Avoid jargon — if you must use a term like "P/E ratio", explain it in one sentence.

### Why People Are Optimistic 🐂
3 bullet points. Keep each to one sentence. No jargon.

### What Could Go Wrong 🐻
3 bullet points. Be honest. Don't sugarcoat.

### The Bottom Line
One paragraph (4-5 sentences). What does the data suggest? Is the risk-reward attractive? What should an investor do BEFORE making any decision?

### Questions to Ask Yourself Before Investing
- [ ] Do I understand how this company makes money?
- [ ] Am I comfortable holding this for at least 3 years?
- [ ] Have I checked the latest quarterly results?
- [ ] Is this more than 5% of my total portfolio?

**Disclaimer:** This is AI-generated research for educational purposes only. It is not SEBI-registered investment advice. Always do your own research (DYOR) and consult a SEBI-registered investment advisor before investing."""
