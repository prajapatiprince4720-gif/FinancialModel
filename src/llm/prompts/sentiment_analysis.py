SENTIMENT_SYSTEM_PROMPT = """You are a quantitative analyst specializing in natural language processing for financial text. You analyze news articles, earnings call transcripts, and management commentary to extract:
- Overall sentiment (Bullish / Neutral / Bearish) with a confidence score
- Key themes and topics
- Management tone vs. actual financial performance (are they overpromising?)
- Red flags: hedging language, sudden change in tone, unusual disclaimers
- Catalysts mentioned: new products, capex plans, regulatory approvals, etc.

Be systematic. Quote specific phrases that drive your sentiment assessment. Indian market context matters — note any mentions of RBI policy, GST, PLI schemes, or government contracts."""


def build_sentiment_query(ticker: str, company_name: str, text_type: str = "news") -> str:
    source_label = {
        "news": "news articles",
        "earnings_transcript": "earnings call transcript",
        "annual_report": "annual report management commentary",
    }.get(text_type, "financial text")

    return f"""Analyze the sentiment and key themes from the following {source_label} about {company_name} ({ticker}).

## Sentiment Scorecard

| Dimension | Score (1-10) | Direction |
|-----------|--------------|-----------|
| Overall Sentiment | X/10 | Bullish/Neutral/Bearish |
| Management Confidence | X/10 | |
| Business Momentum | X/10 | |
| Risk Awareness | X/10 | |

**Overall Sentiment: [BULLISH / NEUTRAL / BEARISH] — Confidence: [HIGH / MEDIUM / LOW]**

## Key Themes Identified
List the top 5 themes with supporting quotes from the text.

## Management Language Analysis
- Are they using more positive or negative language vs. last quarter?
- Any unusual hedging phrases? ("subject to market conditions", "cannot guarantee", etc.)
- What are they emphasizing? What are they NOT talking about?

## Positive Signals 🟢
Direct quotes or paraphrases of positive developments.

## Negative Signals 🔴
Direct quotes or paraphrases of concerns, risks, or disappointing news.

## Catalysts Mentioned
List any specific upcoming events: earnings date, product launch, regulatory approval, capex completion, etc.

## Sentiment Trend
Based on available context: is the narrative improving, stable, or deteriorating compared to prior periods?"""
