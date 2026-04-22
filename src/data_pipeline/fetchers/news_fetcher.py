"""
Fetches news articles for a given company from multiple free sources:
  1. NewsAPI.org  (100 req/day free)
  2. RSS feeds from Economic Times, Business Standard, Moneycontrol

Usage:
    fetcher = NewsFetcher()
    articles = fetcher.fetch("Reliance Industries", days_back=30)
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any

import requests
from bs4 import BeautifulSoup

from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)
settings = get_settings()

# Free RSS feeds covering Indian equities
RSS_FEEDS = {
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "business_standard_markets": "https://www.business-standard.com/rss/markets-106.rss",
    "moneycontrol_news": "https://www.moneycontrol.com/rss/latestnews.xml",
}


class NewsFetcher:
    def __init__(self, raw_data_dir: str | None = None) -> None:
        self.raw_data_dir = raw_data_dir or settings.raw_data_dir
        self.api_key = settings.news_api_key
        ensure_dir(os.path.join(self.raw_data_dir, "news"))

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fetch(self, company_name: str, ticker: str, days_back: int = 30) -> list[dict[str, Any]]:
        """
        Aggregate news from all available sources for a company.
        Returns a list of article dicts and saves to disk.
        """
        articles: list[dict[str, Any]] = []

        if self.api_key:
            articles += self._fetch_newsapi(company_name, days_back)
            time.sleep(0.5)  # be nice to the API

        articles += self._fetch_rss(company_name)

        # Deduplicate by title
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for a in articles:
            key = a.get("title", "")[:80]
            if key not in seen:
                seen.add(key)
                unique.append(a)

        self._save(ticker, unique)
        logger.info(f"Fetched {len(unique)} articles for {company_name}")
        return unique

    # ──────────────────────────────────────────────────────────────────────────
    # Source: NewsAPI.org
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_newsapi(self, company_name: str, days_back: int) -> list[dict[str, Any]]:
        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{company_name}" India stock',
            "from": from_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 50,
            "apiKey": self.api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json().get("articles", [])
            return [
                {
                    "source": "newsapi",
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "content": a.get("content", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                }
                for a in raw
            ]
        except Exception as exc:
            logger.warning(f"NewsAPI fetch failed: {exc}")
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # Source: RSS feeds
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_rss(self, company_name: str) -> list[dict[str, Any]]:
        articles: list[dict[str, Any]] = []
        keywords = company_name.lower().split()

        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                resp = requests.get(feed_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(resp.content, "xml")
                items = soup.find_all("item")
                for item in items:
                    title = item.find("title")
                    title_text = title.get_text() if title else ""
                    description = item.find("description")
                    desc_text = description.get_text() if description else ""
                    combined = (title_text + " " + desc_text).lower()

                    # Only include if the article mentions the company
                    if any(kw in combined for kw in keywords):
                        pub_date = item.find("pubDate")
                        articles.append(
                            {
                                "source": feed_name,
                                "title": title_text,
                                "description": desc_text,
                                "content": "",
                                "url": (item.find("link") or item.find("guid") or type("", (), {"get_text": lambda s: ""})()).get_text(),
                                "published_at": pub_date.get_text() if pub_date else "",
                            }
                        )
            except Exception as exc:
                logger.warning(f"RSS fetch failed for {feed_name}: {exc}")

        return articles

    def _save(self, ticker: str, articles: list[dict[str, Any]]) -> None:
        clean_ticker = ticker.replace(".", "_")
        path = os.path.join(self.raw_data_dir, "news", f"{clean_ticker}.json")
        with open(path, "w") as f:
            json.dump({"ticker": ticker, "articles": articles}, f, indent=2)
