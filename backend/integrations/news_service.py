from __future__ import annotations

import logging

import requests

from core.cache_store import cache_store
from core.config import CONFIG


class NewsService:
    def __init__(self, api_key: str = CONFIG.news_api_key) -> None:
        self.api_key = api_key
        self.cache_key = "news_headlines"

    def get_headlines(self) -> list[str]:
        if not self.api_key:
            return cache_store.get(self.cache_key) or []

        try:
            response = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "language": "en",
                    "country": "in",
                    "pageSize": 8,
                    "apiKey": self.api_key,
                },
                timeout=8,
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            logging.exception("News fetch failed.")
            return cache_store.get(self.cache_key) or []

        headlines = []
        for article in data.get("articles", []):
            title = article.get("title", "").strip()
            if title and " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()
            if title and title not in headlines:
                headlines.append(title)

        cache_store.set(self.cache_key, headlines)
        return headlines


news_service = NewsService()


def get_headlines():
    return news_service.get_headlines()
