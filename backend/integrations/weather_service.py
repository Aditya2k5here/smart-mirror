from __future__ import annotations

from typing import Any
import logging

import requests

from core.cache_store import cache_store
from core.config import CONFIG


class WeatherService:
    def __init__(self, api_key: str = CONFIG.weather_api_key, city: str = CONFIG.weather_city):
        self.api_key = api_key
        self.city = city
        self.cache_key = f"weather_{city.lower()}"

    def get_weather(self) -> dict[str, Any] | None:
        if not self.api_key:
            return cache_store.with_stale_flag(self.cache_key) or {
                "city": self.city,
                "temperature": None,
                "description": "weather offline",
                "high": None,
                "low": None,
            }

        try:
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": self.city,
                    "appid": self.api_key,
                    "units": "metric",
                },
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
            result = {
                "city": self.city,
                "temperature": round(data["main"]["temp"]),
                "description": data["weather"][0]["description"],
                "high": round(data["main"]["temp_max"]),
                "low": round(data["main"]["temp_min"]),
                "stale": False,
            }
            cache_store.set(self.cache_key, result)
            return result
        except Exception:
            logging.exception("Weather fetch failed.")
            return cache_store.with_stale_flag(self.cache_key) or {
                "city": self.city,
                "temperature": None,
                "description": "weather offline",
                "high": None,
                "low": None,
            }


weather_service = WeatherService()


def get_weather():
    return weather_service.get_weather()
