from __future__ import annotations

from typing import Any
import logging
import os

from core.cache_store import cache_store
from core.config import CONFIG

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    SPOTIFY_AVAILABLE = True
except ImportError:
    spotipy = None
    SpotifyOAuth = None
    SPOTIFY_AVAILABLE = False


class SpotifyService:
    def __init__(self) -> None:
        self._client = None

    def get_current_track(self) -> dict[str, Any] | None:
        client = self._get_client()
        if not client:
            return None

        try:
            current = client.current_user_playing_track()
        except Exception:
            logging.exception("Spotify fetch failed.")
            cached = cache_store.with_stale_flag("spotify_current_track", max_age_seconds=900)
            return cached if isinstance(cached, dict) else None

        if not current or not current.get("is_playing"):
            return None

        item = current.get("item") or {}
        album = item.get("album") or {}
        images = album.get("images") or []
        artists = item.get("artists") or []
        result = {
            "title": item.get("name", ""),
            "artist": artists[0].get("name", "") if artists else "",
            "image": images[0].get("url") if images else None,
            "stale": False,
        }
        cache_store.set("spotify_current_track", result)
        return result

    def _get_client(self):
        if self._client:
            return self._client
        if not SPOTIFY_AVAILABLE:
            return None

        client_id = os.environ.get("SPOTIFY_CLIENT_ID")
        client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
        redirect_uri = os.environ.get(
            "SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback"
        )

        if not client_id or not client_secret:
            return None

        cache_path = str(CONFIG.data_dir / ".spotify_cache")
        try:
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-read-currently-playing",
                cache_path=cache_path,
            )
            self._client = spotipy.Spotify(auth_manager=auth_manager)
            return self._client
        except Exception:
            logging.exception("Spotify client setup failed.")
            return None


spotify_service = SpotifyService()


def get_current_track():
    return spotify_service.get_current_track()
