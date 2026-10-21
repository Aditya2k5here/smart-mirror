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

    # ---- playback control (needs a Premium account and an active Spotify device) ----
    def control(self, command: str, query: str = "") -> str:
        """Returns a short sentence for the mirror to say."""
        client = self._get_client()
        if not client:
            return "Spotify isn't connected yet. Add your Spotify keys in the settings."
        try:
            if command == "pause":
                client.pause_playback()
                return "Paused."
            if command == "play":
                client.start_playback()
                return "Playing."
            if command == "next":
                client.next_track()
                return "Next."
            if command == "previous":
                client.previous_track()
                return "Going back."
            if command in ("volume_up", "volume_down"):
                state = client.current_playback() or {}
                volume = (state.get("device") or {}).get("volume_percent", 50)
                new = max(0, min(100, volume + (15 if command == "volume_up" else -15)))
                client.volume(new)
                return "Louder." if command == "volume_up" else "Quieter."
            if command == "search" and query:
                found = client.search(q=query, type="track,playlist", limit=1)
                tracks = (found.get("tracks") or {}).get("items") or []
                playlists = [p for p in ((found.get("playlists") or {}).get("items") or []) if p]
                if playlists and any(w in query.lower() for w in ("playlist", "mix", "radio", "beats", "vibes", "focus", "chill")):
                    client.start_playback(context_uri=playlists[0]["uri"])
                    return f"Playing {playlists[0]['name']}."
                if tracks:
                    client.start_playback(uris=[tracks[0]["uri"]])
                    return f"Playing {tracks[0]['name']} by {tracks[0]['artists'][0]['name']}."
                return f"I couldn't find {query}."
        except Exception as exc:
            logging.warning("Spotify control failed: %s", exc)
            text = str(exc).lower()
            if "premium" in text:
                return "Spotify needs a Premium account to be controlled remotely."
            if "no active device" in text or "not found" in text:
                return "Open Spotify on a device first, and I'll take it from there."
            return "Spotify didn't respond."
        return "I'm not sure how to do that with Spotify."

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
                scope="user-read-currently-playing user-modify-playback-state user-read-playback-state",
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
