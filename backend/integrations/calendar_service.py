from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path

from core.cache_store import cache_store
from core.config import BACKEND_DIR

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False


SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


class CalendarService:
    def __init__(
        self,
        credentials_path: Path = BACKEND_DIR / "credentials.json",
        token_path: Path = BACKEND_DIR / "token.json",
    ) -> None:
        self.credentials_path = credentials_path
        self.token_path = token_path

    def get_calendar_events(self) -> list[dict[str, str]]:
        if not GOOGLE_CALENDAR_AVAILABLE or not self.credentials_path.exists():
            return cache_store.get("calendar_events") or []

        try:
            service = self._build_service()
            now = datetime.now(timezone.utc)
            end = now + timedelta(hours=24)
            events_result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=now.isoformat(),
                    timeMax=end.isoformat(),
                    maxResults=5,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
        except Exception:
            logging.exception("Calendar fetch failed.")
            return cache_store.get("calendar_events") or []

        events = []
        for event in events_result.get("items", []):
            start = event["start"].get("dateTime", event["start"].get("date"))
            title = event.get("summary", "No title")
            events.append({"time": self._format_start(start), "title": title, "start": start})
        cache_store.set("calendar_events", events)
        return events

    def _build_service(self):
        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            self.token_path.write_text(creds.to_json(), encoding="utf-8")

        return build("calendar", "v3", credentials=creds)

    def _format_start(self, start: str) -> str:
        if "T" not in start:
            return "All day"
        return datetime.fromisoformat(start).strftime("%I:%M %p")


calendar_service = CalendarService()


def get_calendar_events():
    return calendar_service.get_calendar_events()
