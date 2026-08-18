from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
import inspect
import logging
import time


EventHandler = Callable[["MirrorEvent"], Awaitable[None] | None]


@dataclass(frozen=True)
class MirrorEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class EventBus:
    """Tiny in-process event bus.

    It keeps services decoupled without introducing a message broker or framework.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: deque[MirrorEvent] = deque(maxlen=80)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    async def publish(self, event_type: str, **payload: Any) -> MirrorEvent:
        event = MirrorEvent(type=event_type, payload=payload)
        self._history.append(event)

        handlers = [
            *self._subscribers.get(event_type, []),
            *self._subscribers.get("*", []),
        ]
        for handler in handlers:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logging.exception("Event handler failed for %s", event_type)

        return event

    def recent(self, limit: int = 20) -> list[MirrorEvent]:
        return list(self._history)[-limit:]
