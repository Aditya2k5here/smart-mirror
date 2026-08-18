from __future__ import annotations

from typing import Any, Callable
import asyncio
import json
import logging


class WebSocketManager:
    def __init__(self, snapshot_provider: Callable[[], dict[str, Any]] | None = None):
        self._clients: set[Any] = set()
        self._snapshot_provider = snapshot_provider

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def handler(self, websocket: Any) -> None:
        self._clients.add(websocket)
        logging.info("Frontend connected. Clients: %s", len(self._clients))

        if self._snapshot_provider:
            await self._safe_send(websocket, self._snapshot_provider())

        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            logging.info("Frontend disconnected. Clients: %s", len(self._clients))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return

        stale_clients = []
        message = json.dumps(payload, default=str)
        results = await asyncio.gather(
            *[client.send(message) for client in list(self._clients)],
            return_exceptions=True,
        )

        for client, result in zip(list(self._clients), results):
            if isinstance(result, Exception):
                stale_clients.append(client)

        for client in stale_clients:
            self._clients.discard(client)

    async def _safe_send(self, websocket: Any, payload: dict[str, Any]) -> None:
        try:
            await websocket.send(json.dumps(payload, default=str))
        except Exception:
            self._clients.discard(websocket)
