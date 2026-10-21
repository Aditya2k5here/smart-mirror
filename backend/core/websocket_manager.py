from __future__ import annotations

from typing import Any, Awaitable, Callable
import asyncio
import json
import logging

MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]

MAX_MESSAGE_BYTES = 8192


class WebSocketManager:
    def __init__(
        self,
        snapshot_provider: Callable[[], dict[str, Any]] | None = None,
        on_message: MessageHandler | None = None,
    ):
        self._clients: set[Any] = set()
        self._snapshot_provider = snapshot_provider
        self.on_message = on_message

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def handler(self, websocket: Any) -> None:
        self._clients.add(websocket)
        logging.info("Frontend connected. Clients: %s", len(self._clients))

        if self._snapshot_provider:
            await self._safe_send(websocket, self._snapshot_provider())

        try:
            async for raw in websocket:
                if not self.on_message or isinstance(raw, bytes) or len(raw) > MAX_MESSAGE_BYTES:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    try:
                        await self.on_message(payload)
                    except Exception:
                        logging.exception("Websocket message handler failed.")
        except Exception:
            pass  # abrupt disconnects are normal
        finally:
            self._clients.discard(websocket)
            logging.info("Frontend disconnected. Clients: %s", len(self._clients))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return

        stale_clients = []
        message = json.dumps(payload, default=str)
        clients = list(self._clients)
        results = await asyncio.gather(
            *[client.send(message) for client in clients],
            return_exceptions=True,
        )

        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                stale_clients.append(client)

        for client in stale_clients:
            self._clients.discard(client)

    async def _safe_send(self, websocket: Any, payload: dict[str, Any]) -> None:
        try:
            await websocket.send(json.dumps(payload, default=str))
        except Exception:
            self._clients.discard(websocket)
