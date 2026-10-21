"""Provider-agnostic LLM access: Groq -> Gemini -> OpenAI with a circuit breaker.

Everything goes through plain `requests` so there are no SDK dependencies, and the
transport is injectable so the whole assistant can be tested without a network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import base64
import logging
import time

import requests

from core.config import CONFIG, MirrorConfig
from core.diagnostics import diagnostics_service

Post = Callable[..., Any]

BREAKER_SECONDS = 45.0


@dataclass
class LLMReply:
    text: str
    provider: str
    latency_ms: int


class LLMRouter:
    def __init__(self, config: MirrorConfig = CONFIG, post: Post | None = None) -> None:
        self.config = config
        self._post = post or requests.post
        self._tripped_until: dict[str, float] = {}

    # ---- availability ----------------------------------------------------
    def configured(self) -> list[str]:
        found = []
        if self.config.groq_api_key:
            found.append("groq")
        if self.config.gemini_api_key:
            found.append("gemini")
        if self.config.openai_api_key:
            found.append("openai")
        return found

    def providers(self, vision: bool = False) -> list[str]:
        mode = self.config.ai_router_mode
        configured = self.configured()
        if mode in {"groq", "gemini", "openai"}:
            order = [mode] if mode in configured else []
        else:
            order = configured
        if vision:
            # Gemini is by far the strongest/cheapest vision option here.
            order = sorted(order, key=lambda p: {"gemini": 0, "openai": 1, "groq": 2}[p])
        now = time.monotonic()
        return [p for p in order if self._tripped_until.get(p, 0) <= now]

    def available(self) -> bool:
        return bool(self.providers())

    def _trip(self, provider: str) -> None:
        self._tripped_until[provider] = time.monotonic() + BREAKER_SECONDS

    # ---- chat ------------------------------------------------------------
    def chat(self, system: str, messages: list[dict[str, str]], max_tokens: int = 320,
             temperature: float = 0.6, json_mode: bool = False) -> LLMReply | None:
        for provider in self.providers():
            started = time.monotonic()
            try:
                text = self._chat_one(provider, system, messages, max_tokens, temperature, json_mode)
            except Exception as exc:  # network, quota, bad JSON ...
                logging.warning("LLM provider %s failed: %s", provider, exc)
                self._trip(provider)
                continue
            if text:
                latency = int((time.monotonic() - started) * 1000)
                diagnostics_service.record_ai(provider, latency)
                return LLMReply(text=text.strip(), provider=provider, latency_ms=latency)
        return None

    def _chat_one(self, provider: str, system: str, messages: list[dict[str, str]], max_tokens: int,
                  temperature: float, json_mode: bool) -> str:
        timeout = self.config.llm_timeout_seconds
        if provider in ("groq", "openai"):
            url, key, model = (
                ("https://api.groq.com/openai/v1/chat/completions", self.config.groq_api_key, self.config.groq_model)
                if provider == "groq"
                else ("https://api.openai.com/v1/chat/completions", self.config.openai_api_key, self.config.openai_model)
            )
            body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "system", "content": system}, *messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            resp = self._post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""

        model = self.config.gemini_model.replace("models/", "")
        gen: dict[str, Any] = {"maxOutputTokens": max_tokens, "temperature": temperature}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages
            ],
            "generationConfig": gen,
        }
        resp = self._post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": self.config.gemini_api_key}, json=body, timeout=timeout,
        )
        resp.raise_for_status()
        parts = resp.json()["candidates"][0]["content"].get("parts", [])
        return " ".join(p.get("text", "") for p in parts)

    # ---- vision ----------------------------------------------------------
    def vision(self, prompt: str, image_jpeg: bytes, system: str = "", max_tokens: int = 260) -> LLMReply | None:
        b64 = base64.b64encode(image_jpeg).decode("ascii")
        for provider in self.providers(vision=True):
            started = time.monotonic()
            try:
                text = self._vision_one(provider, system, prompt, b64, max_tokens)
            except Exception as exc:
                logging.warning("Vision provider %s failed: %s", provider, exc)
                self._trip(provider)
                continue
            if text:
                latency = int((time.monotonic() - started) * 1000)
                diagnostics_service.record_ai(f"{provider}-vision", latency)
                return LLMReply(text=text.strip(), provider=provider, latency_ms=latency)
        return None

    def _vision_one(self, provider: str, system: str, prompt: str, b64: str, max_tokens: int) -> str:
        timeout = self.config.llm_timeout_seconds + 6
        if provider == "gemini":
            model = self.config.gemini_model.replace("models/", "")
            body: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [
                    {"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
            }
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            resp = self._post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": self.config.gemini_api_key}, json=body, timeout=timeout,
            )
            resp.raise_for_status()
            parts = resp.json()["candidates"][0]["content"].get("parts", [])
            return " ".join(p.get("text", "") for p in parts)

        url, key, model = (
            ("https://api.groq.com/openai/v1/chat/completions", self.config.groq_api_key, self.config.groq_vision_model)
            if provider == "groq"
            else ("https://api.openai.com/v1/chat/completions", self.config.openai_api_key, self.config.openai_model)
        )
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]})
        resp = self._post(url, headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "messages": messages, "max_tokens": max_tokens}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""
