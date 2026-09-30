"""OpenAI-compatible LLM client (llama.cpp or similar on http://127.0.0.1:8080/v1).

Streams the answer so the overlay can start rendering tokens immediately.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator

from .config import LLM_DISABLE_THINKING


@dataclass
class LLMResult:
    text: str
    ttft: float        # time to first content token (seconds)
    total: float       # full generation (seconds)
    n_tokens: int
    tps: float


def api_url(base_url: str, path: str) -> str:
    """Join base_url (with or without a trailing /v1) and an API path like /models."""
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + path


def resolve_model(base_url: str) -> str:
    """First served model id (OpenAI `data[].id`; llama.cpp `models[].name` as fallback)."""
    with urllib.request.urlopen(api_url(base_url, "/models"), timeout=10) as r:
        d = json.load(r)
    if d.get("data"):
        return d["data"][0]["id"]
    return d["models"][0]["name"]


def _stream(base_url: str, model: str, system: str, user: str,
            max_tokens: int, temperature: float, top_p: float,
            disable_thinking: bool) -> Iterator[str]:
    messages = [{"role": "user", "content": user}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if disable_thinking:
        # both knobs: llama.cpp reads chat_template_kwargs, other servers reasoning_effort
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["reasoning_effort"] = "none"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(api_url(base_url, "/chat/completions"),
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
                delta = d["choices"][0].get("delta", {})
                tok = delta.get("content")
                if tok:
                    yield tok
            except Exception:
                pass


def ask(base_url: str, model: str, system: str, user: str,
        max_tokens: int = 220, temperature: float = 0.2, top_p: float = 0.9,
        on_token: Callable[[str], None] | None = None,
        disable_thinking: bool = LLM_DISABLE_THINKING) -> LLMResult:
    """Generate a streamed answer; call on_token for each chunk."""
    t0 = time.time()
    ttft = None
    text = ""
    n = 0
    for tok in _stream(base_url, model, system, user, max_tokens, temperature, top_p,
                       disable_thinking):
        if ttft is None:
            ttft = time.time() - t0
        text += tok
        n += 1
        if on_token:
            on_token(tok)
    total = time.time() - t0
    return LLMResult(text=text, ttft=ttft or total, total=total,
                     n_tokens=n, tps=n / max(total, 1e-6))
