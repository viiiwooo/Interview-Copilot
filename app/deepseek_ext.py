"""DeepSeek web backend driven by a Chrome extension.

The user installs the ``extension/deepseek-interview`` unpacked extension into
their normal Chrome (no --remote-debugging-port needed) and stays logged into
https://chat.deepseek.com. The content script on that tab connects to this
server over a WebSocket (``ws://localhost:9090/api/deepseek_ws``). When the
pipeline needs an answer it sends ``ask``; the extension types the question
into DeepSeek's composer, watches the assistant reply in the DOM and streams
``text`` frames (full snapshot of the answer so far) back until ``done``.
Snapshots rather than deltas: if DeepSeek re-renders the answer (thinking block
replaced by the answer, markdown re-layout) the server replaces the text
instead of appending a duplicate.

This keeps all browser-side DOM knowledge inside the extension (robust to our
server code) and avoids the fragile CDP port that Chrome only binds when it is
the sole instance owning the profile.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Awaitable, Callable


# Oldest extension build with the Markdown converter (md.js) and snapshots.
# Plain string compare is enough for the x.y.z versions used here.
EXT_MIN_VERSION = "1.2.0"


class DeepSeekExtError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Server-side bridge: one content-script tab at a time.
# ---------------------------------------------------------------------------
class ExtBridge:
    def __init__(self) -> None:
        self._ws = None                     # the connected extension WebSocket
        self._lock = threading.Lock()       # guards _ws and in_q (ask runs on loop thread)
        self.last_seen = 0.0               # monotonic time of last message
        self.hook_ok: bool | None = None   # page_hook.js detected (fast end-of-answer)
        self.version: str = ""             # extension version from its hello frame
        self.in_q: "asyncio.Queue[dict]" = asyncio.Queue()  # parsed frames -> ask()

    def attach(self, ws) -> None:
        """Called when a content-script tab connects; it becomes active."""
        with self._lock:
            self._ws = ws
            self.last_seen = time.monotonic()

    def detach(self, ws) -> bool:
        """Drop the socket if it is still the active one. True if it was."""
        with self._lock:
            if self._ws is ws:
                self._ws = None
                return True
            return False

    @property
    def connected(self) -> bool:
        return self._ws is not None and getattr(self._ws, "client", None) is not None

    def enqueue_incoming(self, raw: str) -> None:
        """Called by the WS endpoint for every frame from the extension.

        Frames are parsed once here and pushed to ``in_q``; ``ask()`` consumes
        them. This keeps a single reader of the socket (the endpoint) so frames
        are never lost to a second competing reader.
        """
        self.last_seen = time.monotonic()
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(msg, dict):
            return
        if msg.get("type") in ("hello", "ping"):
            # liveness only: keeps last_seen fresh, never reaches ask()
            self.hook_ok = bool(msg.get("hook"))
            if msg.get("type") == "hello":
                self.version = str(msg.get("version") or "")
            return
        with self._lock:
            self.in_q.put_nowait(msg)

    async def ask(
        self,
        question: str,
        on_token=None,
        timeout_s: float = 120.0,
        on_replace=None,
    ) -> dict:
        """Send ``ask`` to the extension and collect its reply.

        ``on_token(delta)`` gets appended text. When a snapshot no longer
        extends the previous one, ``on_replace(full_text)`` gets the whole
        answer instead (without it the change is dropped from the stream, but
        the returned text is still correct).

        Returns {"text": ..., "ttft": float, "total": float}. Raises
        DeepSeekExtError if no tab is connected or it does not answer in time.
        """
        with self._lock:
            ws = self._ws
        if ws is None or ws.client is None:
            raise DeepSeekExtError(
                "extension not connected — install the unpacked extension and open "
                "https://chat.deepseek.com in your Chrome"
            )

        # frames left over from an abandoned ask must not leak into this one
        while not self.in_q.empty():
            self.in_q.get_nowait()

        qid = f"q_{time.time()}"
        buf: dict[str, object] = {"text": "", "ttft": None}
        t0 = time.monotonic()
        deadline = t0 + timeout_s

        await ws.send_text(json.dumps(
            {"type": "ask", "qid": qid, "text": question}, ensure_ascii=False))

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeepSeekExtError(f"no answer from extension within {timeout_s:.0f}s")
            try:
                msg = await asyncio.wait_for(self.in_q.get(), timeout=remaining)
            except asyncio.TimeoutError:
                raise DeepSeekExtError(f"no answer from extension within {timeout_s:.0f}s")

            if msg.get("qid") != qid:
                continue  # stale frame from a previous ask
            mtype = msg.get("type")
            if mtype in ("text", "token"):
                old = str(buf["text"])
                if mtype == "text":
                    new = str(msg.get("text", ""))
                else:  # legacy delta frame (extension 1.0)
                    new = old + str(msg.get("text", ""))
                if new == old:
                    continue
                buf["text"] = new
                if buf["ttft"] is None:
                    buf["ttft"] = time.monotonic() - t0
                if new.startswith(old):
                    if on_token:
                        _maybe_await(on_token, new[len(old):])
                elif on_replace:
                    _maybe_await(on_replace, new)
            elif mtype == "done":
                final = str(msg.get("text") or "")
                if final and final != buf["text"]:
                    if final.startswith(str(buf["text"])) and on_token:
                        _maybe_await(on_token, final[len(str(buf["text"])):])
                    elif on_replace:
                        _maybe_await(on_replace, final)
                    buf["text"] = final
                break
            elif mtype == "error":
                raise DeepSeekExtError(str(msg.get("message", "extension error")))

        text = str(buf["text"])
        if not text.strip():
            raise DeepSeekExtError("extension returned an empty answer")
        total = time.monotonic() - t0
        return {"text": text, "ttft": buf["ttft"] or total, "total": total}


def _maybe_await(fn: Callable[[str], Awaitable[None] | None], value: str) -> None:
    """Call on_token; if it returns a coroutine, schedule it onto the loop."""
    res = fn(value)
    if asyncio.iscoroutine(res):
        asyncio.get_running_loop().create_task(res)


# ---------------------------------------------------------------------------
# Availability probe (used by the UI to show whether the extension is live).
# ---------------------------------------------------------------------------
def bridge_available(bridge: ExtBridge, stale_after_s: float = 30.0) -> tuple[bool, str]:
    if not bridge.connected:
        return False, "расширение не подключено"
    age = time.monotonic() - bridge.last_seen
    if age > stale_after_s:
        return False, f"расширение молчит {age:.0f}s (возможно, вкладка закрыта)"
    ver = bridge.version or "старая версия"
    if bridge.version < EXT_MIN_VERSION:
        return True, (f"подключено, но расширение устарело ({ver}): обновите его в "
                      f"chrome://extensions и перезагрузите вкладку DeepSeek")
    if bridge.hook_ok is False:
        return True, f"подключено v{ver} (без page_hook — конец ответа по таймауту, медленнее)"
    return True, f"подключено v{ver}"


def make_llm_result(data: dict):
    """Wrap an ask() result into the shared LLMResult type."""
    from .llm_client import LLMResult

    text = data["text"]
    n_tokens = max(1, len(text))
    return LLMResult(
        text=text,
        ttft=data["ttft"],
        total=data["total"],
        n_tokens=n_tokens,
        tps=n_tokens / max(data["total"], 1e-6),
    )
