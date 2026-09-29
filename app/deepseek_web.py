"""Drive chat.deepseek.com in the user's own Chrome via CDP (remote debugging).

The user launches their normal Chrome with --remote-debugging-port=9222 and is
already logged into https://chat.deepseek.com. We find that tab, focus its
composer, type the question, press Enter, wait for generation to finish, then
read the assistant's last reply from the DOM.

No browser-automation dependency: raw WebSocket over the Chrome DevTools
protocol (urllib + a tiny stdlib-only WS client). This keeps the venv light and
avoids Playwright/Chromium downloads.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path


class CDPError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Minimal WebSocket client (RFC 6455) over a plain socket — just enough for the
# Chrome DevTools protocol: text frames, ping/pong, close. No extensions.
# ---------------------------------------------------------------------------
def _ws_connect(url: str, timeout: float = 10.0):
    """Connect to a ws:// endpoint; return (sock, send_text, recv_text)."""
    import base64
    import os
    import socket
    from urllib.parse import urlparse

    p = urlparse(url)
    host = p.hostname or "localhost"
    port = p.port or 80
    path = p.path + (("?" + p.query) if p.query else "")
    key = base64.b64encode(os.urandom(16)).decode()

    sock = socket.create_connection((host, port), timeout=timeout)
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    )
    sock.sendall(req.encode())

    # read the HTTP handshake response up to \r\n\r\n
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise CDPError("websocket handshake failed (connection closed)")
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode(errors="replace")
    if "101" not in status_line:
        raise CDPError(f"websocket handshake rejected: {status_line}")

    def _recv_exact(n: int) -> bytes:
        out = b""
        while len(out) < n:
            chunk = sock.recv(n - len(out))
            if not chunk:
                raise CDPError("socket closed mid-frame")
            out += chunk
        return out

    def _send_text(text: str) -> None:
        payload = text.encode("utf-8")
        header = bytearray([0x81])  # FIN + text opcode
        n = len(payload)
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header += n.to_bytes(2, "big")
        else:
            header.append(127)
            header += n.to_bytes(8, "big")
        # no masking for client->server is required by Chrome; send unmasked.
        sock.sendall(bytes(header) + payload)

    def _recv_text() -> str:
        b0 = _recv_exact(1)[0]
        opcode = b0 & 0x0F
        masked = (b0 >> 7) & 1  # server frames are never masked, but be safe
        ln = _recv_exact(1)[0]
        if ln == 126:
            length = int.from_bytes(_recv_exact(2), "big")
        elif ln == 127:
            length = int.from_bytes(_recv_exact(8), "big")
        else:
            length = ln
        mask_key = _recv_exact(4) if masked else b""
        data = _recv_exact(length)
        if masked and len(mask_key) == 4:
            data = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))
        if opcode == 0x8:  # close
            raise CDPError("websocket closed by server")
        if opcode == 9:    # ping -> pong
            _send_text("")
            return _recv_text()
        if opcode not in (1, 2):
            return ""  # ignore other control frames
        return data.decode("utf-8", "replace")

    return sock, _send_text, _recv_text


# ---------------------------------------------------------------------------
# CDP session over one page target
# ---------------------------------------------------------------------------
class CDPSession:
    def __init__(self, ws_url: str):
        self.sock, self._send, self._recv = _ws_connect(ws_url)
        self._id = 0

    def call(self, method: str, params: dict | None = None, timeout: float = 15.0) -> dict:
        """Send a CDP command and wait for its response."""
        self._id += 1
        mid = self._id
        payload = json.dumps({"id": mid, "method": method, "params": params or {}})
        self._send(payload)
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise CDPError(f"timeout waiting for {method}")
            resp = json.loads(self._recv())
            if resp.get("id") == mid:
                if "error" in resp:
                    raise CDPError(f"{method}: {resp['error']}")
                return resp.get("result", {})

    def evaluate(self, expression: str, timeout: float = 15.0) -> object:
        """Run JS in the page; return the (JSON-serializable) result value."""
        res = self.call("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": False,
        }, timeout=timeout)
        r = res.get("result", {})
        if "value" in r:
            return r["value"]
        # exception or undefined
        desc = (r.get("description") or "").strip()
        raise CDPError(f"evaluate failed: {desc[:200] or 'no value'}")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def _http_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def find_deepseek_target(cdp_port: int = 9222) -> CDPSession:
    """Locate the chat.deepseek.com tab in the running Chrome and open a CDP session."""
    try:
        targets = _http_json(f"http://127.0.0.1:{cdp_port}/json", timeout=8)
    except Exception as e:
        raise CDPError(
            f"cannot reach Chrome CDP on port {cdp_port} ({e}). "
            f"Run run/launch_chrome_cdp.bat first, then open https://chat.deepseek.com."
        )
    for t in targets:
        if t.get("type") == "page" and "deepseek.com" in (t.get("url") or ""):
            ws = t.get("webSocketDebuggerUrl")
            if not ws:
                continue
            return CDPSession(ws)
    raise CDPError(
        f"no chat.deepseek.com tab found. Open https://chat.deepseek.com in the "
        f"CDP Chrome window (port {cdp_port}) and log in."
    )


# JS: focus the composer, clear it, type text via native setter + input events,
# then press Enter to submit. Returns True if a composer was found.
_JS_SEND = r"""
(async () => {
  const text = __TEXT__;
  // DeepSeek's composer is contenteditable; fall back to textarea/input.
  let box = document.querySelector('[contenteditable="true"]');
  if (!box) box = document.querySelector('textarea, input[type="text"]');
  if (!box) return { ok: false, reason: 'no composer found' };
  box.focus();
  // select-all + delete to clear any draft
  const sel = window.getSelection();
  if (sel && box.isContentEditable) {
    const range = document.createRange();
    range.selectNodeContents(box);
    sel.removeAllRanges(); sel.addRange(range);
    document.execCommand('delete');
  } else {
    box.value = '';
  }
  // insert text so React/controlled inputs register it
  if (box.isContentEditable) {
    document.execCommand('insertText', false, text);
  } else {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype || window.HTMLInputElement.prototype, 'value').set;
    setter.call(box, text);
    box.dispatchEvent(new Event('input', { bubbles: true }));
  }
  // let React commit before we hit Enter
  await new Promise(r => setTimeout(r, 150));
  const opts = { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true };
  box.dispatchEvent(new KeyboardEvent('keydown', opts));
  box.dispatchEvent(new KeyboardEvent('keypress', opts));
  box.dispatchEvent(new KeyboardEvent('keyup', opts));
  return { ok: true };
})()
"""

# JS: assistant answer blocks, same rules as the Chrome extension (content.js):
# .ds-markdown outside the "thinking" block, outermost only. __FROM__ is the
# number of answer blocks that existed before the question was sent, so user
# messages and older answers are never read back as the new answer.
_JS_ANSWERS = r"""
const __answers = () => {
  const thinkSel = '.ds-think-content, [class*="think"], [class*="Think"]';
  for (const sel of ['.ds-markdown', '[class*="markdown"]']) {
    const all = Array.from(document.querySelectorAll(sel)).filter(el =>
      !el.closest('textarea, [contenteditable="true"]') &&
      !(el.parentElement && el.parentElement.closest(sel)));
    if (!all.length) continue;
    const answers = all.filter(el => !el.closest(thinkSel));
    return answers.length ? answers : all;
  }
  return [];
};
"""
_JS_COUNT_ANSWERS = "(() => {" + _JS_ANSWERS + "return __answers().length; })()"
# Same DOM -> Markdown converter as the extension (keeps list markers, drops
# code-block buttons); re-evaluating it is a no-op once it is defined.
_MD_JS = (Path(__file__).resolve().parent.parent / "extension" / "deepseek-interview"
          / "md.js").read_text(encoding="utf-8")
_JS_READ_ANSWER = (_MD_JS + ";(() => {" + _JS_ANSWERS +
                   "return __answers().slice(__FROM__)"
                   ".map(el => window.__icpMd.answerMarkdown(el)).filter(Boolean)"
                   ".join('\\n\\n'); })()")


class DeepSeekWebClient:
    """Send a question to chat.deepseek.com and read the streamed answer."""

    def __init__(self, cdp_port: int = 9222):
        self.cdp_port = cdp_port

    def ask(self, question: str, on_token=None, timeout_s: float = 180.0,
            on_replace=None) -> "LLMResult":
        """Send `question`, stream the answer via on_token(chunk), return LLMResult.

        on_replace(full_text) is called when the answer text is re-rendered and
        no longer extends what was streamed so far.
        """
        from .llm_client import LLMResult

        session = find_deepseek_target(self.cdp_port)
        try:
            # ensure we're on a fresh-ish state; just send.
            n_before = int(session.evaluate(_JS_COUNT_ANSWERS, timeout=10) or 0)
            read_js = _JS_READ_ANSWER.replace("__FROM__", str(n_before))
            js = _JS_SEND.replace("__TEXT__", json.dumps(question))
            r = session.evaluate(js, timeout=20)
            if not isinstance(r, dict) or not r.get("ok"):
                raise CDPError(f"send failed: {r}")

            t0 = time.time()
            ttft = None
            prev = ""
            full = ""
            deadline = time.time() + timeout_s
            stable_since = None
            while time.time() < deadline:
                ans = session.evaluate(read_js, timeout=15) or ""
                if ans and ans != prev:
                    if ttft is None:
                        ttft = time.time() - t0
                    full = ans
                    if ans.startswith(prev):
                        if on_token:
                            on_token(ans[len(prev):])
                    elif on_replace:
                        on_replace(ans)
                    # answer is growing -> reset the "stable" timer
                    stable_since = None
                elif ans and ans == prev:
                    if stable_since is None:
                        stable_since = time.time()
                    elif time.time() - stable_since > 2.5:
                        # no new text for 2.5s -> generation finished
                        break
                if ans:
                    # an empty read (node re-mounting) must not reset prev, or
                    # the next read would be streamed again as one big delta
                    prev = ans
                time.sleep(0.4)

            total = time.time() - t0
            text = full or prev
            if not text.strip():
                raise CDPError("no answer captured from chat.deepseek.com")
            n_tokens = max(1, len(text))  # char-based; good enough for display
            return LLMResult(text=text, ttft=ttft or total, total=total,
                             n_tokens=n_tokens, tps=n_tokens / max(total, 1e-6))
        finally:
            session.close()


def is_available(cdp_port: int = 9222) -> tuple[bool, str]:
    """Cheap check used by the UI to show whether the CDP Chrome is reachable."""
    try:
        targets = _http_json(f"http://127.0.0.1:{cdp_port}/json", timeout=5)
        for t in targets:
            if t.get("type") == "page" and "deepseek.com" in (t.get("url") or ""):
                return True, f"tab found: {t['url'][:60]}"
        return False, f"Chrome CDP up but no deepseek tab (port {cdp_port})"
    except Exception as e:
        return False, f"CDP unreachable on port {cdp_port}: run launch_chrome_cdp.bat ({e.__class__.__name__})"
