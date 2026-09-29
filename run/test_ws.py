"""WS smoke test: connect to /ws, trigger a file-mode run, log all messages.

Usage: python run/test_ws.py [wav_path] [timeout_s]
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

import websocket  # websocket-client


def main() -> None:
    wav = sys.argv[1] if len(sys.argv) > 1 else "run/test_question.wav"
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0

    messages: list[tuple[float, dict]] = []
    stop = threading.Event()

    def on_message(ws, raw):
        msg = json.loads(raw)
        messages.append((time.time(), msg))
        t = msg.get("type", "?")
        text = msg.get("text", "")
        if t == "answer":
            print(f"  [answer +{time.time()-t0:6.1f}s] {text!r}", flush=True)
        else:
            print(f"  [{t:<8}+{time.time()-t0:6.1f}s] {text!r}", flush=True)

    def on_open(ws):
        print(f"[ws] connected; POST /api/run {wav}", flush=True)
        body = json.dumps({"wav": wav}).encode()
        req = urllib.request.Request(
            "http://localhost:9090/api/run", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                print(f"[api] {r.status} {r.read().decode()}", flush=True)
        except Exception as e:
            print(f"[api] error: {e}", flush=True)

    ws = websocket.WebSocketApp(
        "ws://localhost:9090/ws",
        on_message=on_message,
        on_open=on_open,
        on_error=lambda w, e: print(f"[ws] error: {e}", flush=True),
    )
    t0 = time.time()
    ws_thread = threading.Thread(
        target=lambda: ws.run_forever(ping_interval=10), daemon=True)
    ws_thread.start()

    deadline = time.time() + timeout
    question_seen = False
    while time.time() < deadline:
        if stop.is_set():
            break
        # stop early once we have a question and a finished answer
        types = [m[1].get("type") for m in messages]
        if "question" in types and any(
                m[1].get("type") == "status" and "готово" in m[1].get("text", "")
                for m in messages):
            time.sleep(1.0)
            break
        time.sleep(0.2)

    ws.close()
    stop.set()

    print(f"\n=== summary: {len(messages)} messages in {time.time()-t0:.1f}s ===")
    for t, msg in messages:
        if msg.get("type") in ("status", "question", "latency", "live"):
            print(f"  [{msg['type']:<8}] {msg.get('text', msg)}")
    types = [m[1].get("type") for m in messages]
    print(f"\nquestion received: {'question' in types}")


if __name__ == "__main__":
    main()
