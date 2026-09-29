# Interview Copilot — DeepSeek bridge (Chrome extension)

Types interview questions into **chat.deepseek.com** in your own Chrome and
streams the assistant reply back to the copilot server. No `--remote-debugging-port`
needed, no API key — you stay logged in to DeepSeek as usual.

## Install (one time)

1. Start the copilot web UI so the WebSocket endpoint is up:
   ```bat
   launch_web.bat
   ```
   (the web UI and the extension share one server; extension connects to
   `ws://localhost:9090/api/deepseek_ws`)
2. In Chrome open **chrome://extensions**.
3. Turn on **Developer mode** (top right).
4. Click **Load unpacked** and select this folder
   (`interview-copilot\extension\deepseek-interview`).
5. Open https://chat.deepseek.com in a tab and make sure you are logged in.

## Use it

1. In the copilot web UI, set **LLM (ответы)** → **📡 DeepSeek (расширение)**.
   The indicator under the dropdown turns green when the extension connects.
2. Ask a question (live mic or WAV). When ASR produces text and the backend is
   `deepseek_ext`, the copilot sends it to the extension, which types it into
   DeepSeek's composer, presses Enter and streams the answer back.

## How it works

- `page_hook.js` runs in the page's MAIN world at `document_start`. It wraps
  `fetch`/`XMLHttpRequest` and reports when DeepSeek's completion stream
  (`/api/v0/chat/completion` and similar) starts and ends. It does not parse the
  stream body, so a change in DeepSeek's stream format does not break it.
- `md.js` converts a rendered answer block back to Markdown (keeps list
  bullets/numbers, bold, code, tables; drops copy/download buttons). It is
  shared with the CDP backend (`app/deepseek_web.py`).
- `content.js` runs on `chat.deepseek.com/*`. It opens a WebSocket to the server,
  auto-reconnects every 3 s and sends a `ping` every 10 s.
- On an `ask` frame it:
  1. records which answer blocks (`.ds-markdown` outside the "thinking" block)
     already exist;
  2. types the question into `textarea#chat-input` (fallbacks: any visible
     `textarea`, then a `contenteditable`) and presses Enter; if the composer
     still holds the text after 1.5 s, it clicks the send button;
  3. watches the DOM with a `MutationObserver` and sends `text` frames with the
     full answer so far, read ONLY from answer blocks that appeared after the
     question. User messages and older answers are never read;
  4. sends `done` 250 ms after `page_hook.js` reports the stream closed.
     Without the hook it falls back to 2 s of unchanged text.
- A new `ask` while an answer is still running cancels the old one; the new
  question is typed after DeepSeek finishes generating.

## After updating the extension

The copilot UI shows the connected extension version; an outdated build
(< 1.2.0) is reported as "расширение устарело".

Open **chrome://extensions**, press the reload icon on this extension, then
reload the chat.deepseek.com tab (the page hook must load before the page).

## Notes / fragility

- The answer selector is `.ds-markdown` (fallback `[class*="markdown"]`). If
  DeepSeek redesigns the page and answers stop arriving, update `ANSWER_SEL`,
  `THINK_SEL` and `findComposer()` in `content.js`, and the same selectors in
  `_JS_ANSWERS` in `app/deepseek_web.py` (CDP backend).
- For fast answers turn **DeepThink** and **Search** off in the DeepSeek chat:
  thinking adds many seconds before the first answer token.
- Only one tab is "active" at a time: if you open several deepseek tabs, the last
  one that connects wins. Keep a single deepseek tab for best results.
- The server must be running before the extension can connect (it retries anyway).
