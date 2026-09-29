// Interview Copilot — DeepSeek bridge content script.
// Runs on https://chat.deepseek.com/* in the user's own Chrome (already logged in).
// Connects to the copilot server over a WebSocket, receives `ask` frames with a
// question, types it into DeepSeek's composer and streams the assistant reply
// back as `text` frames (full snapshot of the answer so far) until `done`.
//
// Question vs answer: only assistant answer blocks (.ds-markdown, outside the
// "thinking" block) are read, and only blocks that appeared AFTER the question
// was sent. User messages and older answers are never picked up.
//
// Speed: the DOM is watched with a MutationObserver (no fixed polling delay),
// and the end of generation comes from page_hook.js (the completion stream
// closed) instead of waiting for several seconds of unchanged text.

(() => {
  if (window.__icpDeepseekLoaded) return; // guard against double-injection
  window.__icpDeepseekLoaded = true;

  // Same host/port as the copilot web UI (launch_web.bat -> :9090). The content
  // script runs on an https page, but ws:// to localhost is allowed for
  // developer-installed extensions.
  const WS_URL = 'ws://localhost:9090/api/deepseek_ws';
  const RECONNECT_MS = 3000;
  const HEARTBEAT_MS = 10000;
  const FLUSH_MS = 60;           // throttle for sending text snapshots
  const TICK_MS = 100;           // main wait-loop step
  const END_SETTLE_MS = 250;     // after the stream closed: let React render the tail
  const STABLE_MS = 2000;        // fallback end detection when the hook is silent
  const STABLE_WITH_HOOK_MS = 8000; // safety net if the hook saw a start but no end
  const NO_START_MS = 1500;      // no stream after Enter -> try the send button
  const HARD_CAP_MS = 180000;

  const HOOK_SRC = 'icp-deepseek-hook';
  const CS_SRC = 'icp-deepseek-cs';

  // Assistant answer blocks. DeepSeek renders every reply into .ds-markdown;
  // the R1 "thinking" text is also markdown but sits inside a think container.
  const ANSWER_SEL = '.ds-markdown';
  const FALLBACK_ANSWER_SEL = '[class*="markdown"]';
  const THINK_SEL = '.ds-think-content, [class*="think"], [class*="Think"]';

  let ws = null;
  let connected = false;
  let reconnectTimer = null;

  let chain = Promise.resolve();  // asks run strictly one after another
  let current = null;             // { qid, cancelled } of the running ask

  // ------------------------------------------------------- page hook state
  const hook = { alive: false, active: 0, lastStart: 0, lastEnd: 0 };

  window.addEventListener('message', (ev) => {
    if (ev.source !== window || !ev.data || ev.data.source !== HOOK_SRC) return;
    const m = ev.data;
    if (m.type === 'hook_ready') {
      hook.alive = true;
    } else if (m.type === 'stream_start') {
      hook.alive = true;
      hook.active += 1;
      hook.lastStart = Date.now();
    } else if (m.type === 'stream_end') {
      hook.active = Math.max(0, hook.active - 1);
      hook.lastEnd = Date.now();
    }
  });
  window.postMessage({ source: CS_SRC, type: 'hook_ping' }, '*');

  // ------------------------------------------------------------------ helpers

  function log(...a) { console.log('[icp-deepseek]', ...a); }

  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  function send(obj) {
    if (ws && connected && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function isVisible(el) {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }

  // DeepSeek's composer is textarea#chat-input; keep fallbacks for redesigns.
  function findComposer() {
    const byId = document.querySelector('textarea#chat-input');
    if (byId) return byId;
    const tas = Array.from(document.querySelectorAll('textarea')).filter(isVisible);
    if (tas.length) return tas[tas.length - 1];
    const ces = Array.from(document.querySelectorAll('[contenteditable="true"]'))
      .filter((el) => isVisible(el) && el.getBoundingClientRect().top > window.innerHeight * 0.4);
    return ces.length ? ces[ces.length - 1] : null;
  }

  function composerText(el) {
    return el.isContentEditable ? (el.innerText || '') : (el.value || '');
  }

  function setComposerText(el, text) {
    el.focus();
    if (el.isContentEditable) {
      const sel = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(el);
      sel.removeAllRanges();
      sel.addRange(range);
      // insertText replaces the selection and fires the input events React listens for
      document.execCommand('insertText', false, text);
    } else {
      const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
      setter.call(el, text);
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }
  }

  function pressEnter(el) {
    const opts = { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, cancelable: true };
    el.dispatchEvent(new KeyboardEvent('keydown', opts));
    el.dispatchEvent(new KeyboardEvent('keypress', opts));
    el.dispatchEvent(new KeyboardEvent('keyup', opts));
  }

  // Fallback when Enter did not submit: the send button is the last enabled
  // button-like element with an icon near the composer.
  function clickSendButton(composer) {
    let box = composer.parentElement;
    for (let i = 0; i < 6 && box; i++, box = box.parentElement) {
      const btns = Array.from(box.querySelectorAll('[role="button"], button')).filter((b) =>
        isVisible(b) && b.querySelector('svg') &&
        b.getAttribute('aria-disabled') !== 'true' && !b.disabled);
      if (btns.length) {
        btns[btns.length - 1].click();
        return true;
      }
    }
    return false;
  }

  function isThinking(el) {
    return !!el.closest(THINK_SEL);
  }

  // All assistant answer blocks in DOM order (outermost only). Thinking blocks
  // are dropped unless that would drop everything (think selector too broad
  // after a redesign): then the server just replaces thinking with the answer.
  function answerNodes() {
    for (const sel of [ANSWER_SEL, FALLBACK_ANSWER_SEL]) {
      const all = Array.from(document.querySelectorAll(sel)).filter((el) =>
        !el.closest('textarea, [contenteditable="true"]') &&
        !(el.parentElement && el.parentElement.closest(sel)));
      if (!all.length) continue;
      const answers = all.filter((el) => !isThinking(el));
      return answers.length ? answers : all;
    }
    return [];
  }

  // DOM -> clean Markdown lives in md.js (shared with the CDP backend)
  const answerMarkdown = (el) => window.__icpMd.answerMarkdown(el);

  // Text of the answer to our question: every answer block that did not exist
  // before we sent it. The echoed question itself is never an answer block,
  // but guard against a fallback selector matching the user bubble anyway.
  function readNewAnswer(before, question) {
    const parts = [];
    for (const el of answerNodes()) {
      if (before.has(el)) continue;
      const t = answerMarkdown(el);
      if (!t || t === question.trim()) continue;
      parts.push(t);
    }
    return parts.join('\n\n');
  }

  // ------------------------------------------------------------------- ask

  async function waitIdle(maxMs) {
    const until = Date.now() + maxMs;
    while (hook.active > 0 && Date.now() < until) await sleep(TICK_MS);
  }

  async function handleAsk(job) {
    const { qid, text: question } = job;
    current = job;
    try {
      // a previous (cancelled) answer may still be generating: DeepSeek ignores
      // Enter until it finishes
      await waitIdle(20000);
      if (job.cancelled) { send({ type: 'done', qid, text: '' }); return; }

      const composer = findComposer();
      if (!composer) {
        send({ type: 'error', qid, message: 'composer not found — is the DeepSeek chat open?' });
        return;
      }

      const before = new Set(answerNodes());
      const sentAt = Date.now();
      setComposerText(composer, question);
      await sleep(80); // let React commit the value
      pressEnter(composer);

      let last = '';
      let lastChange = Date.now();
      let flushTimer = null;
      let triedButton = false;

      const flush = () => {
        flushTimer = null;
        const t = readNewAnswer(before, question);
        if (t && t !== last) {
          last = t;
          lastChange = Date.now();
          send({ type: 'text', qid, text: t });
        }
      };
      const observer = new MutationObserver(() => {
        if (!flushTimer) flushTimer = setTimeout(flush, FLUSH_MS);
      });
      observer.observe(document.body, { subtree: true, childList: true, characterData: true });

      try {
        while (Date.now() - sentAt < HARD_CAP_MS) {
          await sleep(TICK_MS);
          if (job.cancelled) break;
          flush();

          const started = hook.lastStart >= sentAt;
          const now = Date.now();

          if (!started && !triedButton && now - sentAt > NO_START_MS &&
              composerText(composer).trim()) {
            // Enter did not submit (composer still holds the question)
            triedButton = true;
            if (clickSendButton(composer)) log('Enter ignored, clicked send button');
          }

          if (started && hook.active === 0 && hook.lastEnd >= hook.lastStart) {
            // completion stream closed: give React a moment to render the tail
            if (now - hook.lastEnd >= END_SETTLE_MS) { flush(); break; }
            continue;
          }
          const stableFor = now - lastChange;
          if (last && !started && stableFor >= STABLE_MS) break;          // hook silent
          if (last && started && stableFor >= STABLE_WITH_HOOK_MS) break; // end missed
        }
      } finally {
        observer.disconnect();
        if (flushTimer) clearTimeout(flushTimer);
      }

      if (!last && !job.cancelled) {
        send({ type: 'error', qid, message: 'DeepSeek returned no answer (busy server or changed page layout?)' });
        return;
      }
      send({ type: 'done', qid, text: last });
    } catch (e) {
      send({ type: 'error', qid, message: String((e && e.message) || e) });
    } finally {
      if (current === job) current = null;
    }
  }

  function onMessage(ev) {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'ask') {
      // a newer question supersedes the one being answered
      if (current) current.cancelled = true;
      const job = { qid: msg.qid, text: msg.text || '', cancelled: false };
      chain = chain.then(() => handleAsk(job));
    }
  }

  function connect() {
    log('connecting to', WS_URL);
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {
      connected = true;
      log('connected, page hook', hook.alive ? 'active' : 'NOT detected');
      send({ type: 'hello', hook: hook.alive, url: location.href, version: chrome.runtime.getManifest().version });
    };
    ws.onmessage = onMessage;
    ws.onclose = () => {
      connected = false;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, RECONNECT_MS); // auto-reconnect
    };
    ws.onerror = () => {}; // close handler manages reconnect
  }

  // keeps the server-side "connected" indicator fresh while idle
  setInterval(() => send({ type: 'ping', hook: hook.alive }), HEARTBEAT_MS);

  connect();
})();
