// Interview Copilot — page-world hook (runs in the MAIN world of chat.deepseek.com).
// Wraps window.fetch and XMLHttpRequest to detect when DeepSeek's completion
// stream starts and ends. The isolated content script cannot see page fetches,
// so this script reports them via window.postMessage. Only the lifecycle is
// reported (no parsing of the SSE body), so a change in DeepSeek's stream
// format does not break it; the answer text itself is still read from the DOM.

(() => {
  if (window.__icpHookLoaded) return;
  window.__icpHookLoaded = true;

  const SRC = 'icp-deepseek-hook';
  // /api/v0/chat/completion, /api/v0/chat/regenerate, /api/v0/chat/resume_stream ...
  const STREAM_RE = /\/api\/v\d+\/chat\/(completion|regenerate|resume|continue|edit)/i;

  function post(type, extra) {
    window.postMessage(Object.assign({ source: SRC, type, t: Date.now() }, extra || {}), '*');
  }

  function urlOf(input) {
    if (typeof input === 'string') return input;
    if (input && typeof input.url === 'string') return input.url;
    return String(input || '');
  }

  let seq = 0;

  const origFetch = window.fetch;
  window.fetch = async function (input, init) {
    const url = urlOf(input);
    if (!STREAM_RE.test(url)) return origFetch.apply(this, arguments);
    const id = ++seq;
    post('stream_start', { id, url });
    let resp;
    try {
      resp = await origFetch.apply(this, arguments);
    } catch (e) {
      post('stream_end', { id, error: String(e) });
      throw e;
    }
    try {
      // read a tee'd copy to the end: the page keeps its own body untouched
      const reader = resp.clone().body.getReader();
      (async () => {
        try {
          for (;;) {
            const { done } = await reader.read();
            if (done) break;
          }
          post('stream_end', { id });
        } catch (e) {
          post('stream_end', { id, error: String(e) });
        }
      })();
    } catch (e) {
      post('stream_end', { id, error: String(e) });
    }
    return resp;
  };

  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url) {
    this.__icpUrl = String(url || '');
    return origOpen.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function () {
    if (STREAM_RE.test(this.__icpUrl || '')) {
      const id = ++seq;
      post('stream_start', { id, url: this.__icpUrl });
      this.addEventListener('loadend', () => post('stream_end', { id }));
    }
    return origSend.apply(this, arguments);
  };

  // the content script may load after us and ask whether the hook is alive
  window.addEventListener('message', (ev) => {
    if (ev.source === window && ev.data && ev.data.source === 'icp-deepseek-cs'
        && ev.data.type === 'hook_ping') {
      post('hook_ready');
    }
  });
  post('hook_ready');
})();
