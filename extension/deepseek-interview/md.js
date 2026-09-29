// Interview Copilot — DeepSeek answer DOM -> clean Markdown.
// Shared by the Chrome extension (loaded before content.js) and by the CDP
// backend (app/deepseek_web.py evaluates this file in the page). Exposes
// window.__icpMd.answerMarkdown(el).
//
// innerText of a DeepSeek answer drags in UI chrome (code-block banner with
// the language name, "Копировать"/"Скачать" buttons, KaTeX duplicates) and
// loses list bullets/numbers. Walk the rendered DOM instead and rebuild
// Markdown (with "- " / "1. " markers) that the copilot UI renders itself.

(() => {
  if (window.__icpMd) return;

  const SKIP_SEL = 'button, [role="button"], svg, style, script, textarea, input, ' +
    '[class*="code-block-banner"], [class*="banner"], [class*="toolbar"], ' +
    '[class*="copy"], [class*="download"], .katex-html, [aria-hidden="true"]';
  const BLOCK_TAGS = new Set(['P', 'DIV', 'SECTION', 'ARTICLE', 'H1', 'H2', 'H3', 'H4',
    'H5', 'H6', 'UL', 'OL', 'LI', 'PRE', 'TABLE', 'BLOCKQUOTE', 'HR', 'FIGURE']);

  // Only short chrome is skipped: a class like "*copy*" or aria-hidden on a
  // wrapper that holds the actual answer must never swallow its text.
  function skip(el) {
    try {
      if (!el.matches(SKIP_SEL)) return false;
      if (el.matches('.katex-html, svg, style, script, button, textarea, input')) return true;
      return (el.textContent || '').trim().length < 60;
    } catch { return false; }
  }

  function inlineMd(node) {
    if (node.nodeType === Node.TEXT_NODE) return node.nodeValue.replace(/\s+/g, ' ');
    if (node.nodeType !== Node.ELEMENT_NODE) return '';
    const el = node;
    if (el.classList.contains('katex') || el.classList.contains('katex-display')) {
      const tex = el.querySelector('annotation');
      return tex ? '`' + tex.textContent.trim() + '`' : '';
    }
    if (skip(el)) return '';
    const inner = () => Array.from(el.childNodes).map(inlineMd).join('');
    switch (el.tagName) {
      case 'BR': return '\n';
      case 'STRONG': case 'B': { const t = inner().trim(); return t ? '**' + t + '**' : ''; }
      case 'EM': case 'I': { const t = inner().trim(); return t ? '*' + t + '*' : ''; }
      case 'CODE': { const t = el.textContent; return t ? '`' + t.replace(/`/g, "'") + '`' : ''; }
      case 'SUP': return '';  // citation markers like [1]
      default: return inner();
    }
  }

  function codeLang(pre) {
    const code = pre.querySelector('code') || pre;
    const m = /(?:language|lang)-([\w+#-]+)/.exec(code.className || '');
    if (m) return m[1];
    // DeepSeek puts the language into the (skipped) banner next to <pre>
    const box = pre.parentElement;
    const info = box && box.querySelector('[class*="infostring"], [class*="lang"]');
    return info ? (info.textContent.trim().split(/\s+/)[0] || '') : '';
  }

  function listMd(list, indent) {
    const ordered = list.tagName === 'OL';
    let n = parseInt(list.getAttribute('start') || '1', 10) || 1;
    const out = [];
    for (const li of list.children) {
      if (li.tagName !== 'LI') continue;
      const marker = ordered ? (n++) + '. ' : '- ';
      const own = [];
      const nested = [];
      for (const c of li.childNodes) {
        if (c.nodeType === Node.ELEMENT_NODE && (c.tagName === 'UL' || c.tagName === 'OL')) {
          nested.push(listMd(c, indent + '   '));
        } else if (c.nodeType === Node.ELEMENT_NODE && BLOCK_TAGS.has(c.tagName)) {
          own.push(blockMd(c, indent + '   ').trim());
        } else {
          own.push(inlineMd(c));
        }
      }
      const text = own.join(' ').replace(/[ \t]+/g, ' ').replace(/\n+/g, '\n' + indent + '   ').trim();
      out.push(indent + marker + text);
      for (const s of nested) if (s) out.push(s);
    }
    return out.join('\n');
  }

  function tableMd(table) {
    const rows = Array.from(table.querySelectorAll('tr')).map((tr) =>
      Array.from(tr.children).map((td) => inlineMd(td).replace(/\|/g, '/').trim()));
    if (!rows.length) return '';
    const lines = rows.map((r) => '| ' + r.join(' | ') + ' |');
    lines.splice(1, 0, '| ' + rows[0].map(() => '---').join(' | ') + ' |');
    return lines.join('\n');
  }

  function blockMd(el, indent) {
    indent = indent || '';
    const out = [];
    let inl = '';
    const flush = () => {
      const t = inl.replace(/[ \t]+/g, ' ').replace(/ *\n */g, '\n').trim();
      if (t) out.push(t);
      inl = '';
    };
    for (const c of el.childNodes) {
      if (c.nodeType !== Node.ELEMENT_NODE || !BLOCK_TAGS.has(c.tagName)) {
        inl += inlineMd(c);
        continue;
      }
      if (skip(c)) continue;
      flush();
      const tag = c.tagName;
      if (/^H[1-6]$/.test(tag)) {
        const t = inlineMd(c).trim();
        if (t) out.push('#'.repeat(Math.min(+tag[1], 4)) + ' ' + t);
      } else if (tag === 'UL' || tag === 'OL') {
        out.push(listMd(c, indent));
      } else if (tag === 'PRE') {
        const code = (c.querySelector('code') || c).textContent.replace(/\n+$/, '');
        out.push('```' + codeLang(c) + '\n' + code + '\n```');
      } else if (tag === 'TABLE') {
        out.push(tableMd(c));
      } else if (tag === 'BLOCKQUOTE') {
        out.push(blockMd(c).split('\n').map((l) => '> ' + l).join('\n'));
      } else if (tag === 'HR') {
        out.push('---');
      } else {
        const t = blockMd(c, indent);
        if (t) out.push(t);
      }
    }
    flush();
    return out.filter(Boolean).join('\n\n');
  }

  // innerText drops list bullets/numbers; used only if the walk found nothing
  function answerMarkdown(el) {
    try {
      const md = blockMd(el).replace(/\n{3,}/g, '\n\n').trim();
      if (md) return md;
    } catch (e) {
      console.warn('[icp-deepseek] markdown conversion failed, using innerText', e);
    }
    return (el.innerText || '').trim();
  }

  window.__icpMd = { answerMarkdown, version: '1.2.0' };
})();
