// Code rendering for the problem browser: HTML escaping + a small WebPPL
// syntax highlighter. (The atom-era output/chart renderers died with the
// P2 problem-centric rewrite.)

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ─── WebPPL syntax highlighter ──────────────────────────────────────────────

const WEBPPL_KW = new Set([
  'var','function','return','if','else','for','while','true','false','null','undefined','new',
]);
const WEBPPL_INFER = new Set([
  'Infer','Enumerate','MCMC','SMC','rejection','sample','factor','observe','condition',
  'expectation','flip','uniform','uniformDraw','gaussian','beta','dirichlet','categorical',
  'discrete','mem','mapData','map','map2','reduce','Categorical','Bernoulli','Binomial',
  'Gaussian','Beta','Dirichlet','Vector','Math','repeat',
]);

interface Tok { t: string; v: string; }

function tokenize(src: string): Tok[] {
  const tokens: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === '/' && src[i + 1] === '/') {
      const j = src.indexOf('\n', i);
      const end = j === -1 ? src.length : j;
      tokens.push({ t: 'cm', v: src.slice(i, end) });
      i = end;
      continue;
    }
    if (c === '/' && src[i + 1] === '*') {
      const j = src.indexOf('*/', i + 2);
      const end = j === -1 ? src.length : j + 2;
      tokens.push({ t: 'cm', v: src.slice(i, end) });
      i = end;
      continue;
    }
    if (c === "'" || c === '"' || c === '`') {
      const q = c;
      let j = i + 1;
      while (j < src.length && src[j] !== q) {
        if (src[j] === '\\') j += 2;
        else j++;
      }
      tokens.push({ t: 's', v: src.slice(i, j + 1) });
      i = j + 1;
      continue;
    }
    if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(src[i + 1] ?? ''))) {
      let j = i;
      while (j < src.length && /[0-9.eE+-]/.test(src[j])) j++;
      tokens.push({ t: 'n', v: src.slice(i, j) });
      i = j;
      continue;
    }
    if (/[A-Za-z_$]/.test(c)) {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_$]/.test(src[j])) j++;
      const v = src.slice(i, j);
      let t = 'i';
      if (WEBPPL_KW.has(v)) t = 'k';
      else if (WEBPPL_INFER.has(v)) t = 'b';
      else if (src[j] === '(') t = 'f';
      tokens.push({ t, v });
      i = j;
      continue;
    }
    if (/[{}()\[\],;:]/.test(c)) {
      tokens.push({ t: 'p', v: c });
      i++;
      continue;
    }
    if (/[+\-*/<>=!&|?]/.test(c)) {
      let j = i;
      while (j < src.length && /[+\-*/<>=!&|?]/.test(src[j])) j++;
      tokens.push({ t: 'o', v: src.slice(i, j) });
      i = j;
      continue;
    }
    tokens.push({ t: 'w', v: c });
    i++;
  }
  return tokens;
}

function highlightLines(src: string): Tok[][] {
  const tokens = tokenize(src);
  const lines: Tok[][] = [[]];
  for (const tok of tokens) {
    const segs = tok.v.split('\n');
    segs.forEach((seg, i) => {
      if (i > 0) lines.push([]);
      if (seg) lines[lines.length - 1].push({ t: tok.t, v: seg });
    });
  }
  return lines;
}

export function renderCode(code: string, lang = 'webppl'): string {
  const lines = highlightLines(code || '');
  const body = lines.map((toks, i) => {
    const content = toks.length === 0
      ? '​'
      : toks.map((tk) => `<span class="tok-${tk.t}">${escapeHtml(tk.v)}</span>`).join('');
    return `<div class="code-line"><span class="code-ln">${i + 1}</span><span class="code-content">${content}</span></div>`;
  }).join('');
  return (
    `<div class="code">` +
    `<div class="code-lang">${escapeHtml(lang)}</div>` +
    `<pre class="code-body">${body}</pre>` +
    `</div>`
  );
}
