import type { AnswerShape, RenderOutput } from './types';

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ─── Markdown ───────────────────────────────────────────────────────────────
// Tiny markdown handler: ``` fenced blocks, `inline code`, **bold**, paragraphs.

interface MdBlock { type: 'p' | 'code'; content: string; lang?: string; }

function parseMarkdown(src: string): MdBlock[] {
  const parts: MdBlock[] = [];
  const lines = src.split('\n');
  let buf: string[] = [];
  let inFence = false;
  let fenceLang = '';
  let fenceBuf: string[] = [];
  const flushPara = () => {
    if (!buf.length) return;
    parts.push({ type: 'p', content: buf.join('\n') });
    buf = [];
  };
  for (const line of lines) {
    if (inFence) {
      if (line.startsWith('```')) {
        parts.push({ type: 'code', lang: fenceLang, content: fenceBuf.join('\n') });
        inFence = false; fenceBuf = [];
      } else {
        fenceBuf.push(line);
      }
      continue;
    }
    if (line.startsWith('```')) {
      flushPara();
      inFence = true;
      fenceLang = line.slice(3).trim();
      continue;
    }
    if (line.trim() === '') { flushPara(); continue; }
    buf.push(line);
  }
  flushPara();
  return parts;
}

function inlineMd(text: string): string {
  const escaped = escapeHtml(text);
  const out: string[] = [];
  let i = 0;
  while (i < escaped.length) {
    if (escaped[i] === '`') {
      const j = escaped.indexOf('`', i + 1);
      if (j > 0) {
        out.push(`<code class="md-inline-code">${escaped.slice(i + 1, j)}</code>`);
        i = j + 1;
        continue;
      }
    }
    if (escaped[i] === '*' && escaped[i + 1] === '*') {
      const j = escaped.indexOf('**', i + 2);
      if (j > 0) {
        out.push(`<strong>${escaped.slice(i + 2, j)}</strong>`);
        i = j + 2;
        continue;
      }
    }
    const next = escaped.indexOf('`', i);
    const nextB = escaped.indexOf('**', i);
    let end = escaped.length;
    if (next > -1) end = Math.min(end, next);
    if (nextB > -1) end = Math.min(end, nextB);
    out.push(escaped.slice(i, end));
    i = end;
  }
  return out.join('');
}

export function renderMarkdown(src: string): string {
  const blocks = parseMarkdown(src);
  return '<div class="md">' + blocks.map((b) =>
    b.type === 'code'
      ? renderCode(b.content, b.lang || 'webppl')
      : `<p>${inlineMd(b.content)}</p>`
  ).join('') + '</div>';
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

// ─── Output normalization ───────────────────────────────────────────────────
// Take the executor's raw answer (the JSON stored as `groundtruth_output` or
// `evaluation.gen.answer`) and convert it to RenderOutput.

export function normalizeOutput(answer: unknown, shape: AnswerShape): RenderOutput {
  if (answer == null) return null;
  if (shape === 'distribution') {
    if (typeof answer === 'object' && answer !== null) {
      const a = answer as Record<string, unknown>;
      if (a.__kind === 'distribution' && Array.isArray(a.support) && Array.isArray(a.probs)) {
        return {
          kind: 'distribution',
          support: (a.support as unknown[]).map(stringifyKey),
          probs: a.probs as number[],
        };
      }
    }
    return { kind: 'value', value: answer };
  }
  if (shape === 'samples') {
    if (Array.isArray(answer)) {
      // Aggregate samples → {support, counts}
      const counts = new Map<string, number>();
      for (const item of answer) {
        const key = stringifyKey(item);
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
      const support = Array.from(counts.keys());
      const cs = support.map((s) => counts.get(s)!);
      return { kind: 'samples', support, counts: cs };
    }
    return { kind: 'value', value: answer };
  }
  if (shape === 'value') {
    return { kind: 'value', value: answer };
  }
  if (typeof shape === 'object' && shape !== null && 'record' in shape) {
    if (typeof answer === 'object' && answer !== null && !Array.isArray(answer)) {
      const fields: Record<string, RenderOutput> = {};
      const subShape = (shape as { record: Record<string, AnswerShape> }).record;
      for (const [k, sub] of Object.entries(subShape)) {
        const inner = normalizeOutput((answer as Record<string, unknown>)[k], sub);
        if (inner) fields[k] = inner;
      }
      return { kind: 'record', fields };
    }
    return { kind: 'value', value: answer };
  }
  return { kind: 'value', value: answer };
}

function stringifyKey(v: unknown): string {
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return JSON.stringify(v);
}

// ─── Mirrored bar chart (SVG) ───────────────────────────────────────────────
// Renders two distributions overlaid: A goes up from mid-axis, B goes down.

interface ChartSeries { kind: 'distribution' | 'samples'; support: string[]; probs?: number[]; counts?: number[]; }

function seriesProbs(s: ChartSeries | null | undefined, support: string[]): number[] {
  if (!s) return support.map(() => 0);
  return support.map((label) => {
    const idx = s.support.indexOf(label);
    if (idx === -1) return 0;
    if (s.kind === 'samples') {
      const total = (s.counts ?? []).reduce((a, b) => a + b, 0) || 1;
      return ((s.counts ?? [])[idx] ?? 0) / total;
    }
    return (s.probs ?? [])[idx] ?? 0;
  });
}

export function renderChart(opts: {
  a: ChartSeries | null;
  b: ChartSeries | null;
  labelA: string;
  labelB: string;
  maxBars?: number;
}): string {
  const { a, b, labelA, labelB, maxBars = 48 } = opts;
  const supportSet = new Set<string>();
  for (const s of a?.support ?? []) supportSet.add(s);
  for (const s of b?.support ?? []) supportSet.add(s);
  let support = Array.from(supportSet);
  if (support.length === 0) {
    return '<div class="out-empty">(no distribution)</div>';
  }

  // Detect numeric support: every label parses as a finite number.
  const numericValues = support.map((s) => Number(s));
  const numeric = numericValues.every((v) => Number.isFinite(v));
  const isContinuous = numeric && support.length >= 14;

  let aProbs = seriesProbs(a, support);
  let bProbs = seriesProbs(b, support);
  let truncated = 0;
  // For categorical, top-N by max(a, b) prob keeps the chart readable.
  // For continuous (already binned upstream), don't truncate; render all bins.
  if (!isContinuous && support.length > maxBars) {
    const ranked = support.map((s, i) => ({ s, p: Math.max(aProbs[i], bProbs[i]) }))
      .sort((x, y) => y.p - x.p)
      .slice(0, maxBars)
      .map(({ s }) => s);
    truncated = support.length - ranked.length;
    support = ranked;
    aProbs = seriesProbs(a, support);
    bProbs = seriesProbs(b, support);
  }
  // Numeric → sort by value so bars are in x-axis order.
  if (numeric) {
    const nums = support.map((s) => Number(s));
    const idx = support.map((_, i) => i).sort((x, y) => nums[x] - nums[y]);
    support = idx.map((i) => support[i]);
    aProbs = idx.map((i) => aProbs[i]);
    bProbs = idx.map((i) => bProbs[i]);
  }
  const maxP = Math.max(0.01, ...aProbs, ...bProbs);

  const w = 640, h = 240;
  const padL = 40, padR = 16, padT = 22, padB = 30;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;
  const midY = padT + innerH / 2;
  const halfH = innerH / 2 - 4;
  const ticks = [0, maxP / 2, maxP];
  const fmtY = (v: number) => v === 0 ? '0' : maxP >= 0.1 ? v.toFixed(2) : maxP >= 0.01 ? v.toFixed(3) : v.toExponential(1);
  const fmtNum = (n: number) => {
    if (Number.isInteger(n)) return String(n);
    const abs = Math.abs(n);
    if (abs >= 100) return n.toFixed(0);
    if (abs >= 10) return n.toFixed(1);
    return n.toFixed(2);
  };

  const gridLines: string[] = [];
  for (let i = 0; i < ticks.length; i++) {
    const t = ticks[i];
    const yA = midY - (t / maxP) * halfH;
    const yB = midY + (t / maxP) * halfH;
    gridLines.push(
      `<line x1="${padL}" y1="${yA}" x2="${w - padR}" y2="${yA}" class="chart-grid"/>` +
      `<text x="${padL - 6}" y="${yA}" class="chart-yt" text-anchor="end" dominant-baseline="middle">${fmtY(t)}</text>`
    );
    if (i > 0) {
      gridLines.push(
        `<line x1="${padL}" y1="${yB}" x2="${w - padR}" y2="${yB}" class="chart-grid"/>` +
        `<text x="${padL - 6}" y="${yB}" class="chart-yt" text-anchor="end" dominant-baseline="middle">${fmtY(t)}</text>`
      );
    }
  }
  const midAxis = `<line x1="${padL}" y1="${midY}" x2="${w - padR}" y2="${midY}" class="chart-axis"/>`;

  const body = isContinuous
    ? renderNumericBars({ support, aProbs, bProbs, maxP, padL, padR, innerW, midY, halfH, h, w, fmtNum })
    : renderCategoricalBars({ support, aProbs, bProbs, maxP, padL, innerW, midY, halfH, h });

  const metaBits: string[] = [];
  metaBits.push(`${support.length} bin${support.length === 1 ? '' : 's'}`);
  if (numeric && support.length > 1) {
    const xs = support.map(Number);
    metaBits.push(`${fmtNum(Math.min(...xs))} … ${fmtNum(Math.max(...xs))}`);
  }
  if (truncated > 0) metaBits.push(`top ${support.length} of ${support.length + truncated}`);
  const metaStr = metaBits.join(' · ');

  return (
    `<div class="chart">` +
    `<div class="chart-legend">` +
    `<span class="chart-legend-item chart-legend-a"><span class="chart-legend-swatch"></span> ${escapeHtml(labelA)}</span>` +
    `<span class="chart-legend-item chart-legend-b"><span class="chart-legend-swatch"></span> ${escapeHtml(labelB)}</span>` +
    `<span class="chart-legend-meta">${escapeHtml(metaStr)}</span>` +
    `</div>` +
    `<svg viewBox="0 0 ${w} ${h}" class="chart-svg" role="img" aria-label="distribution overlay">` +
    gridLines.join('') + midAxis + body +
    `</svg>` +
    `</div>`
  );
}

// "Nice" axis step — round 1/2/5 × 10^k.
function niceTick(step: number): number {
  if (step <= 0) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(step)));
  const norm = step / pow;
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return nice * pow;
}

interface NumericBarsArgs {
  support: string[]; aProbs: number[]; bProbs: number[]; maxP: number;
  padL: number; padR: number; innerW: number; midY: number; halfH: number;
  h: number; w: number; fmtNum: (n: number) => string;
}

function renderNumericBars(args: NumericBarsArgs): string {
  const { support, aProbs, bProbs, maxP, padL, padR, innerW, midY, halfH, h, w, fmtNum } = args;
  const xs = support.map(Number);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const range = xMax - xMin || 1;
  const xPos = (v: number) => padL + ((v - xMin) / range) * innerW;
  const yA = (p: number) => midY - (p / maxP) * halfH;
  const yB = (p: number) => midY + (p / maxP) * halfH;

  // Half-pitch on each side: area covers the bin it represents.
  const pitchHalf = (i: number): number => {
    if (xs.length === 1) return innerW / 2;
    if (i === 0) return (xPos(xs[1]) - xPos(xs[0])) / 2;
    if (i === xs.length - 1) return (xPos(xs[i]) - xPos(xs[i - 1])) / 2;
    return Math.max(xPos(xs[i]) - xPos(xs[i - 1]), xPos(xs[i + 1]) - xPos(xs[i])) / 2;
  };

  const buildArea = (probs: number[], yFn: (p: number) => number): string => {
    let d = '';
    xs.forEach((x, i) => {
      const half = pitchHalf(i);
      const xL = xPos(x) - half;
      const xR = xPos(x) + half;
      const y = yFn(probs[i]);
      if (i === 0) d += `M ${xL.toFixed(2)} ${midY} L ${xL.toFixed(2)} ${y.toFixed(2)} `;
      d += `L ${xR.toFixed(2)} ${y.toFixed(2)} `;
      if (i === xs.length - 1) d += `L ${xR.toFixed(2)} ${midY} Z`;
      else {
        const nextHalf = pitchHalf(i + 1);
        const xN = xPos(xs[i + 1]) - nextHalf;
        const yN = yFn(probs[i + 1]);
        d += `L ${xN.toFixed(2)} ${y.toFixed(2)} L ${xN.toFixed(2)} ${yN.toFixed(2)} `;
      }
    });
    return d;
  };

  const buildLine = (probs: number[], yFn: (p: number) => number): string => {
    let d = '';
    xs.forEach((x, i) => {
      const half = pitchHalf(i);
      const xL = xPos(x) - half;
      const xR = xPos(x) + half;
      const y = yFn(probs[i]);
      if (i === 0) d += `M ${xL.toFixed(2)} ${y.toFixed(2)} `;
      d += `L ${xR.toFixed(2)} ${y.toFixed(2)} `;
      if (i < xs.length - 1) {
        const nextHalf = pitchHalf(i + 1);
        const xN = xPos(xs[i + 1]) - nextHalf;
        const yN = yFn(probs[i + 1]);
        d += `L ${xN.toFixed(2)} ${y.toFixed(2)} L ${xN.toFixed(2)} ${yN.toFixed(2)} `;
      }
    });
    return d;
  };

  // "Nice" x-ticks within [xMin, xMax].
  const tickCount = Math.min(7, Math.max(3, Math.floor(innerW / 90)));
  const step = niceTick(range / (tickCount - 1));
  const start = Math.ceil(xMin / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= xMax + step / 1000; v += step) ticks.push(Number(v.toFixed(10)));

  // Mode markers
  const modeIdx = (probs: number[]): number => {
    let m = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[m]) m = i;
    return probs[m] > 0 ? m : -1;
  };
  const modeA = modeIdx(aProbs);
  const modeB = modeIdx(bProbs);
  const renderMode = (i: number, probs: number[], yFn: (p: number) => number, side: 'a' | 'b') => {
    if (i < 0) return '';
    const x = xPos(xs[i]);
    const y = yFn(probs[i]);
    const dy = side === 'a' ? -8 : 14;
    const label = `${fmtNum(xs[i])} · ${probs[i].toFixed(3)}`;
    return (
      `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="2.5" class="chart-mode chart-mode-${side}"/>` +
      `<text x="${x.toFixed(2)}" y="${(y + dy).toFixed(2)}" class="chart-mode-label chart-mode-label-${side}" text-anchor="middle">${escapeHtml(label)}</text>`
    );
  };

  const tickMarks = ticks.map((v) => {
    const x = xPos(v);
    return (
      `<line x1="${x.toFixed(2)}" y1="${h - 28}" x2="${x.toFixed(2)}" y2="${h - 24}" class="chart-axis"/>` +
      `<text x="${x.toFixed(2)}" y="${h - 10}" class="chart-xt" text-anchor="middle">${escapeHtml(fmtNum(v))}</text>`
    );
  }).join('');

  const hits = xs.map((x, i) => {
    const half = pitchHalf(i);
    const xL = xPos(x) - half;
    const delta = aProbs[i] - bProbs[i];
    const title = `x = ${fmtNum(x)}\nA = ${aProbs[i].toFixed(4)}\nB = ${bProbs[i].toFixed(4)}\nΔ = ${delta.toFixed(4)}`;
    return (
      `<rect x="${xL.toFixed(2)}" y="${(midY - halfH).toFixed(2)}" width="${(half * 2).toFixed(2)}" height="${(halfH * 2).toFixed(2)}" fill="transparent" class="chart-hover">` +
      `<title>${escapeHtml(title)}</title>` +
      `</rect>`
    );
  }).join('');

  return (
    tickMarks +
    `<path d="${buildArea(aProbs, yA)}" class="chart-area chart-area-a"/>` +
    `<path d="${buildArea(bProbs, yB)}" class="chart-area chart-area-b"/>` +
    `<path d="${buildLine(aProbs, yA)}" class="chart-line chart-line-a"/>` +
    `<path d="${buildLine(bProbs, yB)}" class="chart-line chart-line-b"/>` +
    renderMode(modeA, aProbs, yA, 'a') +
    renderMode(modeB, bProbs, yB, 'b') +
    hits
  );
}

interface CategoricalBarsArgs {
  support: string[]; aProbs: number[]; bProbs: number[]; maxP: number;
  padL: number; innerW: number; midY: number; halfH: number; h: number;
}

function renderCategoricalBars(args: CategoricalBarsArgs): string {
  const { support, aProbs, bProbs, maxP, padL, innerW, midY, halfH, h } = args;
  const colW = innerW / Math.max(1, support.length);
  const barW = Math.max(4, Math.min(40, colW * 0.72));
  const showValueLabels = barW >= 28;
  const xLabelStride = Math.max(1, Math.ceil(40 / colW));

  return support.map((s, i) => {
    const x = padL + i * colW + (colW - barW) / 2;
    const ha = (aProbs[i] / maxP) * halfH;
    const hb = (bProbs[i] / maxP) * halfH;
    const sEsc = escapeHtml(s);
    const showXLabel = i % xLabelStride === 0;
    const labelsBlock = showValueLabels
      ? (aProbs[i] > 0.005 ? `<text x="${(x + barW / 2).toFixed(2)}" y="${(midY - ha - 4).toFixed(2)}" class="chart-val chart-val-a" text-anchor="middle">${aProbs[i].toFixed(2)}</text>` : '') +
        (bProbs[i] > 0.005 ? `<text x="${(x + barW / 2).toFixed(2)}" y="${(midY + hb + 12).toFixed(2)}" class="chart-val chart-val-b" text-anchor="middle">${bProbs[i].toFixed(2)}</text>` : '')
      : '';
    const xLabel = showXLabel
      ? `<text x="${(x + barW / 2).toFixed(2)}" y="${(h - 8).toFixed(2)}" class="chart-xt" text-anchor="middle">${sEsc}</text>`
      : '';
    const tipTitle = `${s}\nA = ${aProbs[i].toFixed(3)}\nB = ${bProbs[i].toFixed(3)}`;
    return (
      `<g>` +
      `<rect x="${x.toFixed(2)}" y="${(midY - ha).toFixed(2)}" width="${barW.toFixed(2)}" height="${ha.toFixed(2)}" class="chart-bar chart-bar-a"><title>${escapeHtml(tipTitle)}</title></rect>` +
      `<rect x="${x.toFixed(2)}" y="${midY.toFixed(2)}" width="${barW.toFixed(2)}" height="${hb.toFixed(2)}" class="chart-bar chart-bar-b"><title>${escapeHtml(tipTitle)}</title></rect>` +
      labelsBlock + xLabel +
      `</g>`
    );
  }).join('');
}

// ─── Value / record renderers ───────────────────────────────────────────────

export function renderValueOutput(output: RenderOutput, fallback = '(no output)'): string {
  if (!output) return `<div class="out-empty">${escapeHtml(fallback)}</div>`;
  if (output.kind === 'value') {
    const v = output.value;
    let txt: string;
    if (Array.isArray(v)) {
      txt = '[' + v.map((x) =>
        typeof x === 'number' ? x.toFixed(4) : JSON.stringify(x),
      ).join(', ') + ']';
    } else if (typeof v === 'number') {
      txt = v.toFixed(4);
    } else {
      txt = JSON.stringify(v, null, 2);
    }
    return `<div class="out-value"><pre class="out-value-pre">${escapeHtml(txt)}</pre></div>`;
  }
  if (output.kind === 'record') {
    const rows = Object.entries(output.fields).map(([k, v]) => {
      let val: string;
      if (v && v.kind === 'value') {
        val = typeof v.value === 'number' ? v.value.toFixed(4) : JSON.stringify(v.value);
      } else if (v && v.kind === 'distribution') {
        val = `dist(${v.support.length})`;
      } else {
        val = '...';
      }
      return (
        `<div class="out-record-row">` +
        `<span class="out-record-key">${escapeHtml(k)}</span>` +
        `<span class="out-record-eq">=</span>` +
        `<span class="out-record-val">${escapeHtml(val)}</span>` +
        `</div>`
      );
    }).join('');
    return `<div class="out-record">${rows}</div>`;
  }
  return `<div class="out-empty">${escapeHtml(fallback)}</div>`;
}

export function shapeLabel(shape: AnswerShape): string {
  if (shape && typeof shape === 'object' && 'record' in shape) {
    return `record(${Object.keys((shape as { record: Record<string, AnswerShape> }).record).join(', ')})`;
  }
  return String(shape);
}

export function atomDomId(atomId: string): string {
  return 'atom-' + atomId.replace(/\//g, '--').replace(/ /g, '_');
}

export function isDistLike(o: RenderOutput): o is { kind: 'distribution'; support: string[]; probs: number[] } | { kind: 'samples'; support: string[]; counts: number[] } {
  return !!o && (o.kind === 'distribution' || o.kind === 'samples');
}

/** Cap a distribution / samples output to its top-N most probable values.
 *  LM-generated distributions sometimes carry tens of thousands of support items
 *  (e.g. when the model produced a degenerate distribution); we never need to
 *  ship more than a few dozen to render the chart. */
export function truncateOutput(o: RenderOutput, n = 48): RenderOutput {
  if (!o) return o;
  if (o.kind === 'distribution') {
    if (o.support.length <= n) return o;
    const idx = o.support.map((_, i) => i).sort((a, b) => o.probs[b] - o.probs[a]).slice(0, n);
    return { kind: 'distribution', support: idx.map((i) => o.support[i]), probs: idx.map((i) => o.probs[i]) };
  }
  if (o.kind === 'samples') {
    if (o.support.length <= n) return o;
    const idx = o.support.map((_, i) => i).sort((a, b) => o.counts[b] - o.counts[a]).slice(0, n);
    return { kind: 'samples', support: idx.map((i) => o.support[i]), counts: idx.map((i) => o.counts[i]) };
  }
  if (o.kind === 'record') {
    return { kind: 'record', fields: Object.fromEntries(Object.entries(o.fields).map(([k, v]) => [k, truncateOutput(v, n)])) };
  }
  return o;
}

/** Heuristic: does this output look like a continuous distribution that
 *  needs binning? (Many distinct numeric support values, low per-bucket mass.) */
function looksContinuous(o: RenderOutput): boolean {
  if (!o || (o.kind !== 'distribution' && o.kind !== 'samples')) return false;
  if (o.support.length < 16) return false;
  let numeric = 0;
  let nonInt = 0;
  const sample = o.support.length > 100 ? o.support.slice(0, 100) : o.support;
  for (const s of sample) {
    const x = Number(s);
    if (!Number.isFinite(x)) continue;
    numeric++;
    if (!Number.isInteger(x)) nonInt++;
  }
  if (numeric / sample.length < 0.9) return false;
  // Mostly non-integer values, OR a *lot* of distinct integers → continuous.
  return nonInt / numeric > 0.5 || o.support.length > 64;
}

/** Bin a set of likely-continuous distributions/samples into shared bins.
 *  Keeps charts comparable: all series get the same bin edges from the union
 *  range. nBins ≈ 24 fits the 640px chart cleanly. */
function binShared(outputs: RenderOutput[], nBins = 24): RenderOutput[] {
  let min = Infinity, max = -Infinity;
  for (const o of outputs) {
    if (!o || (o.kind !== 'distribution' && o.kind !== 'samples')) continue;
    for (const s of o.support) {
      const x = Number(s);
      if (!Number.isFinite(x)) continue;
      if (x < min) min = x;
      if (x > max) max = x;
    }
  }
  if (!Number.isFinite(min) || min === max) return outputs;
  const step = (max - min) / nBins;
  const labels: string[] = [];
  const decimals = pickDecimals(step);
  for (let i = 0; i < nBins; i++) {
    const mid = min + step * (i + 0.5);
    labels.push(mid.toFixed(decimals));
  }
  return outputs.map((o) => {
    if (!o) return o;
    if (o.kind !== 'distribution' && o.kind !== 'samples') return o;
    const bins = new Array(nBins).fill(0);
    const weights: number[] = o.kind === 'distribution'
      ? o.probs
      : (() => {
          const total = o.counts.reduce((a, b) => a + b, 0) || 1;
          return o.counts.map((c) => c / total);
        })();
    for (let i = 0; i < o.support.length; i++) {
      const x = Number(o.support[i]);
      if (!Number.isFinite(x)) continue;
      let b = Math.floor((x - min) / step);
      if (b >= nBins) b = nBins - 1;
      if (b < 0) b = 0;
      bins[b] += weights[i];
    }
    return { kind: 'distribution', support: labels, probs: bins };
  });
}

function pickDecimals(step: number): number {
  if (step >= 1) return 1;
  if (step >= 0.1) return 2;
  if (step >= 0.01) return 3;
  return 4;
}

/** Process a per-atom set of outputs: bin if continuous, truncate if huge. */
export function prepareAtomOutputs(outputs: RenderOutput[]): RenderOutput[] {
  const someContinuous = outputs.some(looksContinuous);
  if (someContinuous) return binShared(outputs);
  return outputs.map((o) => truncateOutput(o));
}
