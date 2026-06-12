// ppl-gym page-level script.
// - hash-based atom selection
// - search + bucket filter + reviewed filter
// - keyboard nav (j/k, arrows, /, g, G)
// - source A/B switching → re-renders compare + chart
// - feedback (localStorage + POST + name modal)

(function () {
  'use strict';

  // ── Data / DOM ────────────────────────────────────────────────────────────
  const META = JSON.parse(document.getElementById('ppl-meta').textContent);
  const SYS_PROMPT = JSON.parse(document.getElementById('ppl-system-prompt').textContent);
  const PRIMER = JSON.parse(document.getElementById('ppl-primer').textContent);

  const rows = Array.from(document.querySelectorAll('.atom-row'));
  const groups = Array.from(document.querySelectorAll('.sidebar-group'));
  const details = Array.from(document.querySelectorAll('.detail'));
  const detailEmpty = document.querySelector('[data-detail-empty]');
  const search = document.getElementById('search');
  const bucketChips = Array.from(document.querySelectorAll('.chip[data-bk]'));
  const fbChips = Array.from(document.querySelectorAll('.chip[data-fb-filter]'));
  const visibleCountEl = document.querySelector('[data-visible-count]');
  const reviewedCountEl = document.querySelector('[data-reviewed-count]');
  const unreviewedCountEl = document.querySelector('[data-unreviewed-count]');
  const emptyMsg = document.querySelector('[data-empty]');
  const listScroll = document.getElementById('list-scroll');

  // Collection picker dropdown
  const collPicker = document.querySelector('[data-coll-picker]');
  const collToggle = document.querySelector('[data-coll-toggle]');
  const collMenu = document.querySelector('[data-coll-menu]');
  collToggle?.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = !collMenu.hidden;
    collMenu.hidden = open;
    collToggle.setAttribute('aria-expanded', String(!open));
  });
  document.addEventListener('click', (e) => {
    if (!collMenu || collMenu.hidden) return;
    if (!collPicker.contains(e.target)) collMenu.hidden = true;
  });

  // Pre-fill the system-prompt + primer slots in every atom detail.
  document.querySelectorAll('[data-sysprompt-placeholder]').forEach((el) => { el.textContent = SYS_PROMPT; });
  document.querySelectorAll('[data-primer-placeholder]').forEach((el) => { el.textContent = PRIMER; });

  // ── State ─────────────────────────────────────────────────────────────────
  const bucketFilter = new Set();
  let feedbackFilter = null; // 'reviewed' | 'unreviewed' | null
  // Per-atom source state (atomId -> { a, b }). a/b = 'gt' or runId.
  const sourceState = new Map();
  const defaultB = META.primaryId || (META.runs[0]?.id) || 'gt';
  for (const r of rows) {
    sourceState.set(r.dataset.aid, { a: 'gt', b: defaultB });
  }

  // ── Feedback (localStorage) ───────────────────────────────────────────────
  function readFeedbackMap() {
    const out = {};
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (!k || !k.startsWith('pplgym.fb.')) continue;
      try { out[k.slice('pplgym.fb.'.length)] = JSON.parse(localStorage.getItem(k)); } catch {}
    }
    return out;
  }
  let feedbackMap = readFeedbackMap();
  function setFb(atomId, rec) {
    if (rec) {
      localStorage.setItem(`pplgym.fb.${atomId}`, JSON.stringify(rec));
      feedbackMap[atomId] = rec;
    } else {
      localStorage.removeItem(`pplgym.fb.${atomId}`);
      delete feedbackMap[atomId];
    }
    window.dispatchEvent(new CustomEvent('pplgym.feedback-changed', { detail: { atomId, rec } }));
  }
  window.addEventListener('pplgym.feedback-changed', () => {
    feedbackMap = readFeedbackMap();
    updateAllPips();
    updateFbCounts();
    applyFilter();
  });
  window.addEventListener('storage', () => {
    feedbackMap = readFeedbackMap();
    updateAllPips();
    updateFbCounts();
    applyFilter();
  });

  function getRater() { return localStorage.getItem('pplgym.rater') || ''; }
  function setRater(name) { localStorage.setItem('pplgym.rater', name); }

  // ── Sidebar pips (★) ──────────────────────────────────────────────────────
  function updateAllPips() {
    for (const row of rows) {
      const aid = row.dataset.aid;
      const rec = feedbackMap[aid];
      const pip = row.querySelector('[data-fb-pip]');
      if (!rec) {
        pip.hidden = true;
        row.classList.remove('atom-row-fb');
        pip.className = 'fb-pip';
        continue;
      }
      row.classList.add('atom-row-fb');
      pip.hidden = false;
      const tone = rec.vote === 'up' ? 'great' : rec.vote === 'down' ? 'bad' : 'na';
      const glyph = rec.vote === 'up' ? '▲' : rec.vote === 'down' ? '▼' : '…';
      pip.className = `fb-pip bucket-${tone}`;
      pip.innerHTML =
        `<span class="fb-pip-glyph">${glyph}</span>` +
        (rec.comment ? `<span class="fb-pip-dot"></span>` : '');
      const parts = [];
      if (rec.vote === 'up') parts.push('looks right');
      else if (rec.vote === 'down') parts.push('issue');
      if (rec.comment) parts.push('comment');
      pip.title = `${rec.rater || 'anon'} — ${parts.join(' + ')}` + (rec.comment ? `\n"${rec.comment}"` : '');
    }
  }
  function updateFbCounts() {
    let n = 0;
    for (const k in feedbackMap) if (Object.prototype.hasOwnProperty.call(feedbackMap, k)) n++;
    if (reviewedCountEl) reviewedCountEl.textContent = String(n);
    if (unreviewedCountEl) unreviewedCountEl.textContent = String(rows.length - n);
  }

  // ── Filter / search ───────────────────────────────────────────────────────
  function applyFilter() {
    const q = (search?.value ?? '').trim().toLowerCase();
    let visible = 0;
    for (const row of rows) {
      const aid = row.dataset.aid;
      const okQ = !q || (row.dataset.search ?? '').indexOf(q) !== -1;
      const okBk = bucketFilter.size === 0 || bucketFilter.has(row.dataset.bk);
      const reviewed = !!feedbackMap[aid];
      const okFb = !feedbackFilter
        || (feedbackFilter === 'reviewed' && reviewed)
        || (feedbackFilter === 'unreviewed' && !reviewed);
      const show = okQ && okBk && okFb;
      row.classList.toggle('is-hidden', !show);
      if (show) visible++;
    }
    for (const g of groups) {
      const has = g.querySelector('.atom-row:not(.is-hidden)');
      g.classList.toggle('is-hidden', !has);
    }
    if (visibleCountEl) visibleCountEl.textContent = String(visible);
    if (emptyMsg) emptyMsg.hidden = visible !== 0;
  }

  bucketChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const id = chip.dataset.bk;
      if (bucketFilter.has(id)) {
        bucketFilter.delete(id);
        chip.classList.remove('chip-on');
        chip.setAttribute('aria-pressed', 'false');
      } else {
        bucketFilter.add(id);
        chip.classList.add('chip-on');
        chip.setAttribute('aria-pressed', 'true');
      }
      applyFilter();
    });
  });

  fbChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const val = chip.dataset.fbFilter;
      if (feedbackFilter === val) {
        feedbackFilter = null;
        chip.classList.remove('chip-on');
        chip.setAttribute('aria-pressed', 'false');
      } else {
        feedbackFilter = val;
        for (const c of fbChips) {
          const on = c.dataset.fbFilter === val;
          c.classList.toggle('chip-on', on);
          c.setAttribute('aria-pressed', String(on));
        }
      }
      applyFilter();
    });
  });

  search?.addEventListener('input', applyFilter);
  search?.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.target.blur(); search.value = ''; applyFilter(); }
  });

  // ── Selection / hash ──────────────────────────────────────────────────────
  function domIdFor(atomId) {
    return 'atom-' + String(atomId).replace(/\//g, '--').replace(/ /g, '_');
  }
  function setSelected(atomId) {
    const targetDomId = domIdFor(atomId);
    let found = false;
    for (const d of details) {
      const match = d.id === targetDomId;
      d.classList.toggle('is-active', match);
      if (match) found = true;
    }
    if (detailEmpty) detailEmpty.style.display = found ? 'none' : '';
    for (const r of rows) {
      r.classList.toggle('atom-row-selected', r.dataset.aid === atomId);
    }
    // Scroll the selected row into view if it's outside.
    const sel = document.querySelector(`.atom-row[data-aid="${cssEscape(atomId)}"]`);
    if (sel && listScroll) {
      const rRow = sel.getBoundingClientRect();
      const rList = listScroll.getBoundingClientRect();
      if (rRow.top < rList.top || rRow.bottom > rList.bottom) {
        const offset = rRow.top - rList.top - (rList.height / 2) + (rRow.height / 2);
        listScroll.scrollBy({ top: offset, behavior: 'smooth' });
      }
    }
    document.querySelector('.detail-pane')?.scrollTo({ top: 0 });
    // Mount feedback widget into the active atom.
    if (found) mountFeedback(atomId);
  }
  function cssEscape(s) { return String(s).replace(/(["\\])/g, '\\$1'); }

  function focusFromHash() {
    const h = decodeURIComponent(location.hash.slice(1));
    const idFromDom = h.startsWith('atom-') ? h : ('atom-' + h);
    let found = null;
    for (const r of rows) {
      if (domIdFor(r.dataset.aid) === idFromDom || r.dataset.aid === h) {
        found = r.dataset.aid;
        break;
      }
    }
    setSelected(found || META.firstAtomId);
  }
  window.addEventListener('hashchange', focusFromHash);

  rows.forEach((r) => r.addEventListener('click', () => {
    location.hash = '#' + r.id; // r.id starts with 'atom-...'? actually it doesn't; use computed
  }));
  // The actual hash is the encoded atom id.
  rows.forEach((r) => r.addEventListener('click', () => {
    const aid = r.dataset.aid;
    const next = '#' + encodeURIComponent(aid);
    if (location.hash === next) {
      setSelected(aid);
    } else {
      history.pushState(null, '', next);
      setSelected(aid);
    }
  }));

  // ── Keyboard nav ──────────────────────────────────────────────────────────
  function visibleAtoms() {
    return rows.filter((r) => !r.classList.contains('is-hidden'));
  }
  function indexOfSelected(list) {
    const aid = currentAtomId();
    for (let i = 0; i < list.length; i++) if (list[i].dataset.aid === aid) return i;
    return -1;
  }
  function jump(delta) {
    const list = visibleAtoms();
    if (list.length === 0) return;
    const cur = indexOfSelected(list);
    const next = cur < 0 ? 0 : (cur + delta + list.length) % list.length;
    const aid = list[next].dataset.aid;
    history.pushState(null, '', '#' + encodeURIComponent(aid));
    setSelected(aid);
  }
  function currentAtomId() {
    const active = document.querySelector('.detail.is-active');
    return active?.dataset.aid || META.firstAtomId;
  }
  document.addEventListener('keydown', (e) => {
    const t = e.target;
    if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT') {
      if (e.key === 'Escape') t.blur();
      return;
    }
    if (e.key === '/') { e.preventDefault(); search?.focus(); search?.select(); return; }
    if (e.key === 'j' || e.key === 'ArrowDown' || e.key === ' ' || e.key === 'ArrowRight') {
      e.preventDefault(); jump(1);
    } else if (e.key === 'k' || e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
      e.preventDefault(); jump(-1);
    } else if (e.key === 'g' && e.shiftKey) {
      const list = visibleAtoms(); if (!list.length) return;
      const aid = list[list.length - 1].dataset.aid;
      history.pushState(null, '', '#' + encodeURIComponent(aid));
      setSelected(aid);
    } else if (e.key === 'g') {
      const list = visibleAtoms(); if (!list.length) return;
      const aid = list[0].dataset.aid;
      history.pushState(null, '', '#' + encodeURIComponent(aid));
      setSelected(aid);
    }
  });

  // ── Source switching (compare + chart re-render) ─────────────────────────
  function payloadFor(detailEl) {
    const script = detailEl.querySelector('script[data-source-payload]');
    if (!script) return {};
    try { return JSON.parse(script.textContent); } catch { return {}; }
  }
  function bucketDef(bid) {
    return BUCKETS_BY_ID[bid] || BUCKETS_BY_ID['no-run'];
  }
  function setSource(detailEl, slot, srcKey) {
    const aid = detailEl.dataset.aid;
    const state = sourceState.get(aid) || { a: 'gt', b: defaultB };
    state[slot] = srcKey;
    sourceState.set(aid, state);
    rerenderCompare(detailEl);
  }
  function rerenderCompare(detailEl) {
    const aid = detailEl.dataset.aid;
    const state = sourceState.get(aid) || { a: 'gt', b: defaultB };
    const pay = payloadFor(detailEl);
    const a = pay[state.a];
    const b = pay[state.b];

    // Update run-pill highlights
    detailEl.querySelectorAll('.run-pill').forEach((pill) => {
      const key = pill.dataset.source;
      pill.classList.toggle('is-source-a', key === state.a);
      pill.classList.toggle('is-source-b', key === state.b);
    });

    // Update section-title labels
    const labelA = detailEl.querySelector('[data-ab-label-a]');
    const labelB = detailEl.querySelector('[data-ab-label-b]');
    if (labelA) labelA.textContent = a?.short ?? state.a;
    if (labelB) labelB.textContent = b?.short ?? state.b;

    // Rebuild compare grid
    const grid = detailEl.querySelector('[data-compare-grid]');
    if (grid) grid.innerHTML = compareColHtml(a, 'a') + compareColHtml(b, 'b');

    // Rebuild chart / value-output slot
    const slot = detailEl.querySelector('[data-chart-slot]');
    if (slot) slot.innerHTML = renderChartSection(a, b);
  }

  function compareColHtml(src, side) {
    if (!src) {
      return `<div class="compare-col"><div class="compare-col-hd"><span class="compare-col-tag ${side === 'a' ? 'tag-a' : 'tag-b'}">?</span></div><div class="out-empty">(unknown source)</div></div>`;
    }
    const tag = side === 'a' ? 'tag-a' : 'tag-b';
    let body;
    if (src.error) {
      body = `<div class="compare-error">` +
        `<div class="compare-error-hd">execution error</div>` +
        `<pre class="compare-error-msg">${escapeHtml(src.error)}</pre>` +
        (src.code ? renderCodeBlock(src.code, 'webppl') : '') +
        `</div>`;
    } else if (src.code) {
      body = renderCodeBlock(src.code, 'webppl');
    } else {
      body = `<div class="out-empty">(no code — this run didn't score this atom)</div>`;
    }
    const bucketBadge = src.bucket ? renderBucketBadge(src.bucket, src.tv, 'xs', true) : '';
    return `<div class="compare-col">` +
      `<div class="compare-col-hd">` +
        `<span class="compare-col-tag ${tag}">${escapeHtml(src.short)}</span>` +
        `<span class="compare-col-label">${escapeHtml(src.label)}</span>` +
        bucketBadge +
      `</div>` +
      body +
    `</div>`;
  }

  function renderChartSection(a, b) {
    const aOut = a?.output;
    const bOut = b?.output;
    const aDist = aOut && (aOut.kind === 'distribution' || aOut.kind === 'samples');
    const bDist = bOut && (bOut.kind === 'distribution' || bOut.kind === 'samples');
    if (aDist || bDist) {
      return `<div class="section-title">` +
        `<span class="section-title-num">02</span>` +
        `<span>output overlay</span>` +
        `</div>` +
        renderChart({
          a: aDist ? aOut : null,
          b: bDist ? bOut : null,
          labelA: a?.short ?? 'A',
          labelB: b?.short ?? 'B',
        });
    }
    if (aOut || bOut) {
      return `<div class="section-title">` +
        `<span class="section-title-num">02</span>` +
        `<span>output</span>` +
        `</div>` +
        `<div class="compare-grid">` +
          `<div class="compare-col"><div class="compare-col-hd"><span class="compare-col-tag tag-a">${escapeHtml(a?.short ?? 'A')}</span></div>${renderValueOutput(aOut)}</div>` +
          `<div class="compare-col"><div class="compare-col-hd"><span class="compare-col-tag tag-b">${escapeHtml(b?.short ?? 'B')}</span></div>${renderValueOutput(bOut, aOut ? '(run produced no output)' : '')}</div>` +
        `</div>`;
    }
    return '';
  }

  // ── Click handlers for source switching ───────────────────────────────────
  document.addEventListener('click', (e) => {
    const pill = e.target.closest('.run-pill[data-source]');
    if (pill) {
      const detailEl = pill.closest('.detail');
      if (!detailEl) return;
      setSource(detailEl, e.shiftKey ? 'a' : 'b', pill.dataset.source);
      return;
    }
    const asA = e.target.closest('[data-action-as-a]');
    if (asA && !asA.disabled) {
      const detailEl = asA.closest('.detail');
      if (detailEl) setSource(detailEl, 'a', asA.getAttribute('data-action-as-a'));
      return;
    }
    const asB = e.target.closest('[data-action-as-b]');
    if (asB && !asB.disabled) {
      const detailEl = asB.closest('.detail');
      if (detailEl) setSource(detailEl, 'b', asB.getAttribute('data-action-as-b'));
      return;
    }
  });

  // ── Feedback bar (active atom) ────────────────────────────────────────────
  function mountFeedback(atomId) {
    const detailEl = details.find((d) => d.dataset.aid === atomId);
    if (!detailEl) return;
    const box = detailEl.querySelector('[data-feedback]');
    if (!box || box.dataset.wired === '1') {
      // Already wired; just refresh state for the current atom.
      refreshFeedbackUI(detailEl);
      return;
    }
    box.dataset.wired = '1';

    const voteUp = box.querySelector('[data-vote="up"]');
    const voteDown = box.querySelector('[data-vote="down"]');
    const ta = box.querySelector('[data-fb-text]');
    const submit = box.querySelector('[data-fb-submit]');
    const ack = box.querySelector('[data-fb-ack]');
    const prev = box.querySelector('[data-fb-prev]');
    const raterHost = box.querySelector('[data-fb-rater-host]');
    const raterName = box.querySelector('[data-fb-rater-name]');
    const changeName = box.querySelector('[data-fb-change-name]');

    ta.addEventListener('input', () => { submit.disabled = !ta.value.trim(); });
    voteUp.addEventListener('click', () => wantSubmit('up'));
    voteDown.addEventListener('click', () => wantSubmit('down'));
    submit.addEventListener('click', () => wantSubmit('comment'));
    changeName?.addEventListener('click', () => askName().then((n) => { if (n) { setRater(n); refreshFeedbackUI(detailEl); }}));

    refreshFeedbackUI(detailEl);

    function wantSubmit(kind) {
      const name = getRater();
      if (!name) {
        askName().then((n) => { if (n) { setRater(n); doSubmit(kind); }});
        return;
      }
      doSubmit(kind);
    }
    function doSubmit(kind) {
      const cur = feedbackMap[atomId] || null;
      let nextVote = cur?.vote ?? null;
      let nextComment = cur?.comment ?? null;
      if (kind === 'up' || kind === 'down') {
        nextVote = cur?.vote === kind ? null : kind;
        const keepComment = nextComment || (ta.value.trim() || null);
        if (!nextVote && !keepComment) {
          setFb(atomId, null);
          showAck(true);
          ta.value = ''; submit.disabled = true;
          return;
        }
        const rec = { vote: nextVote, comment: keepComment, rater: getRater(), at: new Date().toISOString() };
        setFb(atomId, rec);
        showAck(false, rec);
        ta.value = ''; submit.disabled = true;
        return;
      }
      if (kind === 'comment') {
        const c = ta.value.trim();
        if (!c) return;
        const rec = { vote: nextVote, comment: c, rater: getRater(), at: new Date().toISOString() };
        setFb(atomId, rec);
        // Server write-through
        postFeedback(atomId, rec).catch(() => {});
        showAck(false, rec);
        ta.value = ''; submit.disabled = true;
      }
    }
    function postFeedback(atomId, rec) {
      return fetch('/api/feedback', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          atom_id: atomId,
          collection: box.dataset.collection || META.slug,
          dataset_version: box.dataset.datasetVersion || META.slug,
          rater_id: getOrCreateRaterId(),
          rater_name: rec.rater || 'anon',
          vote: rec.vote || 'neutral',
          comment: rec.comment || '',
        }),
      });
    }
    function showAck(cleared, rec) {
      ack.hidden = false;
      ack.classList.toggle('feedback-ack-cleared', !!cleared);
      ack.innerHTML = cleared
        ? `cleared. <span class="feedback-ack-meta">no record saved</span>`
        : `saved. <span class="feedback-ack-meta">${rec.vote || 'no vote'}${rec.comment ? ' · with comment' : ''}</span>`;
    }
  }

  function refreshFeedbackUI(detailEl) {
    const atomId = detailEl.dataset.aid;
    const box = detailEl.querySelector('[data-feedback]');
    if (!box) return;
    const voteUp = box.querySelector('[data-vote="up"]');
    const voteDown = box.querySelector('[data-vote="down"]');
    const ack = box.querySelector('[data-fb-ack]');
    const prev = box.querySelector('[data-fb-prev]');
    const raterHost = box.querySelector('[data-fb-rater-host]');
    const raterName = box.querySelector('[data-fb-rater-name]');
    const rec = feedbackMap[atomId];
    voteUp.classList.toggle('vote-on', rec?.vote === 'up');
    voteDown.classList.toggle('vote-on', rec?.vote === 'down');
    ack.hidden = true;
    if (rec) {
      prev.hidden = false;
      const bid = rec.vote === 'up' ? 'val+' : rec.vote === 'down' ? 'val-' : 'no-run';
      const badgeHtml = renderBucketBadge(bid, null, 'xs', false);
      const c = rec.comment ? ` — "${escapeHtml(rec.comment.slice(0, 80))}${rec.comment.length > 80 ? '…' : ''}"` : ' (no comment)';
      prev.innerHTML = `your last record: ${badgeHtml}${c}`;
    } else {
      prev.hidden = true;
      prev.innerHTML = '';
    }
    const name = getRater();
    if (name) {
      raterHost.hidden = false;
      raterName.textContent = name;
    } else {
      raterHost.hidden = true;
    }
  }

  function getOrCreateRaterId() {
    let id = localStorage.getItem('pplgym.rater_id');
    if (!id) { id = crypto.randomUUID(); localStorage.setItem('pplgym.rater_id', id); }
    return id;
  }

  // ── Name modal ────────────────────────────────────────────────────────────
  const modal = document.getElementById('name-modal');
  const modalInput = document.getElementById('name-modal-input');
  const modalSave = document.getElementById('name-modal-save');
  const modalCancel = document.getElementById('name-modal-cancel');
  let modalResolve = null;
  function askName() {
    return new Promise((resolve) => {
      modalResolve = resolve;
      modalInput.value = getRater();
      modalSave.disabled = !modalInput.value.trim();
      modal.hidden = false;
      setTimeout(() => modalInput.focus(), 30);
    });
  }
  modalInput?.addEventListener('input', () => { modalSave.disabled = !modalInput.value.trim(); });
  modalInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && modalInput.value.trim()) modalSave.click();
    if (e.key === 'Escape') modalCancel.click();
  });
  modalSave?.addEventListener('click', () => {
    const v = modalInput.value.trim();
    if (!v) return;
    modal.hidden = true;
    if (modalResolve) { modalResolve(v); modalResolve = null; }
  });
  modalCancel?.addEventListener('click', () => {
    modal.hidden = true;
    if (modalResolve) { modalResolve(''); modalResolve = null; }
  });
  modal?.addEventListener('click', (e) => { if (e.target === modal) modalCancel.click(); });

  // ── Bucket + chart shared client-side helpers ─────────────────────────────
  const BUCKETS_BY_ID = {
    'TV=0':   { label: 'TV=0',   glyph: '●', tone: 'great', desc: 'distribution exact match' },
    'TV<.05': { label: 'TV<.05', glyph: '◉', tone: 'good',  desc: 'very close distribution' },
    'TV<.5':  { label: 'TV<.5',  glyph: '◐', tone: 'ok',    desc: 'moderate distribution disagreement' },
    'TV<1':   { label: 'TV<1',   glyph: '◔', tone: 'poor',  desc: 'poor distribution match' },
    'TV=1':   { label: 'TV=1',   glyph: '○', tone: 'bad',   desc: 'full distribution disagreement' },
    'val+':   { label: 'val+',   glyph: '✓', tone: 'great', desc: 'value/scalar match' },
    'val-':   { label: 'val-',   glyph: '✗', tone: 'poor',  desc: 'value mismatch' },
    'shape!': { label: 'shape!', glyph: '△', tone: 'bad',   desc: 'wrong-shaped answer' },
    'fail':   { label: 'fail',   glyph: '⚠', tone: 'err',   desc: 'execution crashed' },
    'no-run': { label: 'no-run', glyph: '–', tone: 'na',    desc: 'this run did not score this atom' },
  };
  function fmtTV(n) {
    if (n == null) return '—';
    if (n === 0) return '0.000';
    if (n < 0.001) return n.toExponential(1);
    return n.toFixed(3);
  }
  function renderBucketBadge(bid, tv, size, showTv) {
    const b = BUCKETS_BY_ID[bid] || BUCKETS_BY_ID['no-run'];
    const tvBit = showTv && tv != null ? `<span class="bucket-tv">${fmtTV(tv)}</span>` : '';
    return `<span class="bucket bucket-${b.tone} bucket-${size || 'sm'}">` +
      `<span class="bucket-glyph">${b.glyph}</span>` +
      `<span class="bucket-label">${b.label}</span>` +
      tvBit +
    `</span>`;
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // ── WebPPL syntax highlight + code block ──────────────────────────────────
  const KW = new Set(['var','function','return','if','else','for','while','true','false','null','undefined','new']);
  const BUILTIN = new Set([
    'Infer','Enumerate','MCMC','SMC','rejection','sample','factor','observe','condition',
    'expectation','flip','uniform','uniformDraw','gaussian','beta','dirichlet','categorical',
    'discrete','mem','mapData','map','map2','reduce','Categorical','Bernoulli','Binomial',
    'Gaussian','Beta','Dirichlet','Vector','Math','repeat',
  ]);
  function tokenize(src) {
    const toks = []; let i = 0;
    while (i < src.length) {
      const c = src[i];
      if (c === '/' && src[i+1] === '/') {
        const j = src.indexOf('\n', i);
        const end = j === -1 ? src.length : j;
        toks.push({ t: 'cm', v: src.slice(i, end) });
        i = end; continue;
      }
      if (c === '/' && src[i+1] === '*') {
        const j = src.indexOf('*/', i+2);
        const end = j === -1 ? src.length : j+2;
        toks.push({ t: 'cm', v: src.slice(i, end) });
        i = end; continue;
      }
      if (c === "'" || c === '"' || c === '`') {
        const q = c; let j = i+1;
        while (j < src.length && src[j] !== q) {
          if (src[j] === '\\') j += 2; else j++;
        }
        toks.push({ t: 's', v: src.slice(i, j+1) }); i = j+1; continue;
      }
      if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(src[i+1] || ''))) {
        let j = i; while (j < src.length && /[0-9.eE+-]/.test(src[j])) j++;
        toks.push({ t: 'n', v: src.slice(i, j) }); i = j; continue;
      }
      if (/[A-Za-z_$]/.test(c)) {
        let j = i; while (j < src.length && /[A-Za-z0-9_$]/.test(src[j])) j++;
        const v = src.slice(i, j);
        let t = 'i';
        if (KW.has(v)) t = 'k';
        else if (BUILTIN.has(v)) t = 'b';
        else if (src[j] === '(') t = 'f';
        toks.push({ t, v }); i = j; continue;
      }
      if (/[{}()\[\],;:]/.test(c)) { toks.push({ t: 'p', v: c }); i++; continue; }
      if (/[+\-*/<>=!&|?]/.test(c)) {
        let j = i; while (j < src.length && /[+\-*/<>=!&|?]/.test(src[j])) j++;
        toks.push({ t: 'o', v: src.slice(i, j) }); i = j; continue;
      }
      toks.push({ t: 'w', v: c }); i++;
    }
    return toks;
  }
  function highlightLines(src) {
    const tokens = tokenize(src);
    const lines = [[]];
    for (const tok of tokens) {
      const segs = tok.v.split('\n');
      segs.forEach((seg, i) => {
        if (i > 0) lines.push([]);
        if (seg) lines[lines.length - 1].push({ t: tok.t, v: seg });
      });
    }
    return lines;
  }
  function renderCodeBlock(code, lang) {
    const lines = highlightLines(code || '');
    const body = lines.map((toks, i) => {
      const content = toks.length === 0
        ? '​'
        : toks.map((tk) => `<span class="tok-${tk.t}">${escapeHtml(tk.v)}</span>`).join('');
      return `<div class="code-line"><span class="code-ln">${i + 1}</span><span class="code-content">${content}</span></div>`;
    }).join('');
    return `<div class="code"><div class="code-lang">${escapeHtml(lang || 'webppl')}</div>` +
      `<pre class="code-body">${body}</pre></div>`;
  }

  // ── Chart (mirrored vertical bars) ────────────────────────────────────────
  function seriesProbs(s, support) {
    if (!s) return support.map(() => 0);
    return support.map((label) => {
      const idx = s.support.indexOf(label);
      if (idx === -1) return 0;
      if (s.kind === 'samples') {
        const total = (s.counts || []).reduce((a, b) => a + b, 0) || 1;
        return ((s.counts || [])[idx] || 0) / total;
      }
      return (s.probs || [])[idx] || 0;
    });
  }
  function niceTick(step) {
    if (step <= 0) return 1;
    const pow = Math.pow(10, Math.floor(Math.log10(step)));
    const norm = step / pow;
    const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
    return nice * pow;
  }
  function fmtNum(n) {
    if (Number.isInteger(n)) return String(n);
    const abs = Math.abs(n);
    if (abs >= 100) return n.toFixed(0);
    if (abs >= 10) return n.toFixed(1);
    return n.toFixed(2);
  }
  function renderChart({ a, b, labelA, labelB, maxBars }) {
    const maxN = maxBars || 48;
    const supportSet = new Set();
    for (const s of a?.support || []) supportSet.add(s);
    for (const s of b?.support || []) supportSet.add(s);
    let support = Array.from(supportSet);
    if (!support.length) return '<div class="out-empty">(no distribution)</div>';
    const numericVals = support.map((s) => Number(s));
    const numeric = numericVals.every((v) => Number.isFinite(v));
    const isContinuous = numeric && support.length >= 14;
    let aProbs = seriesProbs(a, support);
    let bProbs = seriesProbs(b, support);
    let truncated = 0;
    if (!isContinuous && support.length > maxN) {
      const ranked = support.map((s, i) => ({ s, p: Math.max(aProbs[i], bProbs[i]) }))
        .sort((x, y) => y.p - x.p).slice(0, maxN).map(({ s }) => s);
      truncated = support.length - ranked.length;
      support = ranked;
      aProbs = seriesProbs(a, support);
      bProbs = seriesProbs(b, support);
    }
    if (numeric) {
      const idx = support.map((_, i) => i).sort((x, y) => Number(support[x]) - Number(support[y]));
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
    const fmtY = (v) => v === 0 ? '0' : maxP >= 0.1 ? v.toFixed(2) : maxP >= 0.01 ? v.toFixed(3) : v.toExponential(1);
    let grid = '';
    for (let i = 0; i < ticks.length; i++) {
      const t = ticks[i];
      const yA = midY - (t / maxP) * halfH;
      const yB = midY + (t / maxP) * halfH;
      grid += `<line x1="${padL}" y1="${yA}" x2="${w - padR}" y2="${yA}" class="chart-grid"/>`;
      grid += `<text x="${padL - 6}" y="${yA}" class="chart-yt" text-anchor="end" dominant-baseline="middle">${fmtY(t)}</text>`;
      if (i > 0) {
        grid += `<line x1="${padL}" y1="${yB}" x2="${w - padR}" y2="${yB}" class="chart-grid"/>`;
        grid += `<text x="${padL - 6}" y="${yB}" class="chart-yt" text-anchor="end" dominant-baseline="middle">${fmtY(t)}</text>`;
      }
    }
    const mid = `<line x1="${padL}" y1="${midY}" x2="${w - padR}" y2="${midY}" class="chart-axis"/>`;
    const body = isContinuous
      ? renderNumericBars({ support, aProbs, bProbs, maxP, padL, padR, innerW, midY, halfH, h, w })
      : renderCategoricalBars({ support, aProbs, bProbs, maxP, padL, innerW, midY, halfH, h });
    const metaBits = [`${support.length} bin${support.length === 1 ? '' : 's'}`];
    if (numeric && support.length > 1) {
      const xs = support.map(Number);
      metaBits.push(`${fmtNum(Math.min(...xs))} … ${fmtNum(Math.max(...xs))}`);
    }
    if (truncated > 0) metaBits.push(`top ${support.length} of ${support.length + truncated}`);
    return `<div class="chart">` +
      `<div class="chart-legend">` +
        `<span class="chart-legend-item chart-legend-a"><span class="chart-legend-swatch"></span> ${escapeHtml(labelA)}</span>` +
        `<span class="chart-legend-item chart-legend-b"><span class="chart-legend-swatch"></span> ${escapeHtml(labelB)}</span>` +
        `<span class="chart-legend-meta">${escapeHtml(metaBits.join(' · '))}</span>` +
      `</div>` +
      `<svg viewBox="0 0 ${w} ${h}" class="chart-svg" role="img" aria-label="distribution overlay">` +
        grid + mid + body +
      `</svg>` +
    `</div>`;
  }
  function renderNumericBars({ support, aProbs, bProbs, maxP, padL, padR, innerW, midY, halfH, h, w }) {
    const xs = support.map(Number);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const range = xMax - xMin || 1;
    const xPos = (v) => padL + ((v - xMin) / range) * innerW;
    const yA = (p) => midY - (p / maxP) * halfH;
    const yB = (p) => midY + (p / maxP) * halfH;
    const pitchHalf = (i) => {
      if (xs.length === 1) return innerW / 2;
      if (i === 0) return (xPos(xs[1]) - xPos(xs[0])) / 2;
      if (i === xs.length - 1) return (xPos(xs[i]) - xPos(xs[i - 1])) / 2;
      return Math.max(xPos(xs[i]) - xPos(xs[i - 1]), xPos(xs[i + 1]) - xPos(xs[i])) / 2;
    };
    const buildArea = (probs, yFn) => {
      let d = '';
      xs.forEach((x, i) => {
        const half = pitchHalf(i);
        const xL = xPos(x) - half, xR = xPos(x) + half;
        const y = yFn(probs[i]);
        if (i === 0) d += `M ${xL.toFixed(2)} ${midY} L ${xL.toFixed(2)} ${y.toFixed(2)} `;
        d += `L ${xR.toFixed(2)} ${y.toFixed(2)} `;
        if (i === xs.length - 1) d += `L ${xR.toFixed(2)} ${midY} Z`;
        else {
          const nh = pitchHalf(i + 1);
          const xN = xPos(xs[i + 1]) - nh;
          const yN = yFn(probs[i + 1]);
          d += `L ${xN.toFixed(2)} ${y.toFixed(2)} L ${xN.toFixed(2)} ${yN.toFixed(2)} `;
        }
      });
      return d;
    };
    const buildLine = (probs, yFn) => {
      let d = '';
      xs.forEach((x, i) => {
        const half = pitchHalf(i);
        const xL = xPos(x) - half, xR = xPos(x) + half;
        const y = yFn(probs[i]);
        if (i === 0) d += `M ${xL.toFixed(2)} ${y.toFixed(2)} `;
        d += `L ${xR.toFixed(2)} ${y.toFixed(2)} `;
        if (i < xs.length - 1) {
          const nh = pitchHalf(i + 1);
          const xN = xPos(xs[i + 1]) - nh;
          const yN = yFn(probs[i + 1]);
          d += `L ${xN.toFixed(2)} ${y.toFixed(2)} L ${xN.toFixed(2)} ${yN.toFixed(2)} `;
        }
      });
      return d;
    };
    const tickCount = Math.min(7, Math.max(3, Math.floor(innerW / 90)));
    const step = niceTick(range / (tickCount - 1));
    const start = Math.ceil(xMin / step) * step;
    const ticks = [];
    for (let v = start; v <= xMax + step / 1000; v += step) ticks.push(Number(v.toFixed(10)));
    const tickMarks = ticks.map((v) => {
      const x = xPos(v);
      return `<line x1="${x.toFixed(2)}" y1="${h - 28}" x2="${x.toFixed(2)}" y2="${h - 24}" class="chart-axis"/>` +
        `<text x="${x.toFixed(2)}" y="${h - 10}" class="chart-xt" text-anchor="middle">${escapeHtml(fmtNum(v))}</text>`;
    }).join('');
    const modeIdx = (probs) => {
      let m = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[m]) m = i;
      return probs[m] > 0 ? m : -1;
    };
    const renderMode = (i, probs, yFn, side) => {
      if (i < 0) return '';
      const x = xPos(xs[i]);
      const y = yFn(probs[i]);
      const dy = side === 'a' ? -8 : 14;
      const label = `${fmtNum(xs[i])} · ${probs[i].toFixed(3)}`;
      return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="2.5" class="chart-mode chart-mode-${side}"/>` +
        `<text x="${x.toFixed(2)}" y="${(y + dy).toFixed(2)}" class="chart-mode-label chart-mode-label-${side}" text-anchor="middle">${escapeHtml(label)}</text>`;
    };
    const modeA = modeIdx(aProbs);
    const modeB = modeIdx(bProbs);
    const hits = xs.map((x, i) => {
      const half = pitchHalf(i);
      const xL = xPos(x) - half;
      const delta = aProbs[i] - bProbs[i];
      const tip = `x = ${fmtNum(x)}\nA = ${aProbs[i].toFixed(4)}\nB = ${bProbs[i].toFixed(4)}\nΔ = ${delta.toFixed(4)}`;
      return `<rect x="${xL.toFixed(2)}" y="${(midY - halfH).toFixed(2)}" width="${(half * 2).toFixed(2)}" height="${(halfH * 2).toFixed(2)}" fill="transparent" class="chart-hover"><title>${escapeHtml(tip)}</title></rect>`;
    }).join('');
    return tickMarks +
      `<path d="${buildArea(aProbs, yA)}" class="chart-area chart-area-a"/>` +
      `<path d="${buildArea(bProbs, yB)}" class="chart-area chart-area-b"/>` +
      `<path d="${buildLine(aProbs, yA)}" class="chart-line chart-line-a"/>` +
      `<path d="${buildLine(bProbs, yB)}" class="chart-line chart-line-b"/>` +
      renderMode(modeA, aProbs, yA, 'a') +
      renderMode(modeB, bProbs, yB, 'b') +
      hits;
  }
  function renderCategoricalBars({ support, aProbs, bProbs, maxP, padL, innerW, midY, halfH, h }) {
    const colW = innerW / Math.max(1, support.length);
    const barW = Math.max(4, Math.min(40, colW * 0.72));
    const showValueLabels = barW >= 28;
    const xLabelStride = Math.max(1, Math.ceil(40 / colW));
    return support.map((s, i) => {
      const x = padL + i * colW + (colW - barW) / 2;
      const ha = (aProbs[i] / maxP) * halfH;
      const hb = (bProbs[i] / maxP) * halfH;
      const showXLabel = i % xLabelStride === 0;
      const labelsBlock = showValueLabels
        ? (aProbs[i] > 0.005 ? `<text x="${(x + barW/2).toFixed(2)}" y="${(midY - ha - 4).toFixed(2)}" class="chart-val chart-val-a" text-anchor="middle">${aProbs[i].toFixed(2)}</text>` : '') +
          (bProbs[i] > 0.005 ? `<text x="${(x + barW/2).toFixed(2)}" y="${(midY + hb + 12).toFixed(2)}" class="chart-val chart-val-b" text-anchor="middle">${bProbs[i].toFixed(2)}</text>` : '')
        : '';
      const xLabel = showXLabel
        ? `<text x="${(x + barW/2).toFixed(2)}" y="${(h - 8).toFixed(2)}" class="chart-xt" text-anchor="middle">${escapeHtml(s)}</text>`
        : '';
      const tip = `${s}\nA = ${aProbs[i].toFixed(3)}\nB = ${bProbs[i].toFixed(3)}`;
      return `<g>` +
        `<rect x="${x.toFixed(2)}" y="${(midY - ha).toFixed(2)}" width="${barW.toFixed(2)}" height="${ha.toFixed(2)}" class="chart-bar chart-bar-a"><title>${escapeHtml(tip)}</title></rect>` +
        `<rect x="${x.toFixed(2)}" y="${midY.toFixed(2)}" width="${barW.toFixed(2)}" height="${hb.toFixed(2)}" class="chart-bar chart-bar-b"><title>${escapeHtml(tip)}</title></rect>` +
        labelsBlock + xLabel +
      `</g>`;
    }).join('');
  }
  function renderValueOutput(o, fallback) {
    if (!o) return `<div class="out-empty">${escapeHtml(fallback || '(no output)')}</div>`;
    if (o.kind === 'value') {
      const v = o.value;
      let txt;
      if (Array.isArray(v)) {
        txt = '[' + v.map((x) => typeof x === 'number' ? x.toFixed(4) : JSON.stringify(x)).join(', ') + ']';
      } else if (typeof v === 'number') {
        txt = v.toFixed(4);
      } else {
        txt = JSON.stringify(v, null, 2);
      }
      return `<div class="out-value"><pre class="out-value-pre">${escapeHtml(txt)}</pre></div>`;
    }
    if (o.kind === 'record') {
      const rows = Object.entries(o.fields).map(([k, v]) => {
        let val;
        if (v && v.kind === 'value') val = typeof v.value === 'number' ? v.value.toFixed(4) : JSON.stringify(v.value);
        else if (v && v.kind === 'distribution') val = `dist(${v.support.length})`;
        else val = '...';
        return `<div class="out-record-row"><span class="out-record-key">${escapeHtml(k)}</span><span class="out-record-eq">=</span><span class="out-record-val">${escapeHtml(val)}</span></div>`;
      }).join('');
      return `<div class="out-record">${rows}</div>`;
    }
    return `<div class="out-empty">${escapeHtml(fallback || '(no output)')}</div>`;
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  updateAllPips();
  updateFbCounts();
  applyFilter();
  focusFromHash();
})();
