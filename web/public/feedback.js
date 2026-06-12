// Feedback client. State lives in localStorage (rater_id, name, per-atom vote).
// API is POST /api/feedback. A single (rater, atom) ends up with one canonical
// row because comment/vote submissions both carry the current state forward.

const LS_KEY_ID = 'pplgym.rater_id';
const LS_KEY_NAME = 'pplgym.rater_name';
const LS_VOTE_PREFIX = 'pplgym.vote.';

const VOTE_GLYPH = { up: '👍', down: '👎', neutral: '∅' };

function ensureRaterId() {
  let id = localStorage.getItem(LS_KEY_ID);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(LS_KEY_ID, id);
  }
  return id;
}
const getName = () => localStorage.getItem(LS_KEY_NAME) || '';
const setName = (n) => localStorage.setItem(LS_KEY_NAME, n);

const getStoredVote = (atomId) => localStorage.getItem(LS_VOTE_PREFIX + atomId) || '';
function setStoredVote(atomId, vote) {
  if (vote === 'up' || vote === 'down') localStorage.setItem(LS_VOTE_PREFIX + atomId, vote);
  else localStorage.removeItem(LS_VOTE_PREFIX + atomId);
}

function promptForName(initial = '') {
  const back = document.getElementById('name-modal');
  if (!back) return Promise.resolve('');
  const input = document.getElementById('name-modal-input');
  const save = document.getElementById('name-modal-save');
  const cancel = document.getElementById('name-modal-cancel');
  input.value = initial;
  back.classList.add('open');
  setTimeout(() => input.focus(), 50);

  return new Promise((resolve) => {
    function close(value) {
      back.classList.remove('open');
      save.removeEventListener('click', onSave);
      cancel.removeEventListener('click', onCancel);
      input.removeEventListener('keydown', onKey);
      resolve(value);
    }
    function onSave() {
      const v = input.value.trim();
      if (!v) { input.focus(); return; }
      setName(v);
      close(v);
    }
    function onCancel() { close(''); }
    function onKey(e) {
      if (e.key === 'Enter') { e.preventDefault(); onSave(); }
      if (e.key === 'Escape') { e.preventDefault(); onCancel(); }
    }
    save.addEventListener('click', onSave);
    cancel.addEventListener('click', onCancel);
    input.addEventListener('keydown', onKey);
  });
}

async function ensureName() {
  return getName() || (await promptForName());
}

async function postFeedback(body) {
  const r = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`server returned ${r.status}`);
  return await r.json();
}

const widgetState = (node) => ({
  atomId: node.dataset.fbAtom,
  collection: node.dataset.fbCollection,
  datasetVersion: node.dataset.fbDatasetVersion,
});

function updateNameUI(node) {
  const name = getName();
  const empty = node.querySelector('.fb-name-empty');
  const filled = node.querySelector('.fb-name-filled');
  if (!empty || !filled) return;
  empty.hidden = !!name;
  filled.hidden = !name;
  if (name) filled.querySelector('.fb-name-text').textContent = name;
}

function setStatus(node, msg, kind) {
  const el = node.querySelector('.fb-status');
  if (!el) return;
  el.textContent = msg;
  el.className = 'fb-status' + (kind ? ' is-' + kind : '');
}

function setVoteHighlight(node, vote) {
  node.querySelector('.fb-btn.is-up')?.classList.toggle('active', vote === 'up');
  node.querySelector('.fb-btn.is-down')?.classList.toggle('active', vote === 'down');
}

async function submit(node, vote, comment) {
  const name = await ensureName();
  if (!name) return null;
  updateNameUI(node);
  const { atomId, collection, datasetVersion } = widgetState(node);
  setStatus(node, 'sending…');
  try {
    await postFeedback({
      atom_id: atomId, collection, dataset_version: datasetVersion,
      rater_id: ensureRaterId(), rater_name: name, vote, comment,
    });
    return { atomId, vote };
  } catch (err) {
    setStatus(node, 'error: ' + err.message, 'err');
    return null;
  }
}

async function handleVote(node, clickedVote) {
  const { atomId } = widgetState(node);
  const prior = getStoredVote(atomId);
  const vote = (prior === clickedVote) ? 'neutral' : clickedVote;
  const ta = node.querySelector('textarea');
  const comment = (ta?.value ?? '').trim();
  const result = await submit(node, vote, comment);
  if (!result) return;
  setStoredVote(atomId, vote);
  setVoteHighlight(node, vote);
  setStatus(node, 'recorded ' + VOTE_GLYPH[vote] + (comment ? ' + comment' : ''), 'ok');
}

async function handleComment(node) {
  const ta = node.querySelector('textarea');
  const comment = (ta?.value ?? '').trim();
  if (!comment) {
    setStatus(node, 'comment is empty', 'err');
    return;
  }
  const { atomId } = widgetState(node);
  const vote = getStoredVote(atomId) || 'neutral';
  const result = await submit(node, vote, comment);
  if (!result) return;
  ta.value = '';
  const carried = vote !== 'neutral' ? ` (carried ${VOTE_GLYPH[vote]})` : '';
  setStatus(node, 'comment saved' + carried, 'ok');
}

async function handleEditName() {
  const newName = await promptForName(getName());
  if (newName) document.querySelectorAll('.fb').forEach(updateNameUI);
}

document.addEventListener('click', (e) => {
  if (e.target.closest('.fb-edit')) {
    e.preventDefault();
    return handleEditName();
  }
  const voteBtn = e.target.closest('.fb-btn[data-vote]');
  if (voteBtn) {
    const node = voteBtn.closest('.fb');
    if (node) return handleVote(node, voteBtn.dataset.vote);
  }
  const submitBtn = e.target.closest('.fb-submit');
  if (submitBtn) {
    const node = submitBtn.closest('.fb');
    if (node) return handleComment(node);
  }
});

document.querySelectorAll('.fb').forEach((node) => {
  updateNameUI(node);
  const v = getStoredVote(node.dataset.fbAtom);
  if (v) setVoteHighlight(node, v);
});
