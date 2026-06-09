import { readFile, readdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { Atom, Collection, RunMeta, ScoredRun } from './types';

// Astro's build/prerender runs with CWD = the Astro project root (web/).
const REPO_ROOT = resolve(process.cwd(), '..');
const DATA_DIR = join(REPO_ROOT, 'data');
const RUNS_DIR = join(DATA_DIR, 'eval_runs');

export const COLLECTIONS: Collection[] = [
  {
    slug: 'atomized-v2',
    label: 'ProbMods2 exercises',
    jsonlPath: 'data/atomized_v2.jsonl',
    primaryRun: 'sonnet-46-primer-v3',
    description: 'Hand-curated WebPPL exercises from probmods.org (76 atoms) — the canonical v2 dataset.',
  },
  {
    slug: 'curated-v3-dippl',
    label: 'DIPPL exercises',
    jsonlPath: 'data/curated_v3/dippl.jsonl',
    primaryRun: 'sonnet-46-primer-dippl',
    description: 'Hand-curated WebPPL exercises from dippl.org (17 atoms) — v3 pilot using the M-to-N chunking flow.',
  },
  {
    slug: 'curated-v3-forestdb',
    label: 'forestdb models',
    jsonlPath: 'data/curated_v3/forestdb.jsonl',
    primaryRun: 'sonnet-46-primer-forestdb',
    description: 'Hand-curated WebPPL models from forestdb.org (stratified pilot, 10 atoms) — paper-page style.',
  },
  {
    slug: 'pyro-v3-probmods',
    label: 'ProbMods exercises (Pyro)',
    jsonlPath: 'data/pyro_v3/probmods.jsonl',
    primaryRun: 'sonnet-46-primer-pyro-probmods',
    description: 'Probmods textbook exercises translated to Pyro (40 atoms from v2; 36 atoms not yet translated due to MCMC complexity).',
  },
];

async function readJsonl<T>(absPath: string): Promise<T[]> {
  let text: string;
  try {
    text = await readFile(absPath, 'utf8');
  } catch (e: any) {
    if (e?.code === 'ENOENT') return [];
    throw e;
  }
  const out: T[] = [];
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    try {
      const rec = JSON.parse(t) as T;
      out.push(rec);
    } catch {
      // skip summary trailer or malformed line
    }
  }
  return out;
}

async function loadAllRuns(): Promise<Record<string, Record<string, ScoredRun>>> {
  let entries;
  try {
    entries = await readdir(RUNS_DIR, { withFileTypes: true });
  } catch (e: any) {
    if (e?.code === 'ENOENT') return {};
    throw e;
  }
  const dirs = entries.filter((e) => e.isDirectory());
  const loaded = await Promise.all(
    dirs.map(async (e) => {
      const recs = await readJsonl<ScoredRun>(join(RUNS_DIR, e.name, 'scored.jsonl'));
      // Drop the summary trailer (has no `id`).
      return [e.name, recs.filter((r) => r && (r as any).id)] as const;
    }),
  );
  const out: Record<string, Record<string, ScoredRun>> = {};
  for (const [name, recs] of loaded) {
    if (recs.length === 0) continue;
    const byId: Record<string, ScoredRun> = {};
    for (const r of recs) byId[r.id] = r;
    out[name] = byId;
  }
  return out;
}

let _allRunsCache: Record<string, Record<string, ScoredRun>> | null = null;
export async function loadAllRunsCached() {
  if (_allRunsCache) return _allRunsCache;
  _allRunsCache = await loadAllRuns();
  return _allRunsCache;
}

/** Parse `haiku-45-think-noprimer-v3` → metadata. */
export function parseRunMeta(id: string, primaryRunId: string | null): RunMeta {
  const lower = id.toLowerCase();
  const primer = !lower.includes('noprimer');
  const thinking = lower.includes('think');
  // model: take leading two segments split by `-`
  // e.g. haiku-45-* → haiku-4-5, sonnet-46-* → sonnet-4-6
  let model = id;
  const m = id.match(/^([a-z]+)-(\d)(\d)/);
  if (m) model = `${m[1]}-${m[2]}.${m[3]}`;

  const tagBits = [
    model,
    primer ? '+p' : '−p',
    thinking ? '+t' : '',
  ].filter(Boolean);
  const short = tagBits.join(' ').replace(/^([a-z]+)-(\d+\.\d+)/, (_, n, v) => `${n.slice(0, 1)}${v.replace('.', '')}`);
  // e.g. `haiku-4.5` -> `h45`; combined short: `h45 +p`
  return {
    id,
    label: id,
    short: short || id,
    model,
    primer,
    thinking,
    primary: id === primaryRunId,
  };
}

export interface CollectionData {
  collection: Collection;
  atoms: Atom[];
  runs: Record<string, Record<string, ScoredRun>>;
  runMeta: RunMeta[];
  primary: RunMeta | null;
}

let _cache: Map<string, CollectionData> | null = null;

export async function loadCollection(slug: string): Promise<CollectionData | null> {
  const collection = COLLECTIONS.find((c) => c.slug === slug);
  if (!collection) return null;
  if (!_cache) _cache = new Map();
  const cached = _cache.get(slug);
  if (cached) return cached;
  const atoms = await readJsonl<Atom>(join(REPO_ROOT, collection.jsonlPath));
  const allRuns = await loadAllRunsCached();
  const ids = new Set(atoms.map((a) => a.id));
  const runs: Record<string, Record<string, ScoredRun>> = {};
  for (const [runName, byId] of Object.entries(allRuns)) {
    const matched: Record<string, ScoredRun> = {};
    for (const [aid, rec] of Object.entries(byId)) {
      if (ids.has(aid)) matched[aid] = rec;
    }
    if (Object.keys(matched).length > 0) runs[runName] = matched;
  }
  const primaryRunId = collection.primaryRun && runs[collection.primaryRun]
    ? collection.primaryRun
    : (Object.keys(runs)[0] ?? null);
  const runMeta = Object.keys(runs).sort().map((id) => parseRunMeta(id, primaryRunId));
  const primary = runMeta.find((r) => r.primary) ?? null;
  const data: CollectionData = { collection, atoms, runs, runMeta, primary };
  _cache.set(slug, data);
  return data;
}

export async function loadAllCollections(): Promise<CollectionData[]> {
  const out: CollectionData[] = [];
  for (const c of COLLECTIONS) {
    const d = await loadCollection(c.slug);
    if (d) out.push(d);
  }
  return out;
}

export function groupKey(atom: Atom): string {
  const src = atom.source ?? '';
  if (src) {
    const stem = src.split('/').pop() ?? src;
    return stem.endsWith('.md') ? stem.slice(0, -3) : stem;
  }
  const aid = atom.id;
  return aid.includes('/') ? aid.split('/')[0] : aid;
}

/** Extract atom's leaf name after the last slash (e.g. ex1.b). */
export function atomLeaf(atomId: string): string {
  const i = atomId.lastIndexOf('/');
  return i === -1 ? atomId : atomId.slice(i + 1);
}

/** Extract atom's slash-prefix (everything before the leaf). */
export function atomPrefix(atomId: string): string {
  const i = atomId.lastIndexOf('/');
  return i === -1 ? '' : atomId.slice(0, i);
}
