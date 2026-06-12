import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

/** Corpus file stems, in display order. */
export const CORPORA = ['probmods2', 'dippl', 'forestdb'] as const;
export type Corpus = (typeof CORPORA)[number];

/** Display metadata per corpus — one home for names and descriptions. */
export const CORPUS_META: Record<Corpus, { label: string; description: string; sourceUrl: string; sourceName: string }> = {
  probmods2: {
    label: 'ProbMods',
    description: 'Exercises from Probabilistic Models of Cognition — Bayesian cognition and probabilistic inference.',
    sourceUrl: 'http://probmods.org',
    sourceName: 'probmods.org',
  },
  dippl: {
    label: 'DIPPL',
    description: 'Exercises from The Design and Implementation of Probabilistic Programming Languages — PPL semantics, enumeration, particle filtering.',
    sourceUrl: 'http://dippl.org',
    sourceName: 'dippl.org',
  },
  forestdb: {
    label: 'Forest',
    description: 'Published WebPPL models from the Forest repository — RSA pragmatics, social reasoning, concept learning.',
    sourceUrl: 'http://forestdb.org',
    sourceName: 'forestdb.org',
  },
};

// Astro's build/prerender runs with CWD = the Astro project root (web/).
// import.meta.url-based resolution BREAKS after Vite bundles this into
// dist/_worker.js/chunks/ — keep process.cwd() exactly as-is.
const REPO_ROOT = resolve(process.cwd(), '..');
const DATA_DIR = join(REPO_ROOT, 'data');

// ─── Raw JSONL record types ──────────────────────────────────────────────────

export interface Problem {
  problem_id: string;
  provenance: {
    source: string;
    origin_language: string;
    collection?: string;
  };
  statement: {
    given: string;
    model: string;
    query: string;
  };
  answer_spec: AnswerSpec;
  status: {
    review: 'draft' | 'reviewed' | 'retired';
    notes?: string;
  };
}

export type AnswerSpec =
  | { kind: 'value'; domain: string; estimated?: boolean; support?: unknown[] }
  | { kind: 'dist';  domain: string; labels?: unknown; protocol?: string; support?: unknown[] }
  | { kind: 'record'; fields: Record<string, AnswerSpec> }
  | Record<string, unknown>;

export interface Realization {
  problem_id: string;
  language: string;
  code: string;
}

export interface GatePhaseA {
  problem_id: string;
  status: 'ok' | 'ill_posed' | 'error';
  floor?: number;
  metric?: string;
  k?: number;
  n_runs?: number;
  runtime_sec?: number;
}

export interface GatePhaseB {
  problem_id: string;
  status: 'accept' | 'gt_suspect' | 'underdetermined' | 'solver_failure';
  n_pass?: number;
  distances?: number[];
  solver_agree_distance?: number;
  gt_floor?: number;
  memorization_suspect?: boolean;
  errors?: string[];
  gate_model?: string;
  timeout?: number;
  n_solvers?: number;
}

export interface GateCross {
  problem_id: string;
  status: 'pass' | 'fail' | 'ill_posed' | 'error';
  distance?: number;
  tol?: number;
  target_floor?: number;
  reference_floor?: number;
  metric?: string;
  language?: string;
  reference?: string;
}

// ─── Joined record exposed to pages ─────────────────────────────────────────

export interface ProblemRecord {
  problem: Problem;
  realization: Realization | null;
  pyroRealization: Realization | null;
  gateA: GatePhaseA | null;
  gateB: GatePhaseB | null;
  crosscheck: GateCross | null;
  corpus: Corpus;
}

// ─── JSONL reader ────────────────────────────────────────────────────────────

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
      out.push(JSON.parse(t) as T);
    } catch {
      // skip malformed lines
    }
  }
  return out;
}

// ─── Loaders ────────────────────────────────────────────────────────────────

let _cache: ProblemRecord[] | null = null;

export async function loadAllProblems(): Promise<ProblemRecord[]> {
  if (_cache) return _cache;


  // Read all four files in parallel.
  const [realizations, pyroRealizations, gateARows, gateBRows, crossRows, ...corpusRows] = await Promise.all([
    readJsonl<Realization>(join(DATA_DIR, 'realizations', 'webppl.jsonl')),
    readJsonl<Realization>(join(DATA_DIR, 'realizations', 'pyro.jsonl')),
    readJsonl<GatePhaseA>(join(DATA_DIR, 'problems', '_gate_report.jsonl')),
    readJsonl<GatePhaseB>(join(DATA_DIR, 'problems', '_gate_solver_report.jsonl')),
    readJsonl<GateCross>(join(DATA_DIR, 'problems', '_gate_crosscheck_report.jsonl')),
    ...CORPORA.map((c) => readJsonl<Problem>(join(DATA_DIR, 'problems', `${c}.jsonl`))),
  ]);

  // Index by problem_id for O(1) joins.
  const realizationByPid = new Map<string, Realization>();
  for (const r of realizations) realizationByPid.set(r.problem_id, r);
  const pyroByPid = new Map<string, Realization>();
  for (const r of pyroRealizations) pyroByPid.set(r.problem_id, r);

  const gateAByPid = new Map<string, GatePhaseA>();
  for (const r of gateARows) gateAByPid.set(r.problem_id, r);

  const gateBByPid = new Map<string, GatePhaseB>();
  for (const r of gateBRows) gateBByPid.set(r.problem_id, r);

  const crossByPid = new Map<string, GateCross>();
  for (const r of crossRows) crossByPid.set(r.problem_id, r);

  const out: ProblemRecord[] = [];
  for (let i = 0; i < CORPORA.length; i++) {
    const corpus = CORPORA[i];
    const problems = corpusRows[i] as Problem[];
    for (const problem of problems) {
      // Skip retired problems.
      if (problem.status?.review === 'retired') continue;
      out.push({
        problem,
        corpus,
        realization: realizationByPid.get(problem.problem_id) ?? null,
        pyroRealization: pyroByPid.get(problem.problem_id) ?? null,
        gateA: gateAByPid.get(problem.problem_id) ?? null,
        gateB: gateBByPid.get(problem.problem_id) ?? null,
        crosscheck: crossByPid.get(problem.problem_id) ?? null,
      });
    }
  }

  _cache = out;
  return out;
}

export async function loadAllProblemsGrouped(): Promise<Record<Corpus, ProblemRecord[]>> {
  const all = await loadAllProblems();
  const out = { probmods2: [] as ProblemRecord[], dippl: [] as ProblemRecord[], forestdb: [] as ProblemRecord[] };
  for (const rec of all) out[rec.corpus].push(rec);
  return out;
}

// ─── Slug helpers ────────────────────────────────────────────────────────────

/** problem_id → URL slug: replace '/' with '__' */
export function problemSlug(problemId: string): string {
  return problemId.replace(/\//g, '__');
}

/** URL slug → problem_id: replace '__' with '/' */
export function slugToProblemId(slug: string): string {
  return slug.replace(/__/g, '/');
}

// ─── Answer spec helpers ─────────────────────────────────────────────────────

export function specKind(spec: AnswerSpec): string {
  return (spec as any).kind ?? '?';
}

export function specDomain(spec: AnswerSpec): string {
  return (spec as any).domain ?? '';
}

export function specChip(spec: AnswerSpec): string {
  const k = specKind(spec);
  const d = specDomain(spec);
  if (k === 'record') {
    const fields = (spec as any).fields ?? {};
    return `record(${Object.keys(fields).join(', ')})`;
  }
  return d ? `${k}/${d}` : k;
}

// ─── Dataset stats (computed, used by the landing page and problems hub) ────

export interface DatasetStats {
  total: number;
  byCorpus: { corpus: Corpus; count: number }[];
  bySpec: { chip: string; count: number }[];
  webpplVerified: number;
  pyroVerified: number;
}

export async function datasetStats(): Promise<DatasetStats> {
  const all = await loadAllProblems();
  const byCorpus = CORPORA.map((c) => ({
    corpus: c,
    count: all.filter((r) => r.corpus === c).length,
  }));
  const specCounts = new Map<string, number>();
  for (const r of all) {
    const k = specKind(r.problem.answer_spec) === 'record'
      ? 'record'
      : specChip(r.problem.answer_spec);
    specCounts.set(k, (specCounts.get(k) ?? 0) + 1);
  }
  const bySpec = [...specCounts.entries()]
    .map(([chip, count]) => ({ chip, count }))
    .sort((a, b) => b.count - a.count);
  return {
    total: all.length,
    byCorpus,
    bySpec,
    webpplVerified: all.filter((r) => r.gateB?.status === 'accept').length,
    pyroVerified: all.filter((r) => r.crosscheck?.status === 'pass').length,
  };
}

// ─── Grouping helpers (sidebar) ──────────────────────────────────────────────

/** 'probmods2-conditioning/ex5.b' → 'conditioning' (chapter within corpus). */
export function problemChapter(problemId: string): string {
  const head = problemId.split('/')[0];
  return head.replace(/^(probmods2|dippl|forestdb)-/, '');
}

/** 'probmods2-conditioning/ex5.b' → 'ex5.b'. */
export function problemLeaf(problemId: string): string {
  const i = problemId.indexOf('/');
  return i === -1 ? problemId : problemId.slice(i + 1);
}

// ─── Canonical GT answers (data/problems/_gt_answers.jsonl) ─────────────────

export interface GtAnswerRow {
  problem_id: string;
  language: string;
  answer?: Record<string, unknown>;
  error?: string;
}

let _gtCache: Map<string, Record<string, GtAnswerRow>> | null = null;

export async function loadGtAnswers(): Promise<Map<string, Record<string, GtAnswerRow>>> {
  if (_gtCache) return _gtCache;
  const rows = await (async () => {
    const { readFile } = await import('node:fs/promises');
    try {
      const text = await readFile(join(DATA_DIR, 'problems', '_gt_answers.jsonl'), 'utf8');
      return text.split('\n').filter(Boolean).map((l) => JSON.parse(l) as GtAnswerRow);
    } catch { return [] as GtAnswerRow[]; }
  })();
  const out = new Map<string, Record<string, GtAnswerRow>>();
  for (const r of rows) {
    if (!out.has(r.problem_id)) out.set(r.problem_id, {});
    out.get(r.problem_id)![r.language] = r;
  }
  _gtCache = out;
  return out;
}

/** Canonical answer wire → the RenderOutput shape the chart renderer expects. */
export function canonicalToRenderOutput(a: Record<string, unknown> | undefined): unknown {
  if (!a) return null;
  const kind = a.kind as string;
  if (kind === 'dist_enum') {
    return {
      kind: 'distribution',
      support: (a.support as unknown[]).map((s) => typeof s === 'string' ? s : JSON.stringify(s)),
      probs: a.probs as number[],
    };
  }
  if (kind === 'cloud') {
    // histogram the samples into {kind: 'samples', support, counts}
    const counts = new Map<string, number>();
    for (const s of a.samples as unknown[]) {
      const k = typeof s === 'number' ? String(Math.round(s * 1000) / 1000) : JSON.stringify(s);
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
    const entries = [...counts.entries()];
    // numeric supports sort numerically
    entries.sort((x, y) => {
      const nx = Number(x[0]), ny = Number(y[0]);
      if (!Number.isNaN(nx) && !Number.isNaN(ny)) return nx - ny;
      return x[0] < y[0] ? -1 : 1;
    });
    return { kind: 'samples', support: entries.map((e) => e[0]), counts: entries.map((e) => e[1]) };
  }
  if (kind === 'dist_param') {
    const params = a.params as Record<string, number>;
    const inner = Object.entries(params).map(([k, v]) => `${k}: ${v}`).join(', ');
    return { kind: 'value', value: `${a.family}(${inner})` };
  }
  if (kind === 'exact') {
    return { kind: 'value', value: a.value };
  }
  if (kind === 'record') {
    const fields: Record<string, unknown> = {};
    for (const [n, v] of Object.entries(a.fields as Record<string, Record<string, unknown>>)) {
      fields[n] = canonicalToRenderOutput(v);
    }
    return { kind: 'record', fields };
  }
  return null;
}
