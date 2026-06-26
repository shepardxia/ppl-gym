import { join, resolve } from 'node:path';
import { readJsonl } from './jsonl';

/** Corpus file stems, in display order. */
export const CORPORA = ['probmods2', 'dippl', 'forestdb', 'posteriordb'] as const;
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
  posteriordb: {
    label: 'posteriordb',
    description: 'Bayesian inference problems from posteriordb — applied regression, hierarchical, time-series, GP and ODE models, each with gold reference posterior draws. Realized in Stan.',
    sourceUrl: 'https://github.com/stan-dev/posteriordb',
    sourceName: 'stan-dev/posteriordb',
  },
};

// ─── Per-corpus realization columns + verification model ─────────────────────
// The browser compares two realization columns per problem: A (the ground-truth
// side) and B (the realization cross-checked against A). For the WebPPL corpora
// that is webppl (A) vs pyro (B); for posteriordb it is the gold reference draws
// (A — stored, no code) vs the Stan realization (B). `codeLang: null` marks the
// column as stored ground truth rather than program code.

export interface ColumnSpec {
  lang: string; label: string; short: string;
  /** Code language for syntax highlighting, or null for a stored-GT column (no code). */
  codeLang: string | null;
  /** For a stored-GT column (codeLang === null): the body text shown in place of code. */
  storedGt?: string;
}
export interface CorpusColumns {
  a: ColumnSpec;
  b: ColumnSpec;
  /** The language a solver for this corpus writes — selects the system prompt. */
  solverLang: string;
  /** Whether a solver re-derivation gate (phase B) applies to this corpus. */
  hasSolverGate: boolean;
  /** Name of the solver primer to show in the statement context (key into PRIMERS), or null. */
  primer: string | null;
}

const WEBPPL_COLUMNS: CorpusColumns = {
  a: { lang: 'webppl', label: 'WebPPL ground truth', short: 'webppl', codeLang: 'webppl' },
  b: { lang: 'pyro', label: 'Pyro realization', short: 'pyro', codeLang: 'python' },
  solverLang: 'webppl',
  hasSolverGate: true,
  primer: 'webppl',
};

export const CORPUS_COLUMNS: Record<Corpus, CorpusColumns> = {
  probmods2: WEBPPL_COLUMNS,
  dippl: WEBPPL_COLUMNS,
  forestdb: WEBPPL_COLUMNS,
  posteriordb: {
    a: {
      lang: 'reference', label: 'gold reference posterior', short: 'reference', codeLang: null,
      storedGt: 'Gold reference posterior draws from posteriordb (10 NUTS chains, R-hat ≈ 1). '
        + 'Not program code — the realized marginals are the answer overlay below.',
    },
    b: { lang: 'stan', label: 'Stan realization', short: 'stan', codeLang: 'stan' },
    solverLang: 'stan',
    hasSolverGate: false,
    primer: null,
  },
};

/** All realization languages any corpus column references. */
const REALIZATION_LANGS = Array.from(
  new Set(Object.values(CORPUS_COLUMNS).flatMap((c) => [c.a.lang, c.b.lang])),
);

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
  code?: string;
  /** When false, the problem is not realizable in this language; `reason` says why
   *  and there is no `code`. See data/REALIZATIONS.md §Per-language availability. */
  available?: boolean;
  reason?: string;
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
  status: 'pass' | 'fail' | 'ill_posed' | 'error' | 'unavailable';
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
  /** Column A — the ground-truth side (webppl, or posteriordb gold reference). */
  realizationA: Realization | null;
  /** Column B — the realization cross-checked against A (pyro, or stan). */
  realizationB: Realization | null;
  gateA: GatePhaseA | null;
  gateB: GatePhaseB | null;
  crosscheck: GateCross | null;
  corpus: Corpus;
}

// ─── Loaders ────────────────────────────────────────────────────────────────

let _cache: ProblemRecord[] | null = null;

export async function loadAllProblems(): Promise<ProblemRecord[]> {
  if (_cache) return _cache;


  // Read realization columns, gate reports, and every corpus file in parallel.
  const [realLangRows, gateARows, gateBRows, crossRows, corpusRows] = await Promise.all([
    Promise.all(REALIZATION_LANGS.map((l) =>
      readJsonl<Realization>(join(DATA_DIR, 'realizations', `${l}.jsonl`)))),
    readJsonl<GatePhaseA>(join(DATA_DIR, 'problems', '_gate_report.jsonl')),
    readJsonl<GatePhaseB>(join(DATA_DIR, 'problems', '_gate_solver_report.jsonl')),
    readJsonl<GateCross>(join(DATA_DIR, 'problems', '_gate_crosscheck_report.jsonl')),
    Promise.all(CORPORA.map((c) => readJsonl<Problem>(join(DATA_DIR, 'problems', `${c}.jsonl`)))),
  ]);

  // Index realizations by language → problem_id.
  const realByLang = new Map<string, Map<string, Realization>>();
  REALIZATION_LANGS.forEach((lang, i) => {
    const m = new Map<string, Realization>();
    for (const r of realLangRows[i]) m.set(r.problem_id, r);
    realByLang.set(lang, m);
  });
  const realOf = (lang: string, pid: string) => realByLang.get(lang)?.get(pid) ?? null;

  const gateAByPid = new Map<string, GatePhaseA>();
  for (const r of gateARows) gateAByPid.set(r.problem_id, r);

  const gateBByPid = new Map<string, GatePhaseB>();
  for (const r of gateBRows) gateBByPid.set(r.problem_id, r);

  const crossByPid = new Map<string, GateCross>();
  for (const r of crossRows) crossByPid.set(r.problem_id, r);

  const out: ProblemRecord[] = [];
  for (let i = 0; i < CORPORA.length; i++) {
    const corpus = CORPORA[i];
    const cols = CORPUS_COLUMNS[corpus];
    const problems = corpusRows[i] as Problem[];
    for (const problem of problems) {
      // Skip retired problems.
      if (problem.status?.review === 'retired') continue;
      out.push({
        problem,
        corpus,
        realizationA: realOf(cols.a.lang, problem.problem_id),
        realizationB: realOf(cols.b.lang, problem.problem_id),
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
  const out = Object.fromEntries(CORPORA.map((c) => [c, [] as ProblemRecord[]])) as Record<Corpus, ProblemRecord[]>;
  for (const rec of all) out[rec.corpus].push(rec);
  return out;
}

/** Per-corpus verification bucket — solver+cross for the WebPPL corpora, cross
 *  only for posteriordb (no solver gate). Single source of truth for the sidebar
 *  glyphs, filter chips, and per-problem badges. */
export function verificationBucket(r: ProblemRecord): 'verified' | 'partial' | 'attention' | 'na' {
  const cols = CORPUS_COLUMNS[r.corpus];
  const xOk = r.crosscheck?.status === 'pass';
  if (!cols.hasSolverGate) {
    // cross-language gate is the whole verification story here
    if (r.crosscheck == null || r.crosscheck.status === 'unavailable') return 'na';
    return xOk ? 'verified' : 'attention';
  }
  const sOk = r.gateB?.status === 'accept';
  if (!r.gateB && !r.crosscheck) return 'na';
  if (sOk && xOk) return 'verified';
  if (sOk || xOk) return 'partial';
  return 'attention';
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

// ─── Grouping helpers (sidebar) ──────────────────────────────────────────────

const _CORPUS_PREFIX = new RegExp('^(' + CORPORA.join('|') + ')-');

/** 'probmods2-conditioning/ex5.b' → 'conditioning' (chapter within corpus). */
export function problemChapter(problemId: string): string {
  return problemId.split('/')[0].replace(_CORPUS_PREFIX, '');
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
  const rows = await readJsonl<GtAnswerRow>(join(DATA_DIR, 'problems', '_gt_answers.jsonl'));
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
