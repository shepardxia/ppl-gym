import type { Bucket, ScoredRun } from './types';

export type BucketTone = 'great' | 'good' | 'ok' | 'poor' | 'bad' | 'err' | 'na';

export const BUCKETS: { id: Bucket; label: string; glyph: string; tone: BucketTone; desc: string }[] = [
  { id: 'TV=0',   label: 'TV=0',   glyph: '●', tone: 'great', desc: 'distribution exact match' },
  { id: 'TV<.05', label: 'TV<.05', glyph: '◉', tone: 'good',  desc: 'very close distribution' },
  { id: 'TV<.5',  label: 'TV<.5',  glyph: '◐', tone: 'ok',    desc: 'moderate distribution disagreement' },
  { id: 'TV<1',   label: 'TV<1',   glyph: '◔', tone: 'poor',  desc: 'poor distribution match' },
  { id: 'TV=1',   label: 'TV=1',   glyph: '○', tone: 'bad',   desc: 'full distribution disagreement' },
  { id: 'val+',   label: 'val+',   glyph: '✓', tone: 'great', desc: 'value/scalar match' },
  { id: 'val-',   label: 'val-',   glyph: '✗', tone: 'poor',  desc: 'value mismatch' },
  { id: 'shape!', label: 'shape!', glyph: '△', tone: 'bad',   desc: 'wrong-shaped answer' },
  { id: 'fail',   label: 'fail',   glyph: '⚠', tone: 'err',   desc: 'execution crashed' },
  { id: 'no-run', label: 'no-run', glyph: '–', tone: 'na',    desc: 'this run did not score this atom' },
];

export const STRIP_ORDER: Bucket[] = BUCKETS.map((b) => b.id);

export const BUCKET_BY_ID: Record<Bucket, (typeof BUCKETS)[number]> =
  Object.fromEntries(BUCKETS.map((b) => [b.id, b])) as Record<Bucket, (typeof BUCKETS)[number]>;

export function tvValues(rec: ScoredRun | undefined | null): number[] {
  const m = rec?.evaluation?.metrics ?? {};
  return Object.entries(m).filter(([k]) => k.endsWith('tv')).map(([, v]) => v);
}

export function approxValues(rec: ScoredRun | undefined | null): number[] {
  const m = rec?.evaluation?.metrics ?? {};
  return Object.entries(m).filter(([k]) => k.endsWith('approx')).map(([, v]) => v);
}

export function maxTV(rec: ScoredRun | undefined | null): number | null {
  const tvs = tvValues(rec);
  return tvs.length ? Math.max(...tvs) : null;
}

export function klValues(rec: ScoredRun | undefined | null): number[] {
  const m = rec?.evaluation?.metrics ?? {};
  return Object.entries(m).filter(([k]) => k.endsWith('kl')).map(([, v]) => v);
}

export function maxKL(rec: ScoredRun | undefined | null): number | null {
  const kls = klValues(rec);
  return kls.length ? Math.max(...kls) : null;
}

export function bucketFor(rec: ScoredRun | undefined | null): Bucket {
  if (!rec) return 'no-run';
  const ev = rec.evaluation ?? {};
  if (!ev.gen?.executed) return 'fail';
  const cmp = ev.comparison ?? {};
  const err = cmp.error ?? '';
  if (cmp.ok === false && (err.startsWith('not a') || err.startsWith('samples must'))) {
    return 'shape!';
  }
  const worst = maxTV(rec);
  if (worst !== null) {
    if (worst === 0) return 'TV=0';
    if (worst < 0.05) return 'TV<.05';
    if (worst < 0.5) return 'TV<.5';
    if (worst < 1) return 'TV<1';
    return 'TV=1';
  }
  const exacts = approxValues(rec);
  if (exacts.length > 0) return exacts.every((v) => v === 1.0) ? 'val+' : 'val-';
  return 'shape!';
}

export function toneFor(bucket: Bucket): BucketTone {
  return BUCKET_BY_ID[bucket]?.tone ?? 'na';
}

export function glyphFor(bucket: Bucket): string {
  return BUCKET_BY_ID[bucket]?.glyph ?? '–';
}

export function descFor(bucket: Bucket): string {
  return BUCKET_BY_ID[bucket]?.desc ?? '';
}

export function fmtTV(n: number | null | undefined): string {
  if (n == null) return '—';
  if (n === 0) return '0.000';
  if (n < 0.001) return n.toExponential(1);
  return n.toFixed(3);
}

export function fmtKL(n: number | null | undefined): string {
  if (n == null) return '—';
  return n.toFixed(4);
}
