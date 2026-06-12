import type { APIRoute } from 'astro';

type Vote = 'up' | 'down' | 'neutral';

interface FeedbackBody {
  atom_id: string;
  collection: string;
  dataset_version: string;
  rater_id: string;
  rater_name: string;
  vote: Vote;
  comment?: string;
}

const VOTES: Vote[] = ['up', 'down', 'neutral'];

function bad(msg: string, status = 400) {
  return new Response(JSON.stringify({ error: msg }), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

type FieldSpec = { key: keyof FeedbackBody; max: number; required?: boolean };
const FIELD_SPECS: FieldSpec[] = [
  { key: 'atom_id', max: 256 },
  { key: 'collection', max: 64 },
  { key: 'dataset_version', max: 64 },
  { key: 'rater_id', max: 64 },
  { key: 'rater_name', max: 80 },
];

function validate(body: unknown): FeedbackBody | string {
  if (!body || typeof body !== 'object') return 'body must be a JSON object';
  const b = body as Record<string, unknown>;
  for (const { key, max } of FIELD_SPECS) {
    const v = b[key];
    if (typeof v !== 'string' || v.length === 0) return `${key} must be a non-empty string`;
    if (v.length > max) return `${key} exceeds max length ${max}`;
  }
  if (!VOTES.includes(b.vote as Vote)) return `vote must be one of ${VOTES.join('/')}`;
  if (b.comment !== undefined) {
    if (typeof b.comment !== 'string') return 'comment must be a string';
    if (b.comment.length > 4000) return 'comment exceeds max length 4000';
  }
  return b as unknown as FeedbackBody;
}

export const POST: APIRoute = async ({ request, locals }) => {
  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return bad('invalid JSON');
  }
  const body = validate(raw);
  if (typeof body === 'string') return bad(body);

  const db = locals.runtime.env.DB;
  if (!db) return bad('database not configured', 500);

  // Clients post `atom_id` (the original widget contract); the D1 column was
  // renamed to problem_id in migration 0002. Same id space either way.
  await db
    .prepare(
      `INSERT INTO feedback
       (problem_id, collection, dataset_version, rater_id, rater_name, vote, comment, visibility)
       VALUES (?, ?, ?, ?, ?, ?, ?, 'private')`,
    )
    .bind(
      body.atom_id, body.collection, body.dataset_version,
      body.rater_id, body.rater_name, body.vote, body.comment ?? '',
    )
    .run();
  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
};

export const GET: APIRoute = () =>
  new Response(JSON.stringify({ error: 'feedback is private' }), {
    status: 403,
    headers: { 'content-type': 'application/json' },
  });
