import { readFile } from 'node:fs/promises';

/** Read a JSONL file. Missing file → []; blank and malformed lines are skipped. */
export async function readJsonl<T>(absPath: string): Promise<T[]> {
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
      // skip summary trailer or malformed line
    }
  }
  return out;
}
