// Source of truth lives at data/prompts/{system_base,webppl_primer}.txt;
// eval/prompt.py reads the same files. Vite's `?raw` import inlines the
// file content as a string at build time — no Node fs at runtime.

// @ts-expect-error Vite plugin types for ?raw imports
import systemBase from '../../../data/prompts/system_base.txt?raw';
// @ts-expect-error Vite plugin types for ?raw imports
import webpplPrimer from '../../../data/prompts/webppl_primer.txt?raw';

export const PROMPT_VERSION = 'v2-atom';

export const SYSTEM_PROMPT_BASE: string = (systemBase as string).replace(/\n+$/, '');
export const WEBPPL_PRIMER: string = (webpplPrimer as string).replace(/\n+$/, '');

export function systemPrompt(withPrimer: boolean): string {
  return withPrimer ? `${SYSTEM_PROMPT_BASE}\n\n${WEBPPL_PRIMER}` : SYSTEM_PROMPT_BASE;
}

/** Run names follow `<model>-<primer-flag>-...` convention; "noprimer" → false. */
export function runHasPrimer(runName: string): boolean {
  return !runName.toLowerCase().includes('noprimer');
}
