// Source of truth lives at data/prompts/{webppl,pyro,stan}_system_base.txt +
// {webppl,pyro}_primer.txt; eval/prompt.py reads the same files. Vite's `?raw`
// import inlines the file content as a string at build time — no Node fs at runtime.

// @ts-expect-error Vite plugin types for ?raw imports
import webpplBase from '../../../data/prompts/webppl_system_base.txt?raw';
// @ts-expect-error Vite plugin types for ?raw imports
import pyroBase from '../../../data/prompts/pyro_system_base.txt?raw';
// @ts-expect-error Vite plugin types for ?raw imports
import stanBase from '../../../data/prompts/stan_system_base.txt?raw';
// @ts-expect-error Vite plugin types for ?raw imports
import webpplPrimer from '../../../data/prompts/webppl_primer.txt?raw';

export const PROMPT_VERSION = 'v2-atom';

const trim = (s: string) => s.replace(/\n+$/, '');

export const SYSTEM_PROMPT_BASE: string = trim(webpplBase as string);   // legacy /c/ browser
export const WEBPPL_PRIMER: string = trim(webpplPrimer as string);

// Per-solver-language system base — what a solver writing that language is told.
const SYSTEM_BASE: Record<string, string> = {
  webppl: trim(webpplBase as string),
  pyro: trim(pyroBase as string),
  stan: trim(stanBase as string),
};

/** Solver primers by name (CorpusColumns.primer keys into this). Only webppl has one. */
export const PRIMERS: Record<string, string> = { webppl: WEBPPL_PRIMER };

/** System prompt shown for a corpus whose solver writes `lang`. */
export function systemBaseFor(lang: string): string {
  return SYSTEM_BASE[lang] ?? SYSTEM_BASE.webppl;
}

/** Primer text for a primer name, or '' if none. */
export function primerFor(name: string | null): string {
  return name ? (PRIMERS[name] ?? '') : '';
}

/** Run names follow `<model>-<primer-flag>-...` convention; "noprimer" → false. */
export function runHasPrimer(runName: string): boolean {
  return !runName.toLowerCase().includes('noprimer');
}
