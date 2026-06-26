// The status vocabulary is owned by the Python layer (eval/algebra.py judge
// statuses; eval/gate.py phase-A/B statuses) and documented in data/SCHEMA.md
// §Status vocabulary. Keep this mapping in sync when a status is added.
// Status tone mapping for problem-centric gate statuses. Supersedes the TV/bucket
// vocabulary for these pages; buckets.ts is retained for the legacy /c/ atom browser.

export type StatusTone = 'good' | 'warn' | 'bad' | 'na';

/** Map a gate status string to a visual tone. */
export function statusTone(status: string | null | undefined): StatusTone {
  if (!status) return 'na';
  switch (status) {
    // Phase-B solver statuses
    case 'accept':       return 'good';
    case 'gt_suspect':   return 'warn';
    case 'underdetermined': return 'warn';
    case 'solver_failure': return 'bad';
    // Phase-A statuses
    case 'ok':           return 'good';
    case 'ill_posed':    return 'bad';
    case 'error':        return 'bad';
    // judge()/score statuses
    case 'pass':         return 'good';
    case 'fail':         return 'bad';
    case 'malformed':    return 'bad';
    case 'exec_error':   return 'bad';
    // Problem review statuses
    case 'reviewed':     return 'good';
    case 'draft':        return 'warn';
    case 'retired':      return 'na';
    default:             return 'na';
  }
}

export function statusLabel(status: string | null | undefined): string {
  if (!status) return '—';
  switch (status) {
    case 'accept':          return 'accept';
    case 'gt_suspect':      return 'GT suspect';
    case 'underdetermined': return 'underdetermined';
    case 'solver_failure':  return 'solver failure';
    case 'ok':              return 'ok';
    case 'ill_posed':       return 'ill-posed';
    case 'error':           return 'error';
    case 'reviewed':        return 'reviewed';
    case 'draft':           return 'draft';
    default:                return status;
  }
}
