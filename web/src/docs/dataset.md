## Composition

115 problems across three source corpora (all WebPPL-native teaching material,
re-authored as language-neutral statements):

| corpus | problems | source |
|---|---|---|
| probmods2 | 70 | [Probabilistic Models of Cognition](http://probmods.org) exercises |
| dippl | 16 | [The Design and Implementation of Probabilistic Programming Languages](http://dippl.org) |
| forestdb | 29 | [Forest](http://forestdb.org) model repository (RSA / pragmatics models) |

8 further problems were retired during authoring (duplicates, degenerate
tasks); they are parked with reasons in the repo, not deleted.

## Answer types

| spec | count |
|---|---|
| distribution over finite labels | 43 (+4 draws-protocol) |
| distribution over integers | 16 |
| distribution over booleans | 11 (+1 draws-protocol) |
| distribution over reals | 10 |
| record (product of the above) | 23 |
| value (number / vector / label) | 7 |

Finite-domain specs declare their label space where it is small and closed
(39 problems), so vocabulary errors are caught mechanically rather than judged
as wrong answers.

## Realization columns

| language | status |
|---|---|
| WebPPL | 115/115 — solver-verified (re-derivation gate, report v2; 114 sonnet-gated, 1 opus-gated) |
| Pyro | 115/115 — cross-language-verified against the WebPPL ground truths |
| Stan | planned |
| memo, pluck | planned, with the language creators |

## What the stamps on a problem page mean

- **Phase A** — the ground truth's own multi-seed consistency: its measured
  noise floor and whether the problem can discriminate answers at all.
- **Phase B (solver gate)** — `accept` means at least one independent solver,
  reading only the statement, reproduced the ground truth within measured
  tolerance; the row records the gate model and protocol.
- The Pyro code shown on each page passed the cross-language gate against the
  WebPPL ground truth.
