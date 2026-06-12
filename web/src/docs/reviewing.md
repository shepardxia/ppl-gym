This site exists so PPL experts can sanity-check the dataset. Machine gates
catch inconsistency; they cannot catch a problem that is consistently wrong, an
unidiomatic realization, or a statement that misrepresents its source model.
That's the review.

## What to look at on a problem page

1. **The statement** — does it actually determine the answer? Is anything
   underspecified, overspecified (leaking the solution), or simply not how the
   original model works?
2. **The realizations** — is the WebPPL faithful to the source model? Is the
   Pyro *idiomatic* Pyro, or technically-correct-but-weird? (Every Pyro
   realization matches the WebPPL answer numerically; whether it's code a Pyro
   user would endorse is exactly what we can't verify mechanically.)
3. **The gate panel** — large noise floors or opus-gated rows mark the
   problems where machine verification had the least traction; expert eyes are
   worth the most there.

## Leaving feedback

Vote 👍/👎 and comment directly on any problem page — the first feedback asks
for a display name; no account needed. Feedback lands in a database we triage
with the same rules as gate findings: every report gets investigated to an
evidenced conclusion, and fixes flow back through the gates before shipping.

## Disagreements

When a reviewer, a solver, or another language disagrees with a ground truth,
the tiebreaker is evidence, in this order: the textbook source's intent, the
statement's text, and measured reproduction of the divergence. The full
investigation history (every overturned verdict included) is in
[`data/problems/_gate_triage.md`](https://github.com/shepardxia/ppl-gym/blob/main/data/problems/_gate_triage.md).
