Machine gates catch inconsistency; they can't catch a problem that is
consistently wrong, an unidiomatic realization, or a statement that
misrepresents its source. That's the review.

## On each problem page

- **Statement** — does it determine the answer? Anything missing, leaked, or untrue to the source model?
- **Realizations** — is the WebPPL faithful? Is the Pyro *idiomatic*, not just numerically right?
- **Verification panel** — big noise floors and opus-gated rows are where expert eyes matter most.

## Feedback

Vote 👍/👎 and comment on any problem page (display name only, no account).
Every report is triaged to an evidenced conclusion; fixes go back through the
gates before shipping.

## Disagreements

Tiebreaker, in order: source intent → statement text → measured reproduction.
Full history: [`_gate_triage.md`](https://github.com/shepardxia/ppl-gym/blob/main/data/problems/_gate_triage.md).
