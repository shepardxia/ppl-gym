Model baselines under the current pipeline haven't been run yet — this page
will hold them when they exist.

What will appear here:

- per-model solve rates on each language column (WebPPL, Pyro), scored by the
  same answer algebra and measured tolerances as the gates;
- breakdowns by answer type and source corpus;
- the gap between high-resource (Python/Pyro) and low-resource (WebPPL)
  performance — one of the dataset's central questions.

For now, the closest proxy is the gate campaign itself: with fully determinate
statements, a 2-sample sonnet-class gate re-derives 112/115 WebPPL ground
truths — see [Updates](/updates).
