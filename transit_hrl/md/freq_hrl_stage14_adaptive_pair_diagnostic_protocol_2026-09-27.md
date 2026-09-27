# Stage-14 Adaptive-Continuation Timing Diagnostic

Stage-13 supports local score validity under static future planning times,
but the deployed keep action waits just one check, not necessarily to the
deadline. This diagnostic holds the controller, Stage-12 predictor/threshold,
roots 209011/209061, and Stage-13 sampled path/bin roster fixed.

At each of those causal prefixes, run three complete episodes:

1. Plan now and resume the frozen trigger on subsequent bins.
2. Skip this check and resume the trigger at the next check, five steps later.
3. Wait to offset 25 in this bin, then resume the trigger on subsequent bins.

Every branch uses its own causal observations and the same exogenous path;
all retain exactly one call per bin. The factual arm must reproduce the saved
Stage-12 decision times, ISE, and return. All arms must share an exact causal
prefix. The 64/61 opportunities cost 230,400/219,600 intervention replay
steps, separate from replaying the frozen controller training.

Report now-versus-one-check and now-versus-deadline 50-step/full-episode
contrasts, grouped by factual early/deadline decisions; compare the latter
with the same Stage-13 static-continuation contrast. No model fitting,
threshold selection, new roots, or performance claim is part of this test.
Preflight uses root 208001 and one bin per class before the two-root run.
Use scheduleurm dynamic node001-node006 placement (one CPU, 1536 MB/cell);
transfer only compact JSON inputs and results.
