# Wide A-Only Fit And Prefix Transfer

Reuse all 32 Wide training prefixes/root, 384 windows and two A labels/window:
cached prefixes use A0/1; the new 24 use A4/5. No Dense expansion, B/E access,
new simulation, baseline refit, capacity search or frozen-model replacement.
Reproduce every frozen Wide A-mean loss and roster before new diagnostic fits.

Fit column 0, evaluate column 1, then reverse. Each view fits 32 whole-prefix
leave-one-out DF-3 parent-scaled potentials on 31 prefixes/372 windows each.
For a probe prefix, its own excluded fold measures prefix transfer. The other
31 folds include it and measure independent-future performance at the same
training size and capacity. Average their individual squared losses, not the
squared loss of their mean prediction; this is not an ensemble control.
The same 64 models/root provide every comparison: 256 fits total, no extra
matched-control fits. One unpinned 1CPU/2GiB scheduler job on node001-node006.

Report in-fit, matched-future and prefix-held MSE, paired differences versus zero
and versus matched-future, each A view, cached8/new24 strata, window halves and
all prefixes. Record 98,304 diagnostic window predictions plus 1,536 frozen-model
source checks. Models and predictions stay private on the server; public losses
and cost summaries alone are retrieved. Fifteen focused tests passed.

## Limits And Next Step

These are exposed training-A development diagnostics, not new confirmation.
Overlapping folds and reciprocal A views are not independent fitted-policy
seeds; no CI or view selection. A matched-future gain with prefix-held failure
would indicate a transfer gap within this bank. Failure already on matched
futures does not by itself distinguish representation limits from noisy training
labels. Keep both outcomes, no automatic fitting change or policy release.
E remains reporting-only evidence; original confirmation stays on HOLD.
