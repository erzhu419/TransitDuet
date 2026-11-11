# Wide A-Only Fit And Prefix Transfer

Reuse all 32 Wide training prefixes/root, 384 windows and two A labels/window:
cached prefixes use A0/1; the new 24 use A4/5. No Dense expansion, B/E targets,
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

## Completed Result

t214719 exited 0 on node004: exactly 256 fits, 98,304 diagnostic and 1,536
frozen-source window predictions, zero native steps or baseline refits. Measured
phases used 1.890 CPU seconds. Only 20,757 compact public bytes were retrieved;
all fold models and predictions remain remote. See [summary](../results/plan_validity_boundary_wide_transfer_20261111/summary.json)
and [receipt](../results/plan_validity_boundary_wide_transfer_20261111/release.json).

MSE changes versus zero, two-view descriptive means, negative is better:

| Root | In fit | Matched independent future | Prefix held |
| --- | ---: | ---: | ---: |
| PointMaze 1972110000 | -7.41% | -3.70% | -2.92% |
| PointMaze 1972120000 | -9.76% | -7.80% | -5.99% |
| Reacher 1972210000 | -6.68% | -5.25% | +5.33% |
| Reacher 1972220000 | -4.17% | +0.76% | +0.01% |

Reacher1 has matched-future gains but prefix-held losses in both A views,
both cached8/new24 strata and both window halves. The matched-size prefix gap
is +0.003574 MSE: local predictable signal exists, yet transfer fails here.
Reacher2 matched-future changes are +0.018%/+1.733%, already not beating zero;
prefix-held changes are -1.051%/+1.396%, so the near-zero mean hides a view
conflict. This is not simply the same prefix-transfer failure as Reacher1.
PointMaze keeps overall gains in both views, with a smaller prefix gap; not every
individual window half improves. No development diagnostic is promoted to E proof.

## Limits And Next Step

These are exposed training-A development diagnostics, not new confirmation.
Overlapping folds and reciprocal A views are not independent fitted-policy
seeds; no CI or view selection. A matched-future gain with prefix-held failure
would indicate a transfer gap within this bank. Failure already on matched
futures does not by itself distinguish representation limits from noisy training
labels. Keep both outcomes, no automatic fitting change or policy release.
E remains reporting-only evidence; original confirmation stays on HOLD.
Next prioritize Reacher1's train-only prefix influence/geometry attribution using
these saved folds. For Reacher2, retain the unresolved noisy-label versus model
limitation distinction rather than infer that extra capacity is required. No
additional fitting, simulation or deployment is released by this result.
