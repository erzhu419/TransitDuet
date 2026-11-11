# Boundary Information Audit And Fit

The 32 acquisition tasks exited 0: 11,289,600 steps, 384 sampled state records
and 4,656 new paired futures. Audit every final public ledger and private row,
including prefix/window identity, exact A indices/seeds, cached-context equality
and W+tail reconstruction, before any fit. Old B2/3 and reserved E never fit.

Dense combines cached A0/1 with new A4-9 and the declared A10 windows: 96
windows/root, 780 labels, 84 windows with eight labels and 12 with nine. Wide
combines the common eight-prefix A0/1 bank with 24 new-prefix A4/5 banks:
384 windows/root, 768 labels. Use each window's full A mean, equal window weight.

Retain parent-scaled 120-column zero-intercept potential and DF 3. Fit exactly
one Dense and one Wide model/root, eight ridge fits total, no native steps or
baseline refits. Audit all banks first, save all eight models plus unchanged
controls before training metrics. No threshold, clipping or winner selection.
One 1CPU/2GiB scheduler job; raw banks and coefficients remain server-side.

## Completed Result

Task t214694 exited 0 on node004. All 32 private banks passed audit before the
eight DF-3 fits; the frozen bundle was saved before training readout. Exactly
eight ridge calls, zero baseline refits and zero new native steps were counted.
Audit, fitting and readout used 0.930 CPU seconds; acquisition used 5,588.446
Dense / 5,721.973 Wide CPU seconds. Only native-step budgets are equal.
The [public result](../results/plan_validity_boundary_information_20261111/fit_summary.json)
and [closure receipt](../results/plan_validity_boundary_information_20261111/fit_release.json)
retain costs, rosters and scheduler identities. Retrieved public JSON: 11,389 bytes.

Own-bank training MSE change versus the unchanged base model (negative is lower):

| Root | Dense bank | Wide bank |
| --- | ---: | ---: |
| PointMaze 1972110000 | -0.15% | +0.52% |
| PointMaze 1972120000 | -6.68% | -6.75% |
| Reacher 1972210000 | +1.97% | -8.30% |
| Reacher 1972220000 | -18.15% | -0.03% |

## Limits And Next Step

Own-bank training MSE has different label noise and state coverage in the two
strategies; it cannot rank independent prediction quality. Training metrics
are descriptive and receive no noise subtraction. E prefixes remain unopened.
After fit closure, the next separately released phase is the reserved new-prefix
E readout: eight new prefixes/root, E20-27, all frozen controls and Dense/Wide
on identical states and futures, 7,411,200 native steps total. No candidate is
selected from the training table; no E tasks were submitted in this fit stage.
This is not another training revision or a complete-episode policy claim.
