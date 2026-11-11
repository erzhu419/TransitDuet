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

## Limits And Next Step

Own-bank training MSE has different label noise and state coverage in the two
strategies; it cannot rank independent prediction quality. Training metrics
are descriptive and receive no noise subtraction. E prefixes remain unopened.
After fit closure, the next separately released phase is the reserved new-prefix
E readout, not another training revision or a complete-episode policy claim.
