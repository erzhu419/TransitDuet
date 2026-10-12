# Stage200: frozen-cost wait calibration

Keep Stage199's count-weighted ridge and all three deployed heads unchanged.
Four inner scene folds use their own training-only scales and ridge0.1.
Mean8 calibration states appear once; singleton states enter auxiliary fits
but not the residual estimate. Partition seed200 stratifies root/regime and
mean8 coverage. Only outer-training scenes participate.

Margins: zero, positive selected-override mean wait residual, or that mean
plus z90 times the group-ratio standard error. The last is primary. Identity
wait remains exactly zero and cost ranking is unchanged. Gate: strictly
negative cost and nonpositive wait on both roots across all three splits.

Run: `native_transit_wait_margin_stage200_development_20261013_r1`.
47 fit tasks plus one report,188 auxiliary solves,47 reused deployed models;
0 optimizer/actor/native updates,1CPU/3072MB, unbound node001-node006.
Tests:31 passed. Models, targets and detailed calibration remain server-only.

Next: evaluate all predeclared margins without tuning on held-out results.
Limits: this is an empirical mean-wait correction, not formal OOD coverage,
CI-supported improvement or repeated-policy/end-to-end HRL validation.
