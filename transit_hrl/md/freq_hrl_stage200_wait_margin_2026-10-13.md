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

All47 fits and the report completed with exit0. The zero-margin control
reproduces Stage199. Only the24840-byte report was pulled; no raw/model pull.

| Policy | Cost gate | Wait gate | Overrides: scene / regime / lower-root |
| --- | --- | --- | --- |
| uncalibrated | pass | fail | 133 / 120 / 111 |
| mean_bias | pass | fail | 103 / 78 / 45 |
| cluster_margin90 (primary) | fail | fail | 37 / 25 / 14 |

Mean-bias correction leaves root397 wait positive in all three splits:
scene +0.000891667, regime +0.000069792, lower-root +0.001604167 min.
The primary still worsens scene/root397 wait (+0.001464583 min) and
regime/root401 wait (+0.001659375 min); root401 cost becomes positive
in regime (+0.000290008) and lower-root (+0.000900888). No policy passes.

Next: stop global-margin tuning. Decompose the frozen selections into
retained, rerouted and identity actions, and inspect wait residuals by action
and causal queue/headway context before choosing a conditional wait critic.
The aggregate results show the global correction fails, not its exact cause.

Limits: this is an empirical mean-wait correction, not formal OOD coverage,
CI-supported improvement or repeated-policy/end-to-end HRL validation.
