# Stage204: inner cost/wait decision selection

Keep all four Stage203 controls and frozen outer cost predictions. Each inner
regime receives its own train-only cost model and four wait penalties, using
the same candidate pairs, weights and inner-only scales. Rank penalties by
positive worst-root/regime actual wait, worst-root equal-regime actual cost,
equal-root cost, worst pair MSE, then stronger penalty. If all are wait-harmful,
select the least violation and record infeasibility; no gate tolerance changes.
Refit the selected wait model on outer training. Outer cost ranking, zero
margin, identity and the47 outer folds/240 mean8 states remain unchanged.

Run: `native_transit_decision_wait_stage204_development_20261013_r1`.
47 fits plus one report;230 inner cost,920 inner wait and47 final wait solves.
No new native episodes or actor/optimizer updates. Unbound node001-node006,
1CPU/3072MB each; raw data, models and inner decision ledgers stay server-only.
Tests:20 passed, covering cost batch units, selected-action credit, inner/outer
label isolation, frozen controls and exact1197-fit/47-fold accounting.
Next: compare the decision-selected policy with MSE-selected and fixed controls.
Limits: inner empirical feasibility does not guarantee outer safety or CI;
this is still supervised single-intervention development, not full learned HRL.
