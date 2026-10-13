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
All47 fits`t147795-t147841` and report`t147845` completed with exit0;
1197 solves completed. Only the42783-byte report was pulled, no raw/models.
Cost passes; wait fails. No fold has a grid option satisfying the full inner
root/regime wait condition (stricter than the outer split-level aggregate gate).
Outer regime wait worsens from Stage203's +0.000055729 to +0.001153125 min;
scene/root397 is +0.001333333. Cross-lower-root wait remains negative for both
roots (-0.003069792/-0.001412500), but this does not close the complete gate.
The selector changes penalties without resolving the tradeoff; no promotion.

Next: stop penalty-grid changes. Use archived labels to separate frozen cost
ranking errors, learned wait-feasibility errors and the joint feasible action
frontier before changing the model or action representation. Label-informed
selectors are diagnostics only; independent future-draw halves are needed to
distinguish apparent oracle headroom from selection on noisy sample means.
Limits: inner empirical feasibility does not guarantee outer safety or CI;
this is still supervised single-intervention development, not full learned HRL.
