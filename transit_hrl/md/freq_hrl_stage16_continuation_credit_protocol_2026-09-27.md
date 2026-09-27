# Stage-16 Continuation-Credit Development Protocol

Implementation revision: `0bab1e9e48`.

Frozen roots: 209011/209061; preflight: 208001. Preserve Stage-15's failure.
Reuse the Stage-12 controller, trigger, seed roles, and Stage-15 sampled checks:
eight branch-fit paths, 12 pairs/path, 16 evaluation paths, 1,200 steps.
Now and wait-one-check arms use the frozen reference after the intervention.

Split each action contrast into observed 50-step ISE and predicted suffix ISE.
Collect suffix states from the window boundary every 50 steps; also collect
reference-path states. All training targets follow the reference policy.
Use the 37 causal features plus bin phase, remaining horizon, and spent-budget
indicator, sampled before the action. Fit a continuation ValueNet (width 64),
64 full-batch Adam updates at 0.001. Each path receives equal training weight.
Estimate a remaining-time linear baseline from training paths and learn a
residual multiplied by remaining time, so terminal value is exactly zero.

Leave one entire path out, including all its branches, when predicting its
endpoint values. Fold RNG key: (root, held-out path, 16016). No evaluation
labels or diagnostic Stage-13/14 outcomes enter training. Use predicted
V(wait endpoint)-V(now endpoint) plus observed short-window advantage.
Compare continuation-contrast MSE against a zero-tail-contrast baseline.

Deploy two triggers: short-only control and bootstrapped candidate, with the
same 41 timing features, fixed Ridge alpha 100 and zero threshold. No second
CV on cross-fitted labels (which would couple outer held-out paths to label
construction), threshold search, new policy iterations, or new roots.
One call/bin and offset-25 deadline remain unchanged.

Additional fitting: 9,600 reference + 230,400 paired steps/root; evaluation:
19,200 each for reference, short-only and bootstrap. Report eight critic fits
and 512 optimizer steps separately. Inherited supervision/controller training
is additional. Preflight: two fit paths, two pairs/path, 300-step horizon.

Advance only if BOTH roots have lower out-of-path continuation-contrast MSE
than zero-tail and bootstrap mean evaluation ISE below short-only, fixed,
Stage-12 and Stage-9 (1.143830/0.838439). Otherwise retain the failed screen.
These reused development paths cannot supply independent confirmation.
Run via scheduleurm, dynamic node001-node006, one CPU/1536 MB per cell;
sync compact result JSON only, with no raw trajectories or checkpoints.

## Execution

Preflight `t101192` completed on node006: four pairs, two path-disjoint folds
(128 critic updates), exact reference replay and 4,800 additional primitive
steps. Retrieved JSON: 50,977 bytes. Operational checks passed; the tiny fit
failed continuation qualification (MSE 0.003482 vs zero-tail 0.000000705).
Bootstrap made all ten noninitial calls at the deadline. This is not a
performance result; the frozen two-root development screen remains unchanged.
