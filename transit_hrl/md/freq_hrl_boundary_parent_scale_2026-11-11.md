# Boundary Parent-Scale Repair

Single post-hoc development repair of the [v1 contrast-scale failure](freq_hrl_boundary_tail_probe_2026-11-11.md).
Retain all four roots, caches, labels, A=0/1 and B=2/3, eight whole-prefix folds,
120 columns, total DF 3 and the original qualifying rule. No new simulation or
E8 reuse. Preserve v1 predictions and failures; this is not fresh confirmation.

For each full/fold potential, estimate the 39 parent means/standard deviations
from its TRAINING now/wait arm states, before gating or subtraction. Keep the
existing std>1e-8/otherwise-1 rule for constant parents. Form standardized parent
features, phase/remaining/call-spent, and parent-times-remaining/spent products.
Subtract wait-now and fit direct zero-intercept SVD ridge at DF 3. Do not
restandardize contrasts, fit an offset, clip outputs or select a new DF.
Parent centering changes geometry, not the represented potential-difference span.

Reuse zero/constant/causal-t full and OOF predictions from v1 exactly. Before
fitting, reproduce saved A/B losses on the resident labels and verify rosters.
Refit ONLY nine potentials/root: 36 ridge fits, no constant/baseline fits,
controller loads, optimizer updates or native steps. Four unpinned 1CPU/2GiB
scheduler jobs use node001-node006. Coefficients and raw rows stay server-side.

Report training A/B, prefix-OOF A/B, window halves, prefix contributions and
v1-to-repair changes. Retain negative conditional noise corrections. The same
rule requires OOF B MSE below zero, constant and causal-t overall and in both
window halves on every root. Passing remains a descriptive development result.

## Limits

B and these prefixes have already been inspected. This isolates the declared
geometry change; it is not independent confirmation or a policy gain. Boundary
information is unavailable at the earlier decision. Timing excludes process
imports/scheduler overhead, and lifecycle cost remains incomplete. No further
scaling search, fitting, distillation or native policy release is automatic.
