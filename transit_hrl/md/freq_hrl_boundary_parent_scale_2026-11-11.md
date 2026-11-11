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

## Result

Code `4063651197`; 15 focused tests passed. Tasks t214585-t214588 completed
on node004/node005: 36 ridge fits, zero baseline fits/native steps. All prior
losses reproduced. Measured phases used 1.074 CPU seconds; retrieved 96,896
public bytes, no raw samples or coefficients. See [losses/costs](../results/plan_validity_boundary_parent_scale_20261111/summary.json)
and [execution receipt](../results/plan_validity_boundary_parent_scale_20261111/release.json).

Prefix-OOF tail MSE change versus zero; negative improves:

| Task/root | Suffix A | Suffix B | B-improving prefixes | Original B rule |
| --- | ---: | ---: | ---: | --- |
| PointMaze/1972110000 | -13.96% | -16.80% | 7/8 | Pass |
| PointMaze/1972120000 | +1.19% | -2.07% | 5/8 | Pass |
| Reacher/1972210000 | +1.41% | -0.59% | 4/8 | Pass |
| Reacher/1972220000 | +8.12% | +1.20% | 1/8 | Fail |

PointMaze B MSE fell from v1's 160.668/5631.628 to 0.001621/0.002174 without
clipping. The reachable scale amplification is repaired. Three roots pass
the frozen B/half-window rule, but the all-root rule fails; two also disagree
across A/B. Reacher root1's B gain shrank from 4.96% to 0.59%, so this is not
a uniform accuracy gain. Root2 already loses 6.96% on training-prefix B.
Keep both versions; do not assemble root-specific winners or distill a policy.
Next examine fixed-prediction suffix uncertainty and prefix influence before
choosing new data, rather than searching another scale, DF or loss on this bank.
