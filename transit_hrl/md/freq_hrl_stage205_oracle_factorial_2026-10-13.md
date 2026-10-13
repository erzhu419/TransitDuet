# Stage205: separate wait, cost ranking and action-space headroom

Freeze all five Stage204 policies, their cost/wait predictions and47 outer
folds. Reuse the240 states/40 prefixes and eight conditional future draws.
Primary: draw0-3 select, draw4-7 evaluate, then reverse. Sensitivity: even/odd
draws, also both directions. Neither partition changes the model or gate.

LL is the frozen policy; LO replaces only wait feasibility; OL replaces only
cost ranking; OO replaces both; unconstrained removes the wait filter. Each
oracle sees only its selection half. Report selection-half, other-half and
mean8 hindsight outcomes separately, including both roots and directions.
Count repeated wait errors, stable feasible cost-pair inversions and joint
candidate opportunities. Repair contrasts include their interaction: these
are not independent additive explanations.

Run: `native_transit_oracle_factorial_stage205_diagnostic_20261013_r1`.
One unbound scheduler task, node001-node006,1CPU/3072MB. Zero fitting,
optimizer/actor updates or native episodes. Draws and predictions stay on
the server; sync only the compact report.

Results pending. The next model/action change follows the cross-half results,
not the most favorable hindsight frontier or partition.

Limits: halves share their physical prefix and latent demand profile;40
prefix groups remain40, not80 or480. Replacing predictions with sampled
labels diagnoses mechanisms; it is not a deployable policy or independent
CI, and cannot establish impossibility if the sampled frontier fails.
