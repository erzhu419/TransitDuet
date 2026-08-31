# MuJoCo v18.4 Receding-Horizon Joint Projection Design

## Motivation

V18.3 established a useful but insufficient boundary. Its label-free causal
joint projector made all 120 reused paths feasible and recovered all seven
actor-floor paths, but the selected instantaneous rule changed total actions
far outside the registered trust region: global absolute correction was
1.80755, reference-path RMS reached 0.29345, and actor-floor RMS reached
0.30194. The mechanism proved causal feasibility, not behavioral preservation.

V18.4 tests whether finite-horizon frequency-debt amortization can retain the
feasibility result without the v18.3 distortion. No trust threshold is relaxed.

## Mechanism

At primitive step `t`, the domain-general projector receives the current
proposed upper and lower components and its own executed component prefix. It
forms the exact affine HPF8 and LPF32 residual systems over a finite future
horizon. Remaining endpoint energy is the registered total prefix-plus-horizon
budget minus already incurred causal residual energy.

The optimization is lexicographic:

1. project the upper forecast onto the intersection of the upper-frequency
   ball, the transformed lower-frequency ball, and component bounds while
   preserving the proposed total forecast;
2. if that intersection is not feasible, project upper and lower forecasts
   independently onto their frequency balls and bounds;
3. execute only the first projected pair, update causal state, and replan.

Projection onto each affine quadratic ball is solved by its KKT multiplier.
Intersections use Dykstra iterations. The projector has no observation, reward,
actor-floor label, target correction, future realized action, or episode-end
signal.

## Frozen Candidates

The four candidates are the Cartesian product of:

- planning horizon: 16 or 32 primitive steps;
- causal forecast: hold or damped velocity.

The damped-velocity parameters are fixed at update rate 0.25 and decay 0.75.
All candidates use 64 projection iterations, HPF8 RMS 0.075, LPF32 RMS 0.0475,
unit component bounds, and the existing numerical tolerances. This is a small
mechanism screen, not unrestricted hyperparameter optimization.

## Two-Stage Exact Audit

All four candidates receive a direct endpoint audit on all 120 unchanged v17.8
paths. Selection is lexicographic on validity, direct feasibility, actor-floor
recovery, reference preservation, and correction magnitude. The selected
candidate alone is rerun and audited by the independent full-horizon oracle on
all 120 paths. The other three candidates are recorded as not exact-audited,
not as exact-infeasible. This preserves the exact gate while avoiding 360
oracles that cannot affect selection.

## Advancement Gate

The selected candidate must satisfy all of the following:

- 120/120 valid and directly joint-feasible paths;
- 120/120 independently exact-oracle feasible paths;
- 113/113 reference-feasible paths preserved;
- all seven actor-floor paths and both seed groups recovered;
- nonzero executed-action correction on every actor-floor path;
- total correction absolute maximum no greater than 0.05;
- reference-feasible correction RMS maximum no greater than 0.01;
- actor-floor correction RMS maximum no greater than 0.015.

Prefix budget excursions and numerical nonconvergence are reported even though
the registered frequency contract is the full-path endpoint power. A full pass
authorizes a separately frozen fresh closed-loop experiment only. This reused
screen cannot support reward, generalization, learned-policy, no-tradeoff, or
manuscript claims.
