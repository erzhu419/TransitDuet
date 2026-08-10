# MuJoCo v14.15 Restoration Multiseed Development Protocol v2

Date: 2026-08-10

## Fixed candidate

This screen follows the frozen single-seed v14.15 preflight. The candidate is
fixed before any multiseed outcome is accessed:

- candidate: `group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3`;
- selection source:
  `mujoco_v14_15_closed_loop_restoration_filter_preflight`;
- source decision SHA-256:
  `540036f775d4fce2c10f0d9aa854da4209cf4e5c8801ff87430713493fb16fdf`;
- reward tolerance: `0.005`;
- closed-loop backtracks: `8`;
- restoration funnel multiplier: `3`.

No multiseed arm may replace this candidate. Other v14.15 arms are retained
only as frozen controls and ablations.

The executable `freq_hrl` package is frozen at Git revision
`825871ebf75f55de1bbf5ae2f9c7c5eb0fa97e7a` and source-manifest SHA-256
`4ee9217bc9ad52116239157dde0d284a900a930cdd3ca29ca7eb62002302f550`.
This revision corrects the dimensionally inconsistent restoration-merit
validator documented in
`freq_hrl_mujoco_v14_15_multiseed_r1_invalidation_2026-08-10.md`. The earlier
single-seed preflight remains verified against its original revision and
manifest.

The predecessor run
`mujoco_v14_15_restoration_multiseed_development_20260810_r1` is invalid and
cannot be reused. Protocol v2 requires a new run directory and recomputes all
45 anchors and 405 continuations under one source identity.

## Scope

- Environments: `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`.
- Optimizer seeds: the 15 frozen v14.15 seeds not used by the preflight.
- Excluded selection seed: `1361331598`.
- Continuations per environment and seed: all nine frozen v14.15 arms.
- Shared anchor per environment and seed: one.
- Total cells: `3 x 15 x (1 + 9) = 450`.
- Evaluation paths per continuation: five disturbance modes by eight fixed
  evaluation seeds.

The cluster launch is staged. All 45 anchors must complete and synchronize
before the 405 continuations are dispatched. Tasks use scheduleurm with
`require_node=None` and the allowed set `node001..node006`.

## Statistical unit

The optimizer seed is the independent replicate. The 40 held-out paths inside
one continuation are repeated measurements and are never treated as 40
independent samples. The same optimizer-seed resample is applied jointly to all
three environments in every bootstrap draw.

## Primary effects

The fixed candidate is paired with
`paired_s050_d000_control` within environment, optimizer seed, disturbance
mode, training roots, selection roots, and evaluation roots.

For each environment and optimizer seed, effects are averaged over the five
disturbance modes:

1. normalized return difference:
   `(candidate - comparator) / max(abs(comparator), 1)`;
2. five frequency effects:
   `log(comparator endpoint / candidate endpoint)`.

The primary family has 18 contrasts: one return and five frequency contrasts
for each of three environments. A 20,000-draw paired basic bootstrap produces
one simultaneous 95% lower bound using the maximum downward bootstrap error.
The registered thresholds are:

- return lower bound greater than `-0.02`;
- every frequency lower bound greater than `-log(0.95)`, corresponding to at
  least 5% geometric-mean reduction.

No contrast or threshold may be removed after outcome access.

## Robustness and mechanism gates

- Every environment must have at least 12 of 15 complete candidate cells.
- Across all 45 environment-seed cells, the one-sided 95% Wilson lower bound
  for the complete-gate fraction must be at least `0.70`.
- Every calibration cell must pass pathwise identity.
- Every environment-by-disturbance point estimate must meet its registered
  return or frequency threshold.
- The analyzer reports nominal per-mode intervals, but they are secondary and
  are not substituted for the simultaneous primary family.
- Strict-filter versus restoration effective updates and feasibility rates are
  frozen ablation diagnostics, not candidate-selection criteria.

## Decision

`candidate_ready_for_fresh_confirmation` requires every primary, robustness,
calibration, and complete-gate criterion above. Any failure yields
`candidate_not_ready_for_confirmation`; no post hoc arm substitution is
allowed.

This is a candidate-fixed multiseed development screen, not confirmation. A
positive outcome may authorize a new protocol with a fresh optimizer-seed
namespace. It cannot itself support a confirmatory, universal, or
submission-ready claim.
