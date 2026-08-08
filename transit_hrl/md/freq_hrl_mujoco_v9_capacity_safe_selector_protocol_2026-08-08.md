# Freq-HRL MuJoCo v9 Role-Capacity And Safe-Selector Protocol

Date: 2026-08-08

## Evidence Boundary

This is a development protocol committed before the v9 capacity outcomes are
inspected. All v5-v9 optimizer, checkpoint-selection, safety-selection, and
evaluation paths are development data. They cannot later be relabeled as
confirmatory evidence.

## Rejected Designs

- v5 showed that one fixed leakage budget was not Pareto-consistent across
  HalfCheetah, Hopper, and Walker2d.
- v6 projected a manual cost correction after Adam. It invalidated Adam's
  moment state and failed HalfCheetah.
- v7 kept constrained candidates inside Adam and enforced a local sampled
  surrogate guard. It improved v6 but still produced an episode-return
  tradeoff in Hopper and failed both objectives in HalfCheetah.

These results reject additional post-hoc tuning of one leakage coefficient as
the primary repair.

## Structural Diagnosis

The previous MuJoCo protocol gave the slow upper controller an action scale of
`0.35` while the fast lower controller retained scale `1.0`. A controller that
needs a persistent locomotion action could therefore be forced to place that
action in the lower stream and then be penalized for its low-frequency
content. This is a responsibility-capacity mismatch, not only an optimizer
problem.

Protocol v9 restores a symmetric default action capacity (`upper=1.0`,
`lower=1.0`) and records:

- `UpperActionRMS`;
- `LowerActionRMS`;
- `UpperActionEnergyShare`;
- `AdditiveActionClipRate`;
- the declared upper/lower capacity ratio.

## Trajectory-Safe Selector

`freq_hrl_safe_selector` trains three branches under the same initialization,
training roots, and common-random-number schedule:

1. no leakage constraint;
2. reward-guarded Adam projection;
3. scalarized primal-dual constraint.

Checkpoint selection uses only checkpoint-selection seeds. Branch selection
then uses a separate safety-selection set. Final evaluation seeds are loaded
only after the branch has been frozen.

The branch-selection inference unit is an independent safety-selection seed.
Disturbance modes are first averaged within seed, then paired differences are
bootstrapped across seeds with 4,096 draws and a one-sided 90% bound. A
constrained branch is eligible only when both conditions hold:

- return noninferiority lower bound is at least minus 2% of the no-leakage
  baseline return magnitude;
- leakage-drift difference upper bound is at most minus 10% of baseline drift.

If neither constrained branch is eligible, the selector restores the exact
no-leakage branch. This is a selection-set guarantee, not an episode-level or
population theorem. The method uses three times the branch-training compute;
that multiplier is recorded and must be disclosed in comparisons.

## Pre-Registered Capacity Matrix

Frozen source revision: `b3cc9c90d615404a75d39af53ad1f216b50c4706`.

The development matrix scans one global upper scale and keeps lower scale at
`1.0`:

| run suffix | upper scale | lower scale |
|---|---:|---:|
| `u035` | 0.35 | 1.00 |
| `u060` | 0.60 | 1.00 |
| `u080` | 0.80 | 1.00 |
| `u100` | 1.00 | 1.00 |

Every scale uses all three registered environments, three independent
optimizer replicates, four training disturbance modes, all registered
evaluation disturbances, and both `freq_hrl_safe_selector` and
`freq_hrl_no_leakage`. The external no-leakage cell is required even though
the selector trains an internal baseline: it provides a held-out,
scale-matched control and a checkpoint-hash consistency check.

## Development Gate

A single global scale may advance only if all cells and artifacts pass the
source-bound audit and, for every environment:

1. selected-policy mean return is no worse than 2% below the scale-matched
   no-leakage mean;
2. selected-policy mean `LowerLFDriftAbs` is at least 10% below no leakage;
3. at least two of three optimizer replicates choose a constrained branch;
4. action clipping remains reported and is not hidden by the return metric.

If multiple scales pass, choose the scale with the best worst-environment
relative drift reduction; break a tie by pooled equal-environment return. If
no scale passes, v9 is rejected for a domain-general no-tradeoff claim and the
next repair must change the causal action decomposition rather than tune this
matrix after inspection.

Fresh confirmatory seeds remain forbidden until this development gate passes.
