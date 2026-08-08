# Freq-HRL MuJoCo v11 Canonical-Policy-State Protocol

Date: 2026-08-08

## Evidence Boundary

This protocol is frozen after the v10 one-branch development matrix completed
and before any v11 training result is inspected. All v5-v11 optimizer,
checkpoint-selection, safety-selection, and evaluation paths remain development
data. No v11 path may later be relabeled as confirmatory evidence.

## v10 Outcome And Diagnosis

The v10 one-branch comparison used identical seeds, budgets, policy capacity,
and action scales. Causal responsibility transfer reduced mean responsibility-
level `LowerLFDriftAbs` by 83.3% on HalfCheetah-v5, 51.7% on Hopper-v5, and
83.4% on Walker2d-v5. Hopper and Walker2d met the registered return
noninferiority floor, but HalfCheetah did not: its mean return changed from
399.83 to 327.57. The registered one-branch decision is therefore `failed`,
and v10 cannot advance to fresh confirmatory seeds.

The action reconstruction identity itself held: pre-clipping reconstruction
RMS stayed below `1.7e-8`. The avoidable difference was in the learned policy
state. v10 exposed responsibility-specific upper anchors and running lower
responsibility state directly to the actor. Consequently, two algebraically
equivalent action decompositions induced different actor observations,
training trajectories, and policies. The v10 result rejects the implication
that nominal-action reconstruction alone guarantees retrained return
invariance.

## Structural Repair

v11 separates two causal state contracts.

1. The **canonical policy state** contains the environment observation,
   frequency-routed exogenous bands, the raw upper policy action, the causal raw
   lower LF estimate, and the previous raw lower policy command. It is identical
   for additive and transfer decompositions whenever their raw policy outputs
   and environment trajectory are identical.
2. The **responsibility constraint state** contains the responsibility-level
   upper anchor and the running LF state of the executed lower responsibility.
   Only the lower cost critic receives this state. The lower actor and reward
   critic never receive decomposition-specific state.

The shared actor-critic core must therefore support an optional lower cost-
critic state that is distinct from the lower actor/reward-critic state. This is
a centralized causal critic, not future information: every field is available
at the current transition. The separate cost state must not alter the actor or
reward-critic parameter count.

For an unconstrained branch with matched initialization and random seeds, the
canonical-state contract implies pathwise equality of raw actions and
environment transitions between additive and transfer modes. Responsibility
transfer may change only the upper/lower attribution, responsibility-level
leakage metrics, and diagnostics. Constrained branches may differ because their
registered cost is defined on the responsibility-level lower contribution.

## Fixed Development Matrix

No action-scale, period, filter, optimizer, or threshold sweep is permitted.
The fixed configuration remains:

- upper action scale: `1.0`;
- lower action scale: `1.0`;
- upper period: `16`;
- causal EMA alpha: `0.04`;
- hidden dimension: `64`;
- 64 training iterations with 512 primitive transitions per root and iteration;
- the existing three environments and three development optimizer replicates;
- the existing four training and five held-out disturbance modes.

Two arms use the same source revision and source manifest:

1. `additive/canonical_policy_state`;
2. `causal_lf_transfer/canonical_policy_state`.

Each arm trains `freq_hrl_no_leakage` and `freq_hrl_safe_selector`. The full
matrix is allowed only after unit tests and one synchronized preflight pair
verify source identity, serialized checkpoint integrity, and state-contract
metadata.

## Development Gates

The implementation gate requires:

1. future suffix changes cannot alter any prefix policy or transfer output;
2. additive and transfer canonical actor states are exactly equal for matched
   raw actions and observations;
3. responsibility cost states are causal and may differ only in their declared
   responsibility fields;
4. the no-leakage preflight pair has equal frozen parameter SHA-256, equal raw
   action metrics within `1e-8`, and equal return within `1e-8`;
5. pre-clipping reconstruction RMS is at most `1e-7` on every transfer path;
6. all canonical and responsibility states reset at every episode boundary;
7. source identity and serialized checkpoint hashes verify independently.

For the full one-branch matrix, every environment must satisfy:

1. paired mean return absolute difference at most `1e-7`;
2. paired frozen parameter hashes match for every optimizer replicate;
3. responsibility-level `LowerLFDriftAbs` is at least 10% below additive;
4. raw-lower drift and raw-action RMS match within `1e-7`;
5. all optimizer replicates and held-out disturbance rows are present.

The complete safe-method gate retains the v10 practical criteria: for every
environment, transfer return must be no worse than 2% below additive,
responsibility-level `LowerLFDriftAbs` must fall by at least 10%, and
reconstruction RMS must be at most `1e-7`. Branch selection, confidence bounds,
compute multipliers, and fallback rates must be reported.

If exact one-branch invariance fails, v11 is rejected as an implementation or
state-contract failure. If exact invariance passes but drift reduction fails,
the causal coordinator is rejected. Fresh confirmatory seeds remain forbidden
until both the one-branch and complete safe-method gates pass.

