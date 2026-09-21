# PointMaze Exogenous Frequency-Routing Stage-6 Protocol

Date: 2026-09-22  
Evidence stage: development  
Runtime protocol: `pointmaze_exogenous_frequency_routing_stage6_v1`  
Frozen algorithm revision: `4ed9e8e131235f0f99844e8ea8bfeb737a68276f`

## Purpose

Stage 5 established that an ordinary learned goal-conditioned hierarchy can
learn the separate, causal exogenous-control substrate. Stage 6 asks the next
scientific question: does a causal multiscale representation specifically help
the hierarchy, and does assigning slow information to the upper level and fast
information to the lower level outperform information-equivalent controls?

This stage does not test the retired action projector, promotion, leakage loss,
or responsibility gauge. All remain disabled.

## Frozen Methods

All methods receive the current unfiltered physical state and transformations
of the same 32 causal samples from the separate exogenous stream. Every state
has 134 dimensions. Methods within the same architecture have identical model
capacity and identical initial parameters for a given optimizer root.

1. `flat_exogenous_history`: flat PPO with raw external history.
2. `flat_exogenous_filtered`: flat PPO with causal low-pass history.
3. `flat_exogenous_multiscale_all`: flat PPO with all Haar bands.
4. `hrl_exogenous_history`: goal-conditioned HRL with raw history at both levels.
5. `hrl_exogenous_filtered`: goal-conditioned HRL with filtered history at both levels.
6. `hrl_exogenous_multiscale_all`: goal-conditioned HRL with all bands at both levels.
7. `hrl_exogenous_multiscale_routed`: slow+mid to upper and mid+high to lower; omitted coefficients are zero-masked.
8. `hrl_exogenous_multiscale_swapped`: mid+high to upper and slow+mid to lower.

The all-band and swapped arms are required attribution controls. A result that
only beats raw history is insufficient evidence for selective frequency routing.

## Training And Evaluation

- Optimizer roots: `184007, 184013, 184031, 184043, 184067, 184089, 184113, 184127`.
- Per root and method: 8 training, 16 checkpoint-selection, and 16 held-out evaluation seeds.
- The seed roles are paired across all eight methods and disjoint from Stage 5.
- Training: 768 iterations; horizon: 300 primitive steps.
- Checkpoint selection: every 96 iterations, ranked by held-out selection success and then dense return.
- Primary endpoint: held-out `tracking_success_rate`.
- Statistical unit: optimizer root. Episodes are averaged within root before 95% Student-t confidence intervals are formed.
- Scheduler: one CPU and 2560 MB per cell, dynamically placed across `node001`-`node006`.
- Synced artifact: compact `result.json` only; checkpoints are disabled.

## Preregistered Contrasts

The analysis reports absolute performance, trained-minus-untrained learning,
and the following paired contrasts:

- flat filtered versus flat history;
- flat all-band versus flat history;
- HRL filtered versus HRL history;
- HRL all-band versus HRL history;
- routed HRL versus HRL history;
- routed HRL versus HRL filtered;
- routed HRL versus HRL all-band;
- routed HRL versus swapped HRL;
- hierarchy-by-multiscale interaction:
  `(HRL all - HRL history) - (flat all - flat history)`.

Positive improvements always mean better performance; RMSE and final distance
are sign-reversed before contrasts are classified.

## Strict Claim Gate

The central Freq-HRL routing claim is supported only if all conditions hold:

1. routed HRL success CI lower bound is at least 0.50;
2. routed HRL improves success and return over its paired untrained policy;
3. routed HRL improves success over HRL history;
4. routed HRL improves success over HRL causal filtering;
5. routed HRL improves success over HRL all-band input;
6. routed HRL improves success over swapped routing;
7. the hierarchy-by-multiscale success interaction has a strictly positive CI.

Each component remains separately reportable. Failure of the conjunction is
not repaired by changing seeds, endpoints, margins, or the gate after results.
