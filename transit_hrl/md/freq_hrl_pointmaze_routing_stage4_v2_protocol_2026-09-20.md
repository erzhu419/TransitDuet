# PointMaze Frequency-Routing Stage-4 V2 Protocol

Date: 2026-09-20

## Motivation

Stage-4 V1 matched total model parameters but did not match capacity within
each hierarchy level. Its routed arm used upper/lower state dimensions 38/126
and per-level parameter counts 25,479/42,023; swapped routing reversed both
allocations. V1 therefore changed frequency content and layer-specific
capacity together.

V2 is a fresh attribution repair. It does not reinterpret V1 and does not reuse
its optimizer, training, selection, or evaluation seeds.

Frozen algorithm revision:
`f9ab0b4a532d1bc0c31466b24d4dcbc70634e585`.

## Equal-Shape Routing

Every method gives both hierarchy levels the same 134-dimensional state:
current physical state, the appropriate goal error, and 128 fixed Haar/history
slots. Routed and swapped arms zero excluded coefficient blocks instead of
removing them.

| Method | Upper history slots | Lower history slots |
|---|---|---|
| `hrl_history` | complete raw causal history | complete raw causal history |
| `hrl_causal_filter` | complete causal-filter history | complete causal-filter history |
| `hrl_multiscale_all` | slow + mid + high | slow + mid + high |
| `hrl_multiscale_routed` | slow + mid + zero(high) | zero(slow) + mid + high |
| `hrl_multiscale_swapped` | zero(slow) + mid + high | slow + mid + zero(high) |

For a fixed optimizer root, all five arms have identical network shapes,
68,424 trainable parameters, and identical initial parameter values. Both
levels retain complete current actor-visible physical feedback. Only the
history transformation or registered coefficient mask changes.

## Frozen Matrix

Scenarios remain:

1. `clean`
2. `fast_observation_noise`
3. `slow_drift_fast_action`

The development matrix contains 5 methods x 3 scenarios x 8 independent
optimizer roots = 120 cells. Each cell uses:

- 768 PPO iterations and a 300-step fixed horizon;
- four training seeds;
- 16 disjoint checkpoint-selection seeds;
- 16 disjoint held-out evaluation seeds;
- checkpoint evaluation every 96 iterations;
- the same physical periods, stress amplitudes, reward, option credit, and
  parameter budget as V1.

All V2 seeds are fresh. Optimizer roots are `124007`, `124013`, `124031`,
`124043`, `124067`, `124089`, `124113`, and `124127`; role seeds occupy a
separate `1_200_000` namespace.

## Claim Gate

The V1 gate is unchanged. Selective routing is supported only if:

1. routed success exceeds all-band success in both primary stress scenarios;
2. routed success exceeds swapped success in both primary stress scenarios;
3. routed success exceeds causal-filter success under observation noise; and
4. clean routed success is noninferior to all-band success at margin 0.10.

Effects use root-paired two-sided 95% confidence intervals. The four conditions
form one conjunctive gate. Return and final distance are supportive endpoints.

## Execution Boundary

The preflight is 15 cells: one fresh optimizer root, two PPO iterations, one
seed per role, and a 64-step horizon. It may validate only:

- exact mask content and equal state/model shapes;
- identical initialization for matched roots;
- finite learned-policy updates;
- seed, timing, option-boundary, runtime, and stress-channel contracts;
- dynamic scheduler placement and compact JSON synchronization.

Preflight outcomes are not performance evidence. After it passes, the 120-cell
matrix runs unchanged on node001-node006 with one physical core per cell and no
hard node binding.

## Claim Boundary

Allowed after a positive development gate: within the registered endogenous
PointMaze history setting, selective fixed-shape routing outperforms all-band,
swapped, and causal-filter controls under the specified stress families while
remaining clean-noninferior.

Forbidden regardless of outcome: V2 validates exogenous-state routing,
promotion, leakage constraints, Transit or trading performance, universal
frequency assignment, or domain-general Freq-HRL. Those require explicit
actor-visible exogenous streams and separate frozen experiments.
