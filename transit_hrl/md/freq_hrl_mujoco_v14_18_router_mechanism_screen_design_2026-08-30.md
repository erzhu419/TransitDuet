# Freq-HRL MuJoCo v14.18 Router Mechanism Screen

## Decision context

The v14.17 native primal-dual/CVaR continuation failed its pre-registered
engineering gate in the first complete cell. Native dual variables learned, but
every trained actor update was rejected by the closed-loop restoration guard.
The full 72-cell v14.17 matrix was therefore not dispatched.

A post-failure radial probe reconstructed the selected policy exactly. Scaling
either actor output head failed to produce a reward-safe frequency improvement.
A router-only probe then found that changing the causal joint-band routing
strength preserved the executed action and reward while reducing the closed-loop
frequency violation merit in one HalfCheetah cell. That cell
(`HalfCheetah-v5`, optimizer seed `4196455150`) was inspected before this screen
was frozen and is explicitly marked as the discovery cell.

## Frozen screen

- Role: adaptive mechanism development, not confirmatory paper evidence.
- Anchors: the nine completed v14.17 anchors from
  `mujoco_v14_18_router_mechanism_anchors_20260830_r1`.
- Environments: `HalfCheetah-v5`, `Hopper-v5`, and `Walker2d-v5`.
- Optimizer seeds: `4196455150`, `3082324697`, and `1915709332`.
- Replication unit: one environment by optimizer-seed anchor.
- Router strengths: `0.5`, `0.6`, `0.7`, `0.8`, `0.9`, and `1.0`.
- Actor gains: fixed at `1.0`; actor contraction is not part of this screen.
- Guard profile: frozen v14.17 crossed 16-path, four-mode, five-endpoint
  mode-CVaR guard with the v14.17 reward floor.
- Execution: nine independent one-core tasks through `scheduleurm`, dynamically
  placed across `node001-node006`; no task is pinned and Slurm is not used.

## Nomination rule

The comparator is router strength `0.5`. A single global router strength can be
nominated only when all nine cells have:

1. zero closed-loop reward violations; and
2. strictly lower frequency-violation merit than strength `0.5`.

Per-environment or per-seed strength selection is prohibited. If more than one
global strength qualifies, selection maximizes the minimum relative merit
reduction, then median reduction, then mean reduction, then minimizes total
frequency violations, and finally selects the smaller strength. If no strength
passes all nine cells, the result is a mechanism boundary rather than a selected
v14.18 core change.

## Claim boundary

Passing this screen would justify implementing and independently validating a
function-preserving router restoration step. It would not establish a
cross-domain performance claim, a confidence interval, or a final manuscript
result. Any implemented mechanism still requires fresh continuation seeds and a
pre-registered confirmatory protocol.
