# MuJoCo v14.15 multiseed r2 outcome

## Decision

`mujoco_v14_15_restoration_multiseed_development_20260810_r2` is a valid,
complete, candidate-fixed development screen. Its frozen decision is
`candidate_not_ready_for_confirmation`. It is not confirmatory evidence and
does not support a cross-environment no-tradeoff or submission-readiness claim.

The repaired run completed 45 anchors and 405 continuations (`450/450`) with
no failed or cancelled r2 cells. Anchors were scheduler tasks `t81013` through
`t81057`; continuations were `t81059` through `t81463`. Every task was allowed
to run dynamically on `node001` through `node006`; no task required a specific
node. The equivalent of the deterministic r1 failure cell was r2 task `t81323`
and completed normally.

## Frozen identity

- Development protocol: `mujoco_v14_15_restoration_multiseed_development_screen_v2`
- Analysis profile: `fixed_v14_15_restoration_candidate_multiseed_v2`
- Candidate: `group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3`
- Execution revision: `825871ebf75f55de1bbf5ae2f9c7c5eb0fa97e7a`
- Source manifest: `4ee9217bc9ad52116239157dde0d284a900a930cdd3ca29ca7eb62002302f550`
- Statistical unit: optimizer seed, 15 seeds; held-out paths are repeated
  measurements and are not replicates.
- Primary family: 18 registered environment-by-endpoint contrasts, 20,000
  paired bootstrap draws, simultaneous one-sided 95% lower bounds.

## Registered result

Only 8 of 45 environment-by-optimizer-seed candidate cells passed the complete
gate. The one-sided Wilson lower bound for the aggregate complete fraction was
`0.103187`, below the registered `0.70` threshold.

| Environment | Complete cells | Return effect | Lower LF log effect | Upper HF log effect |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | 0/15 | -0.020402 | -0.002217 | -0.000913 |
| Hopper-v5 | 7/15 | +0.021056 | +0.147073 | +0.198284 |
| Walker2d-v5 | 1/15 | -0.000272 | +0.027336 | +0.026171 |

Return effects are paired normalized differences. Frequency effects are paired
`log(baseline/candidate)` values; positive values favor the candidate. These
point estimates do not override the failed simultaneous family or complete-cell
gates.

## Failure diagnosis

The failure is not simply a lack of optimizer updates. The restoration filter
frequently reduced the independent closed-loop guard from 20 frequency
violations to zero at the training-final model. However, the checkpoint
selector evaluates disjoint crossed-condition paths. HalfCheetah selected the
initial anchor in 13 of 15 replicates and no HalfCheetah replicate passed the
complete gate. Walker2d exhibited the same issue less uniformly.

The current differentiable constraint sees four frozen replay paths, one
training root per disturbance mode, while the derivative-free transaction guard
uses only two crossed roots and averages roots within each mode. This creates
three weaknesses:

1. the projection direction is informed by a narrow state bank;
2. the worst-group objective differentiates through only one active group;
3. restoration can fit averaged guard paths without transferring to disjoint
   checkpoint-selection paths in chaotic environments.

The next algorithm revision must therefore use broader crossed frozen-state
replay, an all-active-violation restoration objective, pathwise guard
constraints, and restoration-specific actor regularization. Merely relaxing the
reward tolerance, changing the gate after seeing r2, or adding more seeds to the
same candidate is not justified.

## Artifact boundary

Authoritative compact artifacts are stored under
`transit_hrl/results/authoritative_evidence_sources_20260810/mujoco_v14_15_multiseed`.
Their SHA-256 values are:

- `decision.json`: `c373c52470b980ace11f0f5dcb459d9f0ffc2e8dab109d00d77ae0ef01587561`
- `primary_contrasts.csv`: `01f6147925f7fc2f6c4d5464eff4fdbb71c3a01cb1dd5429eab757aa5f8fcde3`
- `replicate_rows.csv`: `56f38cafd9641e6ee3a0ce37411a0a209306bc215218309538e49f7879d4c9b7`
- `report.md`: `8d73d55114b14898b8bbb2997369173230dcf7d2e3b5a8540f57996984888a12`

The full checkpoints and cell directories remain untracked experiment data.
The invalid r1 run remains excluded from merge, analysis, registry, and paper
claims as documented separately.
