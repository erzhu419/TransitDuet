# PointMaze Stage-5 Optimization-Stability Screen Protocol

Date: 2026-09-21

## Purpose

Stage-5 V1 supported HRL learning relative to its paired untrained policy but
missed the frozen absolute-success CI gate. One optimizer root failed and one
was borderline. The failed root's dense selection return continued to improve
after a noisy success-rate peak had fixed the selected checkpoint.

This screen tests two bounded repairs: align checkpoint selection with the
dense training objective, and reduce rollout-gradient variance by increasing
the number of independent training paths. It does not change the task,
hierarchy, reward, network capacity, PPO update rule, horizon, or frequency
contract.

## Frozen Design

Algorithm revision: `a4a64730a9542e690650d3a39856a5324ec51fcc`.

Two diagnostic optimizer initializations are reused: `134113` and `134127`.
All train, selection, and evaluation environment seeds are new and are paired
across arms. Reusing the known difficult initializations makes this a post-hoc
development screen, never independent performance evidence.

| Arm | Checkpoint rank | Training rollout roots |
|---|---|---:|
| `v1_control` | success, then dense return | 4 |
| `dense_rank` | dense return, then success | 4 |
| `more_rollouts` | success, then dense return | 8 |
| `dense_rank_more_rollouts` | dense return, then success | 8 |

Each formal cell uses 768 PPO iterations, 16 selection paths, 16 held-out
evaluation paths, a 300-step horizon, and the unchanged Stage-5 task settings.
The matrix has eight single-core cells. Only compact `result.json` artifacts
are synchronized.

## Selection Rule

A non-control arm is eligible only if:

- its tracking-success gain over the paired control is at least 0.05 on each
  difficult optimizer root;
- its mean held-out return is no lower than control;
- final-minus-untrained tracking success is positive on each root; and
- final-minus-untrained return is positive on each root.

Eligible arms are ordered by worst-root success, then mean success, then mean
return. If no arm is eligible, no Stage-5 V2 confirmation is authorized.

## Claim Boundary

This screen can select an optimization recipe for a fresh-seed V2 experiment.
It cannot repair the V1 result, pass the Stage-5 gate, establish hierarchy
advantage, authorize frequency routing, or support a manuscript performance
claim.

The four-cell software preflight completed and passed. See
`freq_hrl_pointmaze_exogenous_stage5_stability_preflight_2026-09-21.md`.

The formal screen selected `more_rollouts`: both eight-rollout arms improved
both difficult roots and were identical, while changing checkpoint rank alone
had no effect. A fresh-seed V2 is authorized with the original rank mode and
eight training rollout roots for both flat and HRL. See
`freq_hrl_pointmaze_exogenous_stage5_stability_result_2026-09-22.md`.
