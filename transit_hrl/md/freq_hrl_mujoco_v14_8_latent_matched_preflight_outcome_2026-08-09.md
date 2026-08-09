# MuJoCo v14.8 Latent Matched-Control Preflight Outcome

Date: 2026-08-09

## Frozen preflight

- Run: `mujoco_v14_8_latent_matched_preflight_20260809_r1`
- Scheduler tasks: `t78886..t78894`
- Frozen algorithm revision: `20d6019f95bb42e7af2ab43ca270d9014826e324`
- Frozen source manifest: `0cab6982d8bd22c9fe4660af5badea18af4da393be8379e992469dc41c1f26c6`
- Environment: `HalfCheetah-v5`
- Optimizer replicates: one fresh development seed (`4166021375`)
- Held-out paths per continuation: 5 disturbance modes x 8 evaluation seeds
- Evidence role: engineering and development preflight only

The anchor and all eight continuations exited naturally on the Linux CPU
cluster. Each continuation used one physical core at approximately 99% CPU
and 405--420 MB resident memory. This preflight is intentionally too small for
optimizer-replicate confidence intervals and was inspected before any 432-cell
screen was launched.

## Calibration

The `s=0.5` projection-only calibration and its zero-strength mean-reward
control selected identical actor tensors and produced identical latent-policy,
executed-action, reward, and return traces on all 40 held-out paths. The
projection reduced reported lower-LF responsibility drift by 55.9--61.1% and
effective upper-HF RMS by 22.1--22.5%, while leaving both latent frequency
diagnostics unchanged. This validates the coordinate transform, not learned
behavior.

## Matched learned comparison

Every learned arm was compared with the same `s=0.5` router and the same
latent-aware checkpoint selector at zero dual learning rate. All learned arms
changed both actor tensors and all 40 latent-policy and executed-action traces.
No arm satisfied the reward and latent-frequency objectives together.

| Shared upper/lower dual LR | Reward result across five modes | Latent lower-LF result | Latent upper-HF RMS result |
|---:|---|---|---|
| 0.03 | lower in all modes by 17.9--71.2% | lower by 2.5--37.5% | lower by 36.1--38.6% |
| 0.05 | mixed: two gains, three losses | worse in four modes | lower by 14.6--22.4% |
| 0.10 | lower in all modes by 26.3--49.7% | worse in all modes | lower by 19.3--23.7% |
| 0.20 | higher in four modes by 4.1--21.7%; standard lower by 1.6% | worse in three modes | lower by 39.3--41.9% |
| 0.30 | lower in all modes by 15.0--34.6% | lower in all modes by 31.8--43.2% | lower by 34.1--39.2% |

These are path means from one optimizer replicate, not confidence-supported
effects. In particular, the `0.20` arm is not a valid selected candidate: its
promising reward pattern coexists with increased latent lower-LF drift in
three of five disturbance modes.

## Root-cause diagnosis

The protocol tied the upper and lower dual learning rates. During this
replicate, the mean upper constraint cost was approximately 0.75, while the
mean lower constraint cost was only 0.005--0.007. Consequently, equal dual
learning rates produced upper final multipliers of 0.74--7.24 but lower final
multipliers of only 0.005--0.050. A single scalar therefore cannot tune the two
frequency responsibilities coherently.

The latent-aware checkpoint score also aggregates endpoint violations into
one scalar. Upper-HF violations dominate that score, so a checkpoint may be
selected despite poor latent lower-LF behavior. A successor must use separate
upper/lower dual schedules and a feasibility-aware checkpoint rule that gives
each registered endpoint an explicit gate before reward ranking.

## Decision and claim boundary

The preregistered 432-cell v14.8 screen is not launched because the preflight
contains no candidate that meets its own joint objective. This avoids turning
a known algorithmic miss into a costly larger negative screen.

Allowed: v14.8 validates matched-comparator plumbing, exact projection-only
calibration, and identifies dual-scale and checkpoint-selection coupling as
development defects.

Forbidden: v14.8 supports a learned no-tradeoff result, reward improvement,
cross-environment generality, or confirmatory evidence.
