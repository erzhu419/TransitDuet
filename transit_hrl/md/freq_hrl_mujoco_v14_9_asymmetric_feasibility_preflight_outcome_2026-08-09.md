# MuJoCo v14.9 Asymmetric Feasibility Preflight Outcome

Date: 2026-08-09

## Frozen preflight

- Run: `mujoco_v14_9_asymmetric_feasibility_preflight_20260809_r1`
- Scheduler tasks: anchor `t78937`, initial continuations `t78988..t78999`,
  additional anchors `t79002..t79005`, and five-replicate continuations
  `t79009..t79020`
- Frozen algorithm revision: `40e86c2f99f7f22d72a89d7ba4dc4799094fd380`
- Frozen source manifest: `4be2354c2383778f6c28507684e719d5a83e841aab652623444ffc0f0e3db5ca`
- Environment: `HalfCheetah-v5`
- Held-out paths per optimizer replicate: 5 disturbance modes x 8 evaluation seeds
- Evidence role: development preflight only

The first fresh optimizer replicate screened all registered v14.9 arms. Only
the matched comparator and the two most informative asymmetric candidates were
then extended to five fresh optimizer replicates. This adaptive extension is
not a frozen full-screen analysis and cannot be used as confirmatory evidence.

## Calibration

The projection-only `s=0.5` arm and its zero-strength control selected identical
actor tensors and produced identical latent-policy, executed-action, reward,
and return traces on all 40 held-out paths. Routing reduced reported lower
responsibility drift by 39--49% and effective upper-HF RMS by approximately
23%, while latent endpoints were unchanged. This again validates the
function-preserving coordinate transform, not learned frequency separation.

## Five-replicate extension

The `u=0.30, l=3.00` candidate changed the policy and improved return relative
to the compute- and selector-matched zero-dual comparator in four registered
modes. Paired optimizer-replicate bootstrap intervals supported strict return
improvement in standard, high-frequency, mixed, and OOD-chirp conditions. The
low-frequency condition remained inconclusive.

| Mode | Mean return difference | One-sided 90% bootstrap interval | Latent upper-HF reduction | Latent lower-LF change |
|---|---:|---:|---:|---:|
| standard | +131.77 | [+53.06, +209.03] | 17.3% lower | 2.3% worse |
| low-frequency | +15.56 | [-89.88, +121.00] | 7.8% lower | 5.3% worse |
| high-frequency | +94.96 | [+19.26, +170.66] | 29.5% lower | 26.5% worse |
| mixed | +86.49 | [+7.92, +165.04] | 24.2% lower | 40.4% worse |
| OOD chirp | +138.40 | [+93.58, +183.22] | 12.3% lower | 9.6% worse |

The five optimizer-level mean return differences, after averaging the 40
held-out paths inside each optimizer replicate, were `+254.95`, `+58.54`,
`+82.03`, `-45.63`, and `+117.28`. The result is therefore not a path-level
pseudoreplicate claim. Nevertheless, neither this arm nor `u=0.20, l=3.00`
passed the registered lower responsibility, raw-behavior, and latent-behavior
reduction gates. There is no joint candidate and the frozen 576-cell screen is
not launched.

## Deployment-alignment defect

The preflight exposed a more fundamental problem than dual-rate scale. PPO
rollouts compute frequency costs from stochastic Gaussian actions, initialized
with `log_std=-0.7` (standard deviation approximately 0.50), while all reported
held-out results deploy the deterministic actor mean. Across the five
`u=0.30, l=3.00` continuations, the final sampled-rollout upper constraint mean
was `0.723` and the lower mean was `0.00676`; the corresponding deterministic
held-out action-frequency powers were roughly one order of magnitude smaller.
The resulting mean final multipliers were `7.09` and `0.636`.

Thus the primal-dual update mostly penalized exploration noise rather than the
policy that was evaluated. This is an objective mismatch, not a tuning failure.
A successor must preserve stochastic PPO exploration for reward learning but
compute a separate differentiable frequency constraint on the deterministic,
squashed actor-mean trajectory, with episode boundaries and upper-action hold
durations represented exactly. Stochastic-policy and deterministic-deployment
metrics must also be reported separately.

## Decision and claim boundary

Allowed: v14.9 provides development evidence that asymmetric dual rates can
improve return and upper-frequency behavior in HalfCheetah, while exposing a
robust lower-frequency failure and a sampled-versus-deployed constraint
mismatch.

Forbidden: v14.9 supports joint learned frequency separation, no-tradeoff,
cross-environment generality, confirmatory reward improvement, or a selected
algorithm for submission.
