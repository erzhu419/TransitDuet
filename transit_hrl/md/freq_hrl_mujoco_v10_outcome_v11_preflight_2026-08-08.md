# MuJoCo v10 Outcome And v11 Preflight Record

Date: 2026-08-08

## Evidence Role

All results in this record are development evidence. They are not fresh-seed
confirmatory results and must not be reported as such in the manuscript.

## v10 Final Decision

The complete 36-cell v10 matrix used revision
`4195ddca121cec5db1d71acfe5f90cbe866d4e91`. Both arms contain three
environments, two methods, and three optimizer replicates, with all registered
held-out disturbance rows and independent checkpoint hashes present.

The independent gate returned `causal_transfer_gate_failed`:

| environment | method | return delta | reward NI | drift reduction | row gate |
| --- | --- | ---: | --- | ---: | --- |
| HalfCheetah-v5 | no leakage | -43.4789 | failed | 81.75% | failed |
| HalfCheetah-v5 | safe selector | +32.3818 | passed | 61.79% | passed |
| Hopper-v5 | no leakage | -2.8693 | passed | 51.54% | passed |
| Hopper-v5 | safe selector | -30.2233 | failed | 78.01% | failed |
| Walker2d-v5 | no leakage | +27.3880 | passed | 82.65% | passed |
| Walker2d-v5 | safe selector | -2.1389 | passed | 17.51% | passed |

The one-branch structural gate and complete safe-method gate both failed. v10
therefore cannot advance to confirmatory seeds. The result rejects the claim
that analytical nominal-action reconstruction alone is sufficient for
retrained return invariance.

Frozen result locations:

- additive matrix: `mujoco_responsibility_v10_additive_20260808_r1`;
- transfer matrix: `mujoco_responsibility_v10_transfer_20260808_r1`;
- independent decision: `mujoco_responsibility_v10_analysis_20260808_r1`.

## Failure Mechanism

v10 exposed decomposition-specific responsibility anchors and lower LF state to
the actor. It also reconstructed the actuator input by separately casting and
re-adding upper and lower responsibility components. Even when the analytical
reconstruction error was around `1e-8`, the mode-specific actor state and
finite-precision actuator path allowed trajectories and retrained policies to
diverge. MuJoCo dynamics amplified those differences.

## v11 Repair

v11 uses revision `8e47614f1005d8a064a3d6691a0ca6e5bb311ee4` and source
manifest `002878a554049947768f7c1b654d92bc58ca332a272ba422bacd0764336bf5f7`.
It introduces:

1. a canonical actor/reward-critic state that is invariant to the
   responsibility mode;
2. a separate causal responsibility state visible only to the lower cost
   critic;
3. one canonical raw actuator sum, avoiding separately rounded reconstruction;
4. exact paired checkpoint and held-out path gates for the no-leakage branch.

## v11 Paired Preflight

The paired HalfCheetah preflight trained additive and transfer no-leakage arms
with the same source, optimizer seed, rollout seeds, and budget. The independent
checks found:

| check | result |
| --- | --- |
| source revision and manifest | identical and verified |
| frozen parameter SHA-256 | identical (`c54133eba32b...`) |
| held-out episode return difference | `0.0` |
| raw lower action RMS difference | `0.0` |
| raw lower LF drift difference | `0.0` |
| responsibility-level LF drift | `3.6636e-6` to `3.0684e-6` (-16.25%) |
| transfer reconstruction RMS | `4.3968e-11` |

This preflight passes the registered implementation gate and demonstrates that
the repair covers PPO training and checkpoint selection, not only a frozen
policy rollout. It does not establish the full cross-environment gate. The
fixed 36-cell v11 development matrix must pass before any fresh confirmatory
seeds are permitted.

