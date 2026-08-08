# MuJoCo v14 Initial Preflight Outcome (Superseded)

## Scope

This is a source-bound path validation, not performance evidence. The run used
64 transitions, a 64-step episode horizon, one train/checkpoint/safety/evaluation
seed, and two PPO iterations on HalfCheetah.

This preflight used the initial boundary-rate proxy at revision `1e7c8d5841`.
It was superseded before the development screen by the endpoint-aligned rolling
filter objective at revision `ae5f0c4607`; none of these initial preflight
artifacts may be used in the v14 screen or a paper claim.

- Run: `mujoco_v14_behavior_safe_preflight_20260809_r1`
- Frozen algorithm revision: `1e7c8d58417f16531dbbd98d0b2ccabc0f14ee6d`
- Freq-HRL source manifest: `3b4ac87799a3da7a8513a90757dd025ec5259f19fb7a31a0d2e0901ece07085e`
- Protocol: `freq_hrl_mujoco_shared_core_v14_behavior_safe_training`
- Scheduler tasks: `t75752` through `t75754`
- Nodes: dynamic placement on `node002` and `node004`; `require_node=None`
- Completion: 3/3 cells and 3/3 checkpoints, all result-synced

## Contract checks

All cells reported `source_identity_status=verified`. The joint method reported
`leakage_constraint_scope=joint_behavior` and upper coefficient `2.0`; the
no-leakage path reported disabled constraint scope and coefficient `0.0`.
Responsibility reconstruction RMS was approximately `4.4e-11` to `4.5e-11`.

The safe selector evaluated all five registered branches and fell back to
`no_leakage`: after only two training iterations, no constrained branch met the
10% responsibility and raw-drift development gates. This is a valid preflight
outcome and must not be interpreted as an algorithm comparison.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `freq_hrl/cell_summary.json` | `8a7ee16ba91a6570d6e015551533d76a97aa956565b58e08d35f6b20c250289d` |
| `freq_hrl_no_leakage/cell_summary.json` | `6518a6e3a78425a059716e38a579b9b982c36ae6fd2ee15c17b22299c9d226b6` |
| `freq_hrl_safe_selector/cell_summary.json` | `fc0d633e9dce8fbc062ea976a11751fb3495b1bbeea86bcb8946f440c0f34f06` |
| merged `summary.json` | `e7d084f30b04d57ae3b2922feec44eda272c37eb18edd85e589325c4175559cc` |

The next admissible step is the preregistered v14 development screen. A fresh
v15 confirmatory protocol may be created only after the screen selects and
freezes an upper continuity coefficient.
