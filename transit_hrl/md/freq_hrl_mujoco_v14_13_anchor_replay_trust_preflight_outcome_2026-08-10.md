# MuJoCo v14.13 Anchor-Replay Trust-Region Preflight Outcome

Date: 2026-08-10

## Frozen run

- Run: `mujoco_v14_13_anchor_replay_trust_preflight_20260810_r1`
- Scheduler tasks: `t79642..t79654`
- Terminal states: 13 done, 0 failed, 0 cancelled
- Placement: anchor on `node004`; 12 continuations on `node001`
- Core protocol: `freq_hrl_mujoco_shared_core_v14_13_anchor_replay_trust_region`
- Algorithm revision: `9f98c57572279611823445ee1e908c73833eeace`
- Source manifest: `fbb3b46e0fb05465b35fbbe4e9c0f7c1e8d6260560924bf5a44c0399071be0e5`
- Evidence role: single-optimizer-seed mechanism preflight, no confidence interval

No `jtl110cpu` artifact was used. The run-scoped sync and merge admitted all 13
cells only after validating source identity, anchor checkpoint provenance,
training history, checkpoint, and the registered evaluation grid.

## Decision

- Calibration: passed
- Eligible arms: none
- Selected arm: none
- Decision: `do_not_expand`

All nine learned arms selected the iteration `-1` anchor fallback. Therefore
the reported held-out candidate metrics equal the matched comparator and cannot
support a learned improvement claim.

## Mechanism audit

The implementation defect targeted by v14.13 was repaired on the training
paths. Replay and trust provenance passed for all requested arms, finite-budget
joint arms recorded zero final group reward-budget violations, and the joint
`eps=0.01` arm reached the same-batch upper and lower frequency targets after
all 32 updates. The actor nevertheless failed to generalize those constraints
to deterministic closed-loop checkpoint-selection paths.

| arm | best trained iteration | worst normalized violation | reward-floor failures | frequency failures | worst endpoint |
|---|---:|---:|---:|---:|---|
| replay only, eps=0.01 | 23 | 0.1428 | 4 | 6 | mixed upper HF |
| trust only, eps=0.01 | 7 | 0.5682 | 2 | 9 | high-frequency latent upper HF |
| replay + trust, eps=0.001 | 23 | 0.1589 | 0 | 8 | low-frequency raw lower LF |
| replay + trust, eps=0.005 | 11 | 0.2474 | 2 | 8 | low-frequency latent upper HF |
| replay + trust, eps=0.01 | 11 | 0.2474 | 2 | 8 | low-frequency latent upper HF |

The initial fallback has normalized violation `0.0526` by construction because
it provides zero reduction against a registered 5% target. Every trained
checkpoint had a worse lexicographic worst-condition rank, so fallback was the
correct frozen decision.

## Diagnosis

v14.13 constrains deterministic actor outputs on current training states plus
four frozen anchor-state trajectories. This is still an open-loop state
distribution contract. Updating either actor changes the subsequent MuJoCo
state distribution; an actor can satisfy every same-batch constraint while its
new deterministic closed-loop trajectory increases upper HF or lower LF on
unseen seeds. Tolerance and projection-step sweeps cannot repair that mismatch.

The next admissible mechanism is a deterministic closed-loop trust gate on a
fresh guard-seed role, separate from training, checkpoint-selection, and held-out
evaluation seeds. It must backtrack or reject the joint actor update using
actual rollout reward floors and effective/raw/latent frequency endpoints.

## Claim boundary

v14.13 is negative development evidence. It supports the narrower statement
that frozen anchor-state replay plus a same-state PPO trust region can preserve
training-path frequency and reward constraints. It does not support an accepted
learned checkpoint, held-out frequency separation, reward improvement,
no-tradeoff behavior, cross-task generality, statistical evidence,
confirmatory evidence, or a submission-ready algorithm.
