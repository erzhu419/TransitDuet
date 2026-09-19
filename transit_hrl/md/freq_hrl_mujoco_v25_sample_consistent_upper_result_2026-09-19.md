# MuJoCo v25 Sample-Consistent Upper Result

## Decision

V25 stops. Tasks `t93706` through `t93753` completed 48/48 unique cells and
1,920 evaluation episodes. The preregistered `action_sample` candidate failed
the validity, reward, correction, and Hopper correction gates. It must not
advance to confirmation.

All cells report algorithm revision `854a65a6e0b39c1174eab8c6eba00023ac1ecf5b`
and one source manifest. These results use the old finite-iteration projector;
they are not evidence for the later optimized solver at `a047161cc8`.

## Frozen Gates

| Gate | Result |
|---|---|
| Validity and capacity | FAIL validity; PASS matched capacity |
| Candidate reward wins | FAIL: 5/12, with environment counts 2/4, 2/4, 1/4 |
| Mean reward improves in at least two environments | FAIL |
| No reward regression beyond 5% | FAIL |
| No correction regression beyond 5% | FAIL |
| Hopper total correction at most 0.25 | FAIL: 0.3622 |
| Component and total improve versus zero in two environments | PASS |

Validity failed on 2/1,920 paths in two zero-consistency cells. Both exceeded
the frozen recursive-fallback limit of 0.05: Hopper mixed reached 0.06944 and
Walker2d low-frequency reached 0.05042. Certificate violations were zero and
maximum upper/lower prefix powers remained within their frozen tolerances.
Removing these two paths would not rescue any performance gate.

## Performance

Means average four optimizer roots and 40 held-out paths per root.

| Environment | Zero reward | Raw-mean reward | Raw-sample reward | Action-sample reward | Action vs raw mean | Wins |
|---|---:|---:|---:|---:|---:|---:|
| HalfCheetah | 1677.50 | 1433.57 | 1647.12 | 1521.58 | +6.14% | 2/4 |
| Hopper | 186.18 | 176.66 | 190.49 | 176.26 | -0.23% | 2/4 |
| Walker2d | 178.06 | 176.02 | 208.44 | 161.43 | -8.29% | 1/4 |

Action-sample reduced component correction versus zero in Hopper and Walker2d,
but regressed against raw mean. Its total correction was worse than raw mean
in every environment: +47.4% HalfCheetah, +25.7% Hopper, and +192.8% Walker2d.
The Hopper increase is paired-positive in all four optimizer roots, so this is
not explained by one outlier root.

The reward differences are heterogeneous. Paired four-root mean deltas versus
raw mean are +88.01 HalfCheetah, -0.40 Hopper, and -14.58 Walker2d; the small-n
95% t intervals all cross zero. Development rejection follows the frozen
gates and observed correction failure, not a post hoc significance claim.

## Mechanism Boundary

`raw_sample` has the highest mean reward in all three environments, but it was
registered as diagnostic-only and cannot be selected after seeing outcomes.
This is a useful hypothesis: action-space tanh derivatives may shrink or
distort the correction gradient, while a fixed raw residual may retain a more
useful update. It is not evidence that raw-sample is superior.

A future raw-sample candidate would require a new frozen protocol, fresh
optimizer/train/selection/evaluation roots, and explicit correction gates.
Do not spend a confirmatory matrix on action-sample. Before any new large run,
use the optimized solver and add deterministic ordered process-parallel
rollouts; the current cell performs 1,616,576 rollout steps serially.

## Reproduction

Run `scripts/analyze_mujoco_v25_sample_consistent_upper.py` on
`results/mujoco_v25_sample_consistent_upper_development_20260914_r1`.
The analyzer writes `analysis.json` and now records failed validity paths and
maxima. Full checkpoints and histories remain server-only; the local evidence
contains the preregistration, 48 summaries, and evaluation rows.
