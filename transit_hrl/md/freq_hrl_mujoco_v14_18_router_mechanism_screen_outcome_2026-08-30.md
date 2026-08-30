# Freq-HRL MuJoCo v14.18 Router Mechanism Screen Outcome

## Execution

The frozen screen in
`mujoco_v14_18_router_mechanism_screen_20260830_r1` completed all nine planned
environment-by-optimizer-seed cells. Scheduler tasks `t84667` through `t84675`
finished successfully on `node005`; each task used one CPU core and approximately
400 MB RAM. The preregistration records source revision
`0e56e46572153bfaa83971e22fae6c2918bf3c0e`.

## Frozen decision

No global router strength was nominated. Every tested strength preserved the
closed-loop reward floor in all nine cells, but no non-baseline strength reduced
frequency-violation merit in all nine cells.

| Router strength | Reward-safe cells | Merit-improved cells | Median relative merit reduction | Minimum relative merit reduction |
|---:|---:|---:|---:|---:|
| 0.6 | 9/9 | 6/9 | +54.39% | -136.12% |
| 0.7 | 9/9 | 6/9 | +52.31% | -638.56% |
| 0.8 | 9/9 | 5/9 | +20.95% | -1680.06% |
| 0.9 | 9/9 | 3/9 | -241.17% | -3522.59% |
| 1.0 | 9/9 | 1/9 | -604.03% | -6477.37% |

Strength `0.6` reduced merit in every HalfCheetah and Hopper cell. The same
change increased merit in all three Walker2d cells. This direction reversal is
not a seed-only anomaly and rules out replacing the v14.17 strength `0.5` with a
single larger constant.

## Mechanism interpretation

The reward result is useful: causal joint-band routing changed attribution while
preserving executed behavior across all 45 non-baseline candidate evaluations.
The frequency objective, however, is policy- and environment-dependent. A
constant global routing coefficient is therefore the wrong control variable.

The next development hypothesis is a deployment-aligned router adapter that
uses the same frozen closed-loop guard in every environment and searches both
directions around the current strength. This is algorithmic adaptation, not
manual per-environment tuning. It can proceed only if a predeclared bidirectional
grid contains a reward-safe strict merit improvement in every cell. If that
condition fails, router-only restoration is rejected and the remaining latent
endpoints require an actor-level intervention.

## Claim boundary

This is a development result. It supports function-preserving reward behavior
for the tested routing family and rejects a universal fixed strength. It does
not support a final v14.18 algorithm claim, statistical significance, or a
manuscript performance result.
