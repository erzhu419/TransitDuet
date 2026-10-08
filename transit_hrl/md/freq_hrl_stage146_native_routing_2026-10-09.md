# Stage146: Native Routing Performance

Stage145 passed full learned-control equivalence on both roots: 136 episodes,
6,829,878 steps, zero action/network differences. Stage146 keeps that native
RE-SAC backend, headway/holding actions, rewards and harmonic/OD estimator.

`correct` keeps original FreqDuet inputs. `raw_history_common` replaces the
upper's dynamic LF value/slope with current/difference of two realized bins,
and the lower block with four realized local bins. `swapped_common` sends HF
value/slope upward and the four native LF features downward. Upper forecast,
HF energy and both OD slots stay common; dimensions stay 16/33 with identical
network sizes. The forecaster and diagnostic/reward state are not ablated.

The development matrix is eight fresh roots x three methods, 300 training
episodes with the original 30-episode warmup, then four paired evaluations
in each of five regimes: low noise, high noise, hour burst, persistent shift,
and a two-hour peak shift outside the training shift support. Each evaluation
reloads the last trained deployment; fleet is fixed at 12. No checkpoint is
selected using evaluation results. The input timetable has 262 trips (the config
cap is 264), ending at 46,980 seconds. Full episodes preserve that demand window
and the native four-hour clearance horizon, fixed at 61,380 seconds. Only short
software preflight uses 2 training
episodes, smaller replay batches and a 5,400-second clock.

Primary outcome is the equal-regime mean restricted service cost, including
unserved passengers. Correct-minus-control contrasts cluster by optimizer root;
10,000 bootstrap draws and Bonferroni-adjusted intervals cover the two primary
contrasts. Reward, restricted/observed wait, service completion and drift are
secondary. Grouped qualification requires matching network sizes, updates,
clocks and exogenous passenger realizations. Only compact JSON is downloaded;
final inference checkpoints remain in server-side sibling artifact directories.

## Limitations

The controls retain a common upper forecast, so this is conditional routing
attribution, not a pure all-input Swapped test or proof that learning outperforms
forecasting. The historical control is explicitly history-plus-common-context.
Eight-root outcomes are development evidence, not independent confirmation or
domain-general validation. The hour burst is temporal, not station-local.
