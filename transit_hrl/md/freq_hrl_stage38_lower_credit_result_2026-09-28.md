# Stage-38 Lower Credit Result

Tasks `t102300`-`t102339`: all 40 cells and 2560 evaluation episodes complete.
Forty trajectory audits, 80 final/selected native policy replays, 40 initial
training-credit replays and the independent seven-endpoint bootstrap pass.
All jobs exited zero, without duplicate executions. Source `8c9487b78e`;
full registration `244dc93aad`.

| Primary final-weight return contrast | Mean | Seven-endpoint adjusted CI |
|---|---:|---:|
| Intrinsic-option vs frozen | -10.2575 | [-13.3875, -7.0815] |
| Intrinsic-episode vs frozen | -5.4674 | [-9.6609, -1.4295] |
| Task-option vs frozen | -10.5107 | [-30.5032, -0.4949] |
| Task-episode vs frozen | -16.9592 | [-24.4872, -6.3792] |
| Task minus intrinsic main effect | -5.8725 | [-19.7000, 1.8740] |
| Episode minus option main effect | -0.8292 | [-10.0651, 10.0846] |
| Reward x boundary interaction | -11.2386 | [-23.2165, 4.1994] |

All four lower-update arms harm final return. Neither changing the entire
lower reward nor removing option cuts rescues this recipe. The main effects
and interaction are inconclusive; they do not establish equivalent mechanisms.
Frozen return is 904.1960; learned final means span 887.2368-898.7286.
Selected cohorts remain secondary: 15 of 32 learned cells select iteration0;
their small pooled gains cannot replace the negative final-weight endpoints.

Initial lower values average 0.2120 in every arm. First-batch GAE return
targets average 0.1893/0.3550 for intrinsic-option/episode and 8.6650/12.7197
for task-option/episode. This identifies a critic-calibration diagnostic,
not a cause: even the unchanged intrinsic-option objective degrades.

Method cost: 54192000 primitive steps/1186499 upper/1977116 gate calls.
Verification: 144000 steps/3560 upper/5227 gate calls, charged separately.
Raw arrays and weights remain remote; only the compact summary is local.

## Limitations And Next

Conditional development on eight reused roots, not independent confirmation.
Task reward also removes the intrinsic action penalty; its comparison is
not a pure credit-alignment test. Inherited critics are not recalibrated.
Next isolate critic calibration with a frozen actor and measure policy drift
from the first lower PPO updates, before another full training experiment.
No additional training jobs, seed extensions or post-outcome retuning launched.
