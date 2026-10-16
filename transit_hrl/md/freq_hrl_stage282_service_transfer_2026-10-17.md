# Stage282: Learned Service Transfer With Frozen Risk

- Run: `native_transit_service_transfer_stage282_20261017_r1`.
- Fit only service residuals on each full60/prefix54 model's assigned16 labels. Reuse the exact Linear9 and matched-df action/observed73 feature maps, normalizers and penalties. No raw/risk/prior/basis refitting.
- One common service correction clip radius per fit: state-balanced RMS of the training-mean service residual. Report both unclipped and clipped predictions; no label-selected penalty, strength or threshold.
- Cross each learned service with frozen raw / Linear9 / observed-df risk; preserve raw wait feasibility and candidate/tie rules. Zero service update reproduces each frozen controller exactly. Retain the three frozen Stage281 MC-service information benchmarks separately.
- Primary comparisons: each clipped learned service versus its own unchanged-risk controller on mixed-pooled excluded/opposite and full-held/opposite; compare service MSE and cost/wait together. Report all train/retained/excluded/held/prefix-head-held, views/banks and fixed8+8 evaluation halves without choosing a winner.
- One unpinned scheduler task, any node001-node006, 1 CPU / 3072 MiB. Budget: 2112 service solves, 32 cells / 704 contexts / 2048 queries / 84480 repeated state exposures. No native simulations or actor updates. Full models/predictions stay remote; scalar pull only.
- Verification: five tests passed, including exact weighted service solves, frozen geometry/penalties, zero-update controller identity, query clip/benchmark isolation, sparse cached model indices, complete 22-fit-context/64-query replay, group accounting and completed-parent gate. Both cold CLI startups passed.
- Completed: `t166159`, node006, done/0 in 79.55s, no retries. All 2112 service fits / 32 cells qualified; cell/group/control replay exact, max normal-equation error 6.08e-18. Pulled 2,175,802 scalar bytes (310,153 transport bytes), no models or query ledgers.

## Results And Next
Mixed-pooled service MSE x1e6. MC is the frozen Stage281 information benchmark, not a learned predictor.

| Service | Train / assigned | Train / opposite | Prefix excluded / opposite | Outer held / opposite |
| --- | ---: | ---: | ---: | ---: |
| Raw | 71.492 | 71.492 | 71.492 | 111.846 |
| Linear9 clipped | 53.817 | 57.503 | 79.160 | 115.157 |
| Action-df clipped | 50.551 | 54.905 | 76.514 | 111.893 |
| Observed-df clipped | 49.352 | 54.607 | 74.834 | 111.813 |
| MC service | 0 | 35.050 | 35.050 | 41.128 |

Observed-df reduces training-opposite MSE 23.6%, but excluded MSE worsens 4.7% and held changes only -0.03%. Uncut observed MSE is 76.833/110.903 excluded/held, so clipping alone does not explain failed transfer; linear uncut is worse at 116.162/153.249. Prefix54 heads on the same outer-held states give observed clipped 112.891, not an improvement from merely removing training states.
With raw risk, clipped linear/action/observed service changes excluded cost by -81.928/-57.632/-0.292e-6 and wait by +0.000089453/-0.000155892/+0.000202051 min; held changes cost by -252.600/-189.998/-143.179e-6 and wait by -0.000918555/-0.000443262/-0.000499837 min. Thus there are local benefits, but the Stage281 MC-service wait gains -0.009911719/-0.009253255 min are not reproduced. Excluded action-df cost changes sign across evaluation halves; observed-df raw-risk held harms both cost/wait in mixed-b pooled.
Changing frozen risk also changes the conclusion. Observed clipped service versus its observed-risk control improves excluded/held cost -200.345/-34.793e-6 and wait -0.000178125/-0.001032650 min, but excluded service saves only -15.883e-6 while fleet saves -184.462e-6. This is not replication of MC's service-driven gain. Cost signs still change across banks/halves; no risk or service arm is promoted.
Training raw signed service error +0.001984759 is nearly removed; outer-held observed clipped error is -0.001491752 versus raw +0.000335407. This motivates, but does not prove, a nonoriginal-action offset calibration mismatch. Next Stage283: split cached uncut corrections into fitted intercept offset and feature slopes, preserve their cross terms, and evaluate offset-only/slope-only with the unchanged training clip/risk/gate. Attribute decisions to original/nonoriginal switches. Zero new fits or simulations.
- Limitations: reused development single-intervention supervision, not full learned HRL or seed-CI confirmation. Only MC benchmarks use query outcome labels; learned service uses causal features and training-only labels.
