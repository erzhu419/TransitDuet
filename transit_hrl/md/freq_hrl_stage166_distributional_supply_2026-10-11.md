# Stage166: Expected Peak Cost, Not Cost at Mean Duration

Stage165 is a valid zero-effect result: both roots reproduce 20/20 legacy
schedules and all selected strengths; cost/wait/reward deltas are exactly zero.
Mean duration fixes the startup estimate without fixing the supply objective's
lost tail cost. Keep that result; do not train the negative allocation frontier.

Change only candidate fleet cost to the mean native peak penalty over 64 fixed
duration scenarios. Each uses direction-centered individual-trip residuals
from all five Stage164 fit scenes, added to the same prior/observed EWMA mean.
Use local RNG seed 166000+root and stable trip IDs; candidates and successive
decisions share draws without touching native RNG. Active durations retain the
same progress fraction. First completion still replaces the prior, with EWMA
.2 thereafter. Each member retains the irreversible observed historical peak.

Three conditions: window forecast, Stage165 calibrated supply, distributional
supply. Same two frozen native lower roots, five regimes, four registered scenes:
120 episodes, 7,365,600 native ticks, 80 exact physics/schedule/selector controls,
zero training updates. Controls execute first in every scene. All strengths,
wait score, fleet target/cap, daily endpoints and 44 six-trip commitments stay
unchanged. Report actual cost/wait/reward and fleet/nonfleet deltas versus both
controls, plus candidate fleet discrimination and historical-floor plateaus.

First check fit-only variance against independent Stage164 check trip times,
conditioning on their recorded launch schedules. This measures duration-model
peak cost, not prospective control. Then run the matched closed-loop matrix.
Fit residuals are frozen before either result; no quantile/weight tuning.
All ten schedule-conditioned check panels improve fleet-cost error: mean absolute
error falls from 0.416667 to 0.111458. The ensemble still underestimates high-noise
peak cost. 39 focused tests pass, including unchanged mean controls, scalar/batched
occupancy equivalence, shared RNG draws, fit/check isolation, causal EWMA, native
commitments, score/histogram analysis and scheduler/worker budgets.

Ten unpinned scheduler tasks on node001-006, 1 CPU/3 GB each. Stage only code and
the new run's frozen model inputs; historical result/raw directories are never
input sync targets. Only compact result JSON is pulled; CSV/checkpoints remain
remote. Learned allocation remains contingent on physical value on both roots.

Run `native_transit_distributional_supply_stage166_frozen_20261011_r1`, revision
`47581e6da1`: t142266-t142275 are running on node001/004/005/006. The initial
input-staging queue cleared through scheduler dispatch. Root397's first window
and calibrated-supply controls reproduce exactly. Full qualification/performance
remains pending; the input directory is approximately 80 KB.

## Limitations

Independent residual sampling omits correlated disruptions and action-dependent
holding/capacity effects. The fixed 64-member mean has Monte Carlo error. Reused
development scenes and frozen lower policies do not establish learned upper
authority, cross-root confirmation or end-to-end Freq-HRL effectiveness.
