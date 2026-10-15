# V48: passenger-process uncertainty

Freeze the real U200/L200 deployment, V46 shared travel paths, seven-bin holding
model and 32-gap APC arrival/alighting means. V47 supplies verified raw APC
prefixes; its regressing terminal travel model is not promoted. Reuse the same
12 service-day replays / 612 origins as engineering regression, not fresh tests.

Compare initial_queue, arrival_queue and passenger_joint. Draw initial queues
as last observed left-behind stock plus Poisson arrivals since that observation.
The second adds Poisson arrival events; the third adds binomial alighting with
integer onboard/queue states. Tick-aggregated arrivals precede vehicle service.
Only the first two ablations retain expected/fractional alighting. Preserve
capacity, passenger mass, published service, physical reuse and action masks.
Private RNGs and full 30-minute event draws preserve 15-minute process prefixes.
Do not fit to future APC or use latent demand/queues. Historical means stay fixed;
rate-parameter uncertainty and cross-station demand dependence are not modelled.

Smoke: focused tests, exact stored V46 path reproduction without the new process,
then all three modes. Scheduler-only 612-origin matrix and server aggregation;
retrieve small summaries. Report raw/calibrated errors, CRPS, coverage/width,
state timing and day-paired deltas. Keep completed-window calibration observable
only. Full raw 30-minute screen per cohort: holding MAE <= V46 and historical,
absolute bias <=5%, passenger MAE <=105% of historical, holding coverage >=80%.
No controller change without fresh-context and action-response validation.

## Outcome And Next Step
`77ea3b2eb0`: smoke `t155712`, 53 tests; matrix `t155720-t155724`, 307 workers.
611 windows succeeded; window 62 hit fractional-capacity roundoff (50.00000000000001,
mass error 4.5e-13). `708707ce3e` / `t155765`: 54 tests, real-overflow rejection,
exact V46 replay and repair of that one window; aggregate `t155767`, all 612.
Maximum passenger-mass error across the matrix: 7.1e-11. No simulation or mean
parameter changed in the repair; failed diagnosis `t155763` and cancelled retry
are retained in scheduler history. Canonical outputs: experiment `summary`.

Full passenger_joint holding MAE (development/reused/formerly-fresh demand)
415/453/406 -> 409/446/405 s. Matched-event load coverage improves 31-36% ->
76-82%, but raw holding coverage remains 62.7/59.8/66.2%. Reused-demand waiting
MAE is 86,505 person-seconds (historical 80,630); late arrival-clock MAE remains
91/84/88 s. All full screens fail. Retain V46 and the V48 diagnostic candidate;
do not promote the real controller. Next audit conditional travel/holding
residuals and their temporal/cross-vehicle dependence using completed prefixes
and multi-day priors, instead of adding further independent passenger noise.
