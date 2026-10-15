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
