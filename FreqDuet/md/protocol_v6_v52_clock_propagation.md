# V52: matched clock-propagation audit

Freeze the real U200/L200 controller, all V48/V50 fitted models, and passenger /
holding processes. Replay 12 certified days at clocks 6300/7200, 20700/21600
and 35100/36000 (72 origins), 64 draws each, comparing V48, V50 independent and
V50 shared. The paired offsets expose hour crossings absent from the original
common clocks. Require exact
cached scalar-metric replay before accepting each diagnostic origin.

Match trip/stop/stage identities including launch/terminal boundaries. Arrival
error telescopes into initial boundary, launch/first-stop pipeline, travel,
service, decision and holding contributions on complete matched chains. Report
this conditional population separately from capped 64-path arrival scores and
missing-path counts. Break out early/late leads, initial/future trips, traversed
stops and hour crossings. Pair by day and cluster on six reused demand seeds.

Use the result to choose the next repair, not to promote a controller. V50's
mean posterior uncertainty is estimated but not sampled; the 12-feature holding
surrogate also omits actual-policy observables such as service dwell, absolute
target headway and the full route-speed vector. Neither is yet established as
the main cause of late-clock error. Computation stays scheduler-only on the
Linux CPU nodes; retrieve summaries, not raw traces or checkpoints.

Teacher-forced extension compares V48/V50 completed-leg means with frozen
observable AVL movement means on the same future departures. Future arrivals
serve only as labels, even beyond the forecast horizon, avoiding completion-time
selection of fast late-window legs. This extension diagnoses mean mismatch;
it neither supplies future labels to forecasts nor changes their distributions.

## Results
Smoke t160169 and 72-origin matrix t160170-t160174 pass exact replay; summary
t160175. Shared-model late MAE delta versus V48 is +2.24 s [-0.63,5.29]. On
complete matched late chains, travel/service/holding bias contributions are
-28.32/+0.75/+7.78 s. At whole-hour origins, late MAE 94.88 versus 88.06 s
despite larger path spread (70.10 versus 60.04 s): noise inflation alone is not
supported. Teacher audit t160179 passes six tests and finds V48's long-lead
leg MAE 4.32 s lower than V50 [-4.74,-3.94]; observable movement means give
no long-lead gain. V53 therefore tests prefix-learned horizon mean reversion
before changing holding inputs or the real controller.
