# V47: terminal legs and APC attribution

Freeze U200/L200 and V45 holding/flow snapshots. Read-only acquisition adds
successful terminal launch clocks, return clocks and raw APC service events;
require unchanged full-day outcomes and exact V45 flows at all 51 origins.
Pair terminal clocks with adjacent stop events, retaining strict completion
before issue. Re-run V46 shared travel blocks with all directed legs observed.
Use all 12 previous days / 612 windows, explicitly as regression data.

APC audit retains untruncated inferred arrival mass: board + left - previous
left. Negative mass invalidates the data contract. Compare legacy 32-gap rates
with a predeclared four-gap weighted rate using only completed prefix gaps.
Future APC service clocks, preload and counts are teacher-forced audit labels,
not rollout inputs. Score next-service preboarding queues, completed-gap arrival
counts and alighting counts. This isolates flow errors before a new stochastic
flow model; do not change the controller or claim counterfactual validation.

Scheduler-only smoke, 12 acquisition replays, then 612 independent computation
forecast windows and server aggregation. Retrieve small summaries only. Keep
V46 untouched; report raw/calibrated holding and passenger errors, day-level
paired differences, timing scores, missing-leg support and APC audit separately.

## Results And Next Step
Acquisition `c933f6112b` / `t155658-t155662`: all 12 frozen controls and all 51
prefix flows/day match; no negative inferred arrival mass. Forecast source
`ac79f900b9`: smoke `t155664`, 45 tests; matrix `t155665-t155669`, 308 workers;
aggregate `t155670`. All 612 windows have samples for all 42 directed legs.
Earlier smoke corrected a merge-order test assertion (`t155654`, retry cancelled)
and CSV direction inference (`t155656` acquisition valid, forecast excluded).

All-origin 30-minute holding MAE, development/reused/formerly-fresh demand:
V46 415/453/406 s, V47 415/473/423 s. Backlog MAE worsens from 294/327/263 to
336/367/322 trip-seconds. Onboard MAE improves modestly, but waiting remains
64,549/87,984/54,071 person-seconds and holding coverage only 64.7/59.3/67.2%.
No overall closure or forecast promotion; actual controller stays frozen.

APC audit: four-gap arrival-count MAE worsens in each cohort (1.631->1.666,
1.773->1.838, 1.683->1.738 passengers). Next-service queue MAE improves only
slightly (1.635->1.606, 1.815->1.739, 1.694->1.646). The old mass bookkeeping
is correct; shortening history alone is not supported. Preserve V46 as the
forecast reference and retain the completed telemetry. Next test stochastic
arrival/alighting and initial-queue propagation with historical-prior means,
capacity and passenger-mass conservation, before fresh-context/action validation.
Canonical small outputs: `protocol_v6_v47_boundary_apc/summary`.
