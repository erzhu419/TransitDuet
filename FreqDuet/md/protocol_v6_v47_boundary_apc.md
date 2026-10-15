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

Scheduler-only smoke, 12 independent acquisition days, then 612 independent
forecast windows and server aggregation. Retrieve small summaries only. Keep
V46 untouched; report raw/calibrated holding and passenger errors, day-level
paired differences, timing scores, missing-leg support and APC audit separately.
