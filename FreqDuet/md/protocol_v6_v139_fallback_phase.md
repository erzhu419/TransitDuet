# V139: read-only fallback phase

Read the same V137 paths and V138 accounting. No replay, new demand, model
update, tuning or control change. Run tests and analysis through scheduler.

Explain the original strict-prefix MovingPrefix gate using saved launch,
arrival, ready and departure events: pending stops, missing earlier departure,
position/segment mismatch and same-clock boundaries. Separate mandatory service
from post-ready stop time. Compare each explanation to the original saved
forecast status. Do not relax the event-completion rule.

Group V138's changed source-switch actions by fallback reason and stop phase;
retain both directions of switching, exposure, passenger bands and vehicle
turnover. Full details stay remote; pull only the final small summary.

Limitations: selected consumed paths, retrospective phase labels, no independent
policy evidence or per-action causal mediation; stop-level queues are not saved.

Outcome: source66688f6737, t195271/node006 exit0; 6 tests pass. Original gate
statuses are explained throughout all12 paths. Final summary186272 bytes only.
Each path has942-1012 fallbacks: pending stops dominate; only2-5 are next-stop
position boundaries. No missing-component-history status occurred.
12071/2351015: 80 predicted->fallback changed actions add350s/11900person-s;
81 reverse switches remove165s/5740person-s. All are pending-stop gates; only
10/15 respectively touch a same-clock event.12143/2351015 nets+15s from source
switches but hides+365s/-350s in opposite directions.
Trip219/stop8 changes from in-transit to mandatory-service evidence, not a
same-second boundary: ETA547.22->879.43s, actual follower-gap change+4s,
command5->45s. This isolates a phase-coverage discontinuity, not a prefix bug.

Next: design a passive stop-phase ETA using strictly completed duration labels
and currently observed elapsed service/known issued holding. Check errors and
phase transitions before any control trial; keep gain, bins and main frozen.
