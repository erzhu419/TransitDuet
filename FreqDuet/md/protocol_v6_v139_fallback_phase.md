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
