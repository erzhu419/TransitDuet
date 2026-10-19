# V143: read-only support and hour-13 diagnosis

Copy V142 bounds and V140/V141 point forecasts without regeneration. Read only
the same twelve V137 paths, no replay or training. Explain support loss as
no completed history, fewer than nine total prefix labels, a sparse one-hour
window, or elapsed-duration survivor depletion. Preserve the original fallback
rule: no survivors does not trigger an expanded window.

For hour13, report paths, stations, directions and lower/upper misses, observed
duration versus completed historical durations, and pending-label selection.
Pending final durations are retrospective diagnostics, never predictor inputs.
Reproduce V142 coverage and bounds before interpreting the decomposition.
No tuning on these consumed selected paths; main policy stays frozen.

Next: freeze an independent calibration-evaluation design separately; no new
path collection or control trials are authorized by this diagnostic stage.
