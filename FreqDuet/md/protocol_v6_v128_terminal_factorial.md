# V128: executable terminal spacing x lower rule

Use the V127 consumed 2-actor x 2-day grid. Cross original/split bilateral025
with no terminal intervention versus the native executable 360 s minimum actual
dispatch spacing. Reuse eight no-intervention controls; run eight treated paths.
No sweep, new day, learned update or main switch. All same-direction trips use
the same spacing bound; preserve published release and the lower target goal.
Existing vehicle movement, return visibility and physical inventory stay intact.

The terminal bound may add unbounded waiting. Primary joint diagnosis uses
passenger time, actual arrival CV2, total published-eligibility-to-launch delay
and completion, not native readiness alone. Record the native readiness and
qualification-clock shift separately; require their exact sum to the inclusive
delay. Reference clocks use only query-time published commitments observed on
the live simulation grid. No saved future return or seed/day identity enters
the coordinator. Policy weights remain frozen; closed-loop states may change.

Server smoke: four tests, then raw/split neutral instrumentation on actor 12011,
day 2231004. Require original endpoints, full timing arrays, and split decision
trace exact, with zero reference-clock shift. Then execute both treated arms;
require actual gaps >=360 s, chronological trip order and nonzero interventions.
Reuse those treated paths; only then run the remaining six. Report all pairwise
contrasts, terminal x lower interaction and crossed actor/day intervals as
consumed diagnostics, never automatic promotion or independent confirmation.
Use scheduler CPU conda on node001--node006, one worker/job; return small summaries.

## Outcome (2026-10-19)

Scheduler t195190 passed four tests and both exact neutral replays; t195191--194
completed the remaining paths and server aggregation, all with child exit 0.
Only the 113175-byte summary and tiny completion markers were returned locally.

Relative to the same lower rule without terminal control, paired means are:

| Lower | Journey, min | Actual arrival CV2 | Inclusive terminal delay, s |
| --- | ---: | ---: | ---: |
| Original bilateral | +0.18535 | -0.017291 | +686.08 |
| Split-prediction bilateral | +0.21043 | -0.015838 | +607.45 |

These directions hold in all four actor/day cells; all trips complete. The split
arm's added passenger cost is mostly waiting (+232303 person-s); holding falls
11.54 s/trip. With spacing in both arms, split improves journey by 0.20630 min
but worsens arrival CV2 by 0.001783 (crossed CI [0.000480, 0.003286]). All three
registered joint gates fail. No promotion, training expansion or main switch.

Next: use the existing terminal records to quantify accumulated release delay
and available recovery slack before designing bounded, plan-anchored coordination.
Do not tune another spacing constant on these consumed days. Evidence:
`freqduet/results_freqduet/protocol_v6_v128_terminal_factorial/summary/status.json`.
