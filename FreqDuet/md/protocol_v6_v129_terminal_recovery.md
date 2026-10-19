# V129: read-only V128 terminal delay and recovery

Read the eight V128 treated terminal records and their eight V127 controls.
No rollout, new day, parameter search, training or main-policy change.
Use the existing CPU environment through scheduler; return only summary JSON.

Reconstruct each launch from its query-time published clock, preceding actual
release +360 s, and selected vehicle's observed dispatch visibility. Retain any
unexplained residual. Balance end-of-direction lateness against inherited delay,
short published gaps, recovery slack and new supply constraints. Follow the same
vehicle's prior trip to expose carried launch lateness. Quantify conditional
immediate release advance, sticky-eligibility clocks and vehicle remapping.

Six synthetic tests cover persistence, recovery, short gaps, turnover inheritance,
unexplained waits and paired return propagation. Match recorded V128 delay
endpoints; split matched return changes into prior launch, holding and other time.
All quantities are consumed-path accounting, not a new controller or an estimate
of downstream policy benefit. Decide on any bounded plan-anchored candidate only
after these records show where delay can actually recover.

## Outcome (2026-10-19)

Scheduler t195195 and t195197 exited 0; final six tests passed. All eight paths
reconstruct exactly. Per direction, positive planned slack totals only 7--111 s,
versus final lateness 1093--1964 s. No vehicle/prior-trip remapping occurred.
Across the eight matched-return cohorts (250 returns each), return visibility
shifts +462--718 s: prior launch +462--734 s, holding decreases 8.5--16.2 s, other
elapsed -0.94 to +9.32 s, visibility lag unchanged. Reference shifts average <0.05 s.

Conditional hard-spacing wait averages 156--157 s/trip; instantaneous release
advance preserving recorded past/order/vehicle is 146--152 s/trip. These are not
closed-loop benefits. Native readiness includes 13.6--16.4 s/trip of hard-spacing
wait, so it is not pure vehicle-supply delay. Final small summary: 65957 bytes at
`freqduet/results_freqduet/protocol_v6_v129_terminal_recovery/return_review/summary/status.json`.

Decision: stop uncapped actual-release-anchored spacing. A next small candidate
should bound discretionary wait per published trip, allow lateness recovery and
use only current published clocks, past releases and currently ready vehicles.
Do not set its budget from hindsight release advances. No controller change or
new training in this analysis; any candidate still needs joint service validation.
