# V137: passive ready truth and holding exposure

Freeze the six V136 example pairs:12071/2351015,12143/2351015,
933059/2351002,12071/2351007,12037/2351004,12109/2351014.12 consumed-path
replays, zero new demand days or parameter updates. No terminal coordination.

Run eight focused tests and one exact old/handoff pair before the remaining
ten paths, through scheduler on Linux CPU. Require identical V135 endpoints,
full timing/actions, sparse original evidence/forecasts and terminal paths.
Passively save existing ready labels, query contexts and holding-person
counters; conserve exact system exposure in15-minute bands. No future label
or saved trajectory enters the live controller.

Compare original and handoff forecasts against the same trip/bus ready truth.
Keep identity mismatches and missing/negative ready gaps explicit. Check local
pair-compensation and frozen-event departure geometry, then contrast direct
holding exposure with later action/load changes. These selected development
diagnostics are not new policy evidence or per-action causal values. No tuning
on V135, no training expansion or main-policy switch. Raw traces stay remote;
fetch only completion logs and the small summary before choosing the next step.

Outcome: t195268/node006 exited0; all12 unique paths exactly reproduce V135.
Source485ee348e0;9 admitted3519653af6 paths reused,3 completed after the
native-ledger bin fix. Initial collector failures: JSON clock-key representation
(t195256; retry195257 cancelled) and LF-bin rounding (t195258 stopped). No
scientific tolerance or controller changed. Final summary35746 bytes fetched.

All16 used handoff decisions resolve to the same bus/trip and next departure.
Ready MAE360.490->45.398s;15/16 absolute errors improve. Fixed-follower ready
pair geometry improves12/16; frozen-departure geometry improves11/16, not all.
12143/2351015 improves both local decisions yet adds20309 passenger-person-s:
handoff-stop exposure-1350 versus other-stop exposure+4565 person-s.12071/
2351007 also improves its sole local decision but waiting outweighs onboard
savings. These are outcome-selected diagnostic paths, not independent gains;
local fixed-neighbour geometry is not a causal full-day intervention value.

Next: read these complete traces only. Decompose subsequent request/action
changes into observed forward/follower-gap changes, same-follower forecast error
and discrete-action residuals, separating identity/regime changes. Align with
the saved passenger/turnover timing. No extra replay, refit or main-policy switch.
