# V414: Entry-Hour Running Forecast

Compare the frozen pooled V413 link mean with one passive candidate. At each
slot query, index strictly completed and available visits by direction,
link and observed entry hour (`arrival_s - run_s`). For a full future link,
look up the hour of its simulated launch or departure. With fewer than three
same-hour visits use the original pooled mean. In particular, an upcoming
hour with no completed visits falls back; there is no future data or new prior.

Keep partial-link initialization, locked commands, passenger forecasts,
dispatch semantics and capped bilateral continuation unchanged. Different
simulated clocks can change the continuation's chosen actions; report holding
and service components separately from the running difference. Never submit
shadow actions or replace the baseline forecasts.

**Independent screen.** Proposed one engineering path on old day `2520208`
to prove exact original streams, public-slot records and endpoints, followed
by one slow capped path on each of eight reserved new days. Both forecasts
observe the same live states at leads 300 and 0. No physical paths released
by this protocol. A later release must retain the same budgets, with no retry.

Primary comparison: advance launch-clock absolute error and late-launch
misses/false alarms on the same baseline-supported slots. If the shadow
abstains, score the baseline fallback and disclose the abstention. Report
hourly/pooled usage, eight paired day effects, and queue strata. Compare
matched-cycle return components and first-return rank (same V413 strata and
30 s split); at-clock results are diagnostic only. Continue toward controller
testing only if the mean day-level launch MAE falls, most days improve and
neither aggregate advance error count rises. This is a forecast screen, not a
control effect. Do not tune on V413 or the reserved days.

[Frozen predictor and proposed budget](running_shadow_protocol.json)

**Zero-traffic check.** `t215015` passed six checks on `node004`. No reserved
day conflict was found in checked case directories and 25 protocol manifests.
[Small server receipt](review/running_shadow_checks/results.json)

**Engineering release.** The subsequent release is only the old-day path
`2520208`. Task `t215018` was manually dispatched high on `node004`; one
physical start was confirmed. Acceptance requires exact V413 live streams,
endpoints, public slots and baseline components, plus shadow component closure.
No retry or confirmation day is released.
[Submission receipt](submission_running_shadow_parity.json)

**Engineering acceptance.** `t215018` finished successfully. All 12 frozen
streams, 5,200 random events, 522 public-slot records, endpoints and baseline
components matched. All 492 shadow scenarios completed and all 5,818 shadow
vehicle component records closed. Of 321,548 full-link lookups across the
shadow continuations, 50,370 (15.7%) used the entry-hour mean and 271,178
(84.3%) used the pooled fallback. These are simulated link calls, not the
coverage of target public slots. Engineering parity passes; forecast accuracy
still requires the reserved-day comparison. No new day has been released.
[Small parity result](review/running_shadow_parity/results.json)

**Independent release.** Zero-traffic entry check `t215022` passed. The eight
reserved days `2540001`--`2540008` are running as `t215023`--`t215030`, high and
manually dispatched to nodes 002, 003 and 004. Each wrote one physical-start
receipt. The accepted shadow and live runtime are reused unchanged; no retry,
new policy, or additional path is released. Forecast accuracy is pending.
[Release check](review/running_shadow_release_checks/results.json),
[submission receipt](submission_running_shadow_confirmation.json).

**Independent result.** All eight paths finished once with exit code 0 and
5,200 random-event records each. The read-only scheduler task `t215033`
checked the saved slot and component associations and applied the frozen
gate. Only the small summary was retrieved. The paths took 20.0--39.3 min.

| Advance 300 s, first scenario | V413 pooled running | V414 entry-hour running |
| --- | ---: | ---: |
| Supported, paired slots | 1,929 / 2,080 | 1,929 / 2,080 |
| Launch-clock MAE, equal weight per day (s) | 76.28 | 61.75 |
| Missed late launches | 39 | 27 |
| False alarms | 62 | 32 |
| First-return rank errors, no ready bus or earlier slot | 0 / 368 | 0 / 368 |

All eight day-level MAE differences favor V414 (range -18.27 to -11.72 s;
mean -14.54 s, or -19.1%). No supported launch or rank score needed V413
fallback. The running-time lookup itself used the pooled fallback in 83.8%
of simulated full-link calls; this is a different denominator. Slot-weighted
MAE fell both with earlier slots queued (90.32 to 73.64 s, 1,489 slots) and
without (28.77 to 21.47 s, 440 slots). The second scenario agrees in direction
on its 1,697 supported pairs (76.67 to 61.84 s, misses 31 to 20, false alarms
51 to 28). At the public clock both forecasts classify all 1,944 supported
first-scenario slots correctly; their equal-day MAEs are 53.62 and 42.55 s.

On 1,850 advance slots where both forecasts and the actual launch use the
same inbound cycle, mean return-clock absolute error fell from 84.48 to
68.17 s. Mean signed travel error fell from +12.98 to +2.26 s; service error
changed from +1.71 to +1.60 s and hold residence error from -1.85 to -0.25 s.
These paired component differences do not attribute a control effect.

**Decision.** All four advance forecast conditions pass. Retain V414 as the
frozen forecast candidate for a bounded closed-loop comparison against the
same controller with pooled running forecasts. Keep the fleet, caps, action
space and service endpoints fixed; use fresh paired days and test against the
existing slow-speed readiness and public-delay criteria as well as journey
and actual CV2. The present eight days are consumed; there is no new path
release or main-policy change. This is evidence of more accurate forecasts on
one slow-speed setting, not yet of better control, transferred circulation,
or better rank discrimination (only one advance near-tie).

[Small server readout](review/running_shadow_readout/results.json) |
[Readout task receipt](submission_running_shadow_readout.json)
