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
