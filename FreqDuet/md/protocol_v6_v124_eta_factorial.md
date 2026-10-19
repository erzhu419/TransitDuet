# V124: close the consumed forecast x holding-rule factorial

Retain all 24 V123 paths; add only split_bilateral025 on actors 12011/933013 x
consumed days 2231001--2231004. Same .25 bilateral gain, seven bins, feasible
masks, ready-time V122 predictor, 3600 s completed-prefix window and raw fallback.
The new arm has no innovation or quota/expiry; no actor/critic/dual training.

Server smoke: V122/V123 request/V124 tests, then both original bilateral controls
on day 2231001. Require all cached endpoint/account/coverage fields and full
timing/launch arrays exact before the two split arms run. Verify actual commands,
completed holding and ready-clock forecasts; both actors must exercise changes.
Reuse these two split paths, then run only the other six consumed cells. Total:
eight missing-arm trajectories plus two raw-control replays; no new demand day.

Report all six pairwise contrasts plus interaction:
`(split_funded - raw_funded) - (split_bilateral - raw_bilateral)`.
Use original 5000-draw crossed actor/day intervals and joint passenger-time/CV2/
readiness/completion gates without changed margins. Preserve V123 effects and
inspect day 2231004 separately alongside the complete grid, not as a new test.
This separates forecast feedback from the combined innovation/quota/expiry
package, not each package component. All runs/tests/aggregation through scheduler
on node001--node006; return small receipts/JSON only. Main remains frozen.

## Execution

Source `60c5b44c1a`, scheduler `t195159` on node006, six workers maximum.
All 16 tests passed; both raw controls reproduce endpoints and full timing
exactly. The two split smoke paths passed action-execution/ready-clock checks
and are reused. The remaining six paths precede the server-side factorial summary.

## Outcome and next decision

All ten runs and aggregate completed; actual child exit 0, only 93,997 bytes
returned. Split bilateral minus raw bilateral: journey -0.253047 min, readiness
-22.09 s, holding -22.68 s/trip; CV2 -0.002170 [-0.005504,0.002035], joint gate
fails. Versus raw funded the joint gate passes; versus split funded, journey
+0.011829 min, holding +6.17 s/trip, CV2 -0.001606 [-0.003287,0.000173].
Both actors retain lower passenger time/readiness in all eight cells versus raw
bilateral, but both still worsen CV2 on day 2231004 (+0.004356/+0.003579).
Thus the regularity failure is not unique to innovation/quota/expiry. The package
attenuates the forecast journey gain (interaction +0.065445 [0.006599,0.111355]
min), without establishing separate causal effects of its components.
Keep main frozen; no new-day confirmation or training yet. Next read the four
consumed arms' actual release, cumulative holding and downstream arrival timing
by time/stop, locating the first regularity divergence before changing a rule.
