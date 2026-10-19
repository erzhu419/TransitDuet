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
