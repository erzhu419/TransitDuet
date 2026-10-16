# V75: structured resource value and independent training-day expansion

Freeze V59's four actors (two training seeds), upper/B200 deployment, dual,
production values and main. Retain original complete encoded state and append
38 controlled-vehicle/terminal resource features. Inbound vehicles supply the
opposite terminal after return; use current AVL/APC, pending commands and
published releases, never future returns/launches or prospective assignments.

Fix additional training days 1750001-1750032; independent validation days
1751001-1751012. Smoke days 1759901/1759902 are separate. Prior V65 used
750001-750016, so those are not fresh. Keep four actors x three windows x
32 futures x all seven actions: 176 new native references / 118272 action
rollouts, including validation. Preserve exact prefixes, common future startup,
logical-event CRN, forced-native smoke and once-only system credit.

Sole primary: expanded44/structured. Train on old12 alone and old12+new32;
both budgets fit action-only, original-state, full-fleet and structured models
with unchanged signed quadratic action contrasts, day-balanced ridge 1 and
five full-day native-action targets. Add the zero-difference predictor. The
same-representation budget contrasts separate data expansion from representation.
Freeze all eight fits before reading validation labels; no model/gate selection.

Once on twelve independent validation days: primary MAE/MSE CI uppers <0
versus zero and the three expanded-budget controls for all three continuous
outputs. Joint gate: passenger upper <0, all four constraint uppers <=0.
Keep day-cluster inference and every negative result. Passing does not switch
the controller; single-intervention evidence precedes closed-loop validation.

Server smoke: eight focused tests, V70 seven-action/event identity, fresh native
references/three windows with forced-native replays, real old-cache resource fit.
Then native collection, 24 parallel label shards, fixed fits and validation via
scheduler on node001-node006. Only small completion markers/summary return locally.

Next: adjudicate the fixed primary and same-budget controls once; keep main
frozen until a separately validated closed-loop control candidate exists.

## Execution

Source `3a2617340a`. Scheduler: smoke `t165783`, references `t165784`,
24 label shards `t165785-t165808` (12 workers each), fixed fits `t165809`,
validation `t165810`. Allowed nodes are node001-node006 only. Smoke passed:
eight tests, exact legacy seven-action/event identity, four forced-native groups,
34 replays and real old-cache fitting (279 primary inputs). No formal validation
day was used for smoke. Formal native collection and the label matrix are released
by completion-marker dependencies; actor, dual and main remain frozen.

## Outcome

All 28 tasks completed, exit 0/no retries. On 12 independent validation days
(144 prefixes, 32 paired futures each), both registered gates fail.
Passenger MAE for structured old12 / expanded44: 2014.23 / 1725.92 person-s;
paired improvement -288.30 [-535.79,-69.15]. Expanded original-state MAE is
1728.21, full-fleet 1834.90, action-only 1649.13, zero 1785.79. Structured
versus expanded original-state MAE -2.29 [-34.10,26.65], not a demonstrated
representation gain. Readiness MSE improves versus zero, but not all prediction
criteria pass. Expanded structured changes 58/144 actions: 9 passenger gains,
47 harms, 2 ties; 5 jointly nonworse point estimates. Mean exposure increases
260.11 [111.46,434.48] person-s per single intervention, holding increases
14.68 [10.43,19.18] seconds, and 11/12 day means worsen. This is not a joint
deployment/day-total policy effect. Do not promote the candidate or action-only.

Read-only saved-choice diagnosis (`t166084`, source `84d08eb357`) passed two
tests and reproduced all eight models' validation scores/choices and the primary
effect summary. Of 58 overrides, 54 increase holding; 52 choose 45 s. Across
all prefixes, predicted passenger effect is -519.64 [-708.98,-350.48], actual
+260.11 [111.46,434.48]. Of the 54 longer choices, 43 harm passengers, 9 help,
2 tie. Thirty-six overrides harm in both fixed 16-future halves; only 3 help in
both. Primary passenger sign/order agreement is about 56.6%. In-sample choice
errors also persist (200 longer choices, 161 passenger harms), so cross-day
overfit is not a sufficient explanation. All four actors have positive harm CIs.

Next diagnose seven-action response-shape mismatch and large-tail contributions
on development labels before another model fit. Data expansion improves global
error, not decision calibration; do not add days or tune gates as the sole fix.
Actor/dual/main stay frozen. Limitations: only two original training seeds;
the same-mean32 best feasible choice is an optimistic, label-selected diagnostic,
not a validated controller. Validation days are now consumed; later tuning makes
them development data, not fresh confirmation for the resulting candidate.
