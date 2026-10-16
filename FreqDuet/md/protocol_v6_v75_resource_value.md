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
validation `t165810`. Allowed nodes are node001-node006 only. Eight focused
tests passed; smoke replay admission is pending. Completion-marker dependencies
prevent formal labels/fits/validation from starting before their prerequisites.
