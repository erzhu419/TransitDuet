# V87: frozen mean-preference / state-response factorial

V86 locates taped-path amplification mainly in waiting after one hour; total
holding and readiness do not explain it alone. Retain reactive lower feedback.
Test whether the B40/B200 deterioration follows average action preference or
state-dependent logits, under the same frozen B200 upper/deployment context.

For each of the original actors 12011/933013, replay 16 independent calibration
days 1870001-1870016 under native stochastic B200. At every current observable
state evaluate both frozen lower actors (ep39/199), on B200's identical input
normalization and action support. Subtract each state's seven-logit mean to
remove softmax gauge. Average the early-minus-late centered-logit difference
within day, then equally across days. No outcome, action-value or future label
enters this constant shift; no reward/dual/critic fitting or step-size search.

On separate days 1871001-1871016, cross mean and centered-logit response: native
late, native early, late plus the fixed early-minus-late mean, and early minus
that mean. All four remain closed-loop, causal seven-bin policies. Upper B200
reacts normally with frozen weights; all other deployment state remains B200.
This preserves logit state differences, not probability derivatives or exact
marginal holding totals. It is not a physical LF/HF decomposition by itself.

Three logical-event matched sampler repeats per actor/day give 96 paired cells,
384 evaluation rollouts plus 32 calibration rollouts. Report both mean effects,
both response effects and their interaction; compare passenger time, cumulative
arrival CV-squared, readiness, pending trips, holding and its passenger exposure.
Demand-day intervals are conditional on two actors, not training-seed evidence.
No controller is admitted or tuned on this diagnostic. If mean shifts fail,
do not attenuate them against these evaluation labels.

Before batch execution, scheduler smoke must pass algebra/support/day-weight
tests, reproduce native B200 metrics, full event uniforms and physical holding
for both actors, and exercise all four corners. Calibration, mean estimation,
evaluation and aggregation run only through scheduler on node001-node006.
Keep raw paths/checkpoints on servers; retrieve small summaries only. Main and
all actor/critic/dual weights remain frozen.

Source `376437a3d9`; smoke `t166436` passes on node004: five unit tests,
both actors' exact native metrics/event/physical-holding paths, and all four
full-day policy corners. Calibration shards
`t166437-t166440` cover `[0,8)`, `[8,16)`, `[16,24)`, `[24,32)` on
node002/004/005/006, eight workers each, released by the smoke marker. Mean
estimation `t166441` waits for all four calibration markers. Evaluation shards
`t166442-t166445` cover `[0,24)`, `[24,48)`, `[48,72)`, `[72,96)`, 24 workers
each on the same four nodes, waiting for frozen means. Aggregate `t166450`
waits for their exact completion markers. No V87 outcome is claimed at dispatch.
