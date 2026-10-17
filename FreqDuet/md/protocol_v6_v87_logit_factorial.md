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

All 32 calibration and 384 evaluation runs completed exit 0/retry 0. Retrieved
only the 40,348-byte summary. Read-only reviews `t166482` / `t166487` pass eight
tests and reproduce the original contrast means. The order-symmetric changes
below use the forward 40-to-200 direction; smaller costs are better:

| Component | Journey min | CV-squared | Readiness delay s | Holding s/trip |
| --- | ---: | ---: | ---: | ---: |
| Total | +0.56091 | -0.005573 | +213.19 | +30.72 |
| Mean preference | -0.15344 | +0.004517 | -63.80 | -8.47 |
| Centered-logit response | +0.71435 | -0.010090 | +276.99 | +39.19 |

Response journey CI is [0.5380,0.9379], mean preference [-0.2106,-0.1026].
Forward interaction is -0.11799 min [-0.17393,-0.05893] and -58.48 s readiness;
it is already shared between the symmetric components, not an extra summand.
CV interaction is unresolved; journey interaction is much stronger for 12011
than 933013. Both actors show worse journey/readiness but better CV at 200.
Saved common-state expected actions decompose +1.54165 s/decision into policy
law +2.35213 [2.31586,2.38420] and visitation -0.81048 [-0.90954,-0.71437].
Logit mean is not physical mean holding. No corner solves the joint tradeoff;
do not tune another constant bias. V88 locates the common-state response by
observable context, preserving both native laws and the main policy.
