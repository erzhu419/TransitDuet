# Stage193: Fleet Physics Prior and Residual

Stage191/192 locate the dominant continuation noise in fleet peak cost; a small
mean-target overlay alone did not pass the cost/wait screen. Test an explicit
physical prediction path before increasing encoder capacity or collecting seeds.

Reuse the17 scalar mean-target Stage192 critics. Freeze their nonfleet and wait
predictions across four variants: cached absolute fleet, causal fleet prior,
fleet learned from zero, and prior plus learned residual. The prior is the
existing duration-ensemble full candidate fleet cost, not a realized peak.
Clamp future fleet predictions at zero above the irreversible observed floor.

Both new fleet heads use identical98-feature inputs,98-16-1 networks,1601
parameters, initialization, training-only scales and100 AdamW updates. Their
output projection starts at zero; the only difference is the explicit prior.
Loss is absolute future-fleet MSE plus pair-fleet MSE. Reusing the controls
adds their earlier100 updates to both composite models; no equal-total-budget
claim is made against the original shared three-head critic.

Keep all600 states/100 scenes and the same30 mean8 targets. Use the same ten
scene, five regime and two lower-root folds, holding all callbacks out together.
Evaluate the same30 states on both original and mean8 labels. Report all root/
split cells, residual-minus-control differences, fleet MSE and selected bias.
Primary gate remains negative cost and nonpositive wait for both roots in all
three mean8 splits. Exact ties keep the original action.

Budget:34 new fleet fits/3400 updates,17 cached controls/1700 earlier updates,
zero new native episodes or actor updates. Scheduler nodes001-006 unbound,
one CPU/3GB per task. Full banks, labels and models stay remote.

## Limitations
This is a ten-prefix development screen;570 training states still have noisy
single-outcome labels. A physical prior win alone is not learned authority.
Passing this screen does not open reserved labels or establish repeated upper
control, promotion or end-to-end HRL.
