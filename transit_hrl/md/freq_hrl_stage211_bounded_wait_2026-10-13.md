# Stage211: bounded and functionally-shrunk wait correction

Return to original600 inputs/240 mean8 targets/360 singletons/47 folds.
Same3393-parameter head,initialization/optimizer;residual=1*tanh(raw/1).
Paired variants use functional ridge0 or0.1 on training wait-mass-weighted
squared residuals. Fixed bound/penalties/threshold0.5,no parameter search.
Four inner scene folds rebuild their GLM/scales on inner training only;
select steps0/50/100/200/400/600 by held mean8 crossentropy,ties earliest.
Refit on outer training;original GLM,costs,menu and Stage208/209 controls
remain frozen. Primary:shrunk_bounded. Gate:negative cost/nonpositive wait
both roots/all three splits,and lower held excess than GLM in all splits.

Run:`native_transit_bounded_wait_stage211_development_20261013_r1`.
47 fits/one report;188 shared inner GLMs,376 inner neural paths,94 outer
model assemblies,at most282000 AdamW steps. Unbound node001-node006,
1CPU/3072MB;raw/models remote,small report only. No cost/actor/native training.

Completed fits`t148345-t148391`,report`t148413`,all48 exit0;
source`f7544c70bd`;six tests passed,one778378-byte report pull,no raw/models.
Actual227000 AdamW steps,188 shared GLMs/6442 L-BFGS iterations.
Nonzero refits:bounded_only4/47,shrunk_bounded13/47. Both fail wait and the
full development gate. Shrunk regime/root397 wait+0.000179167 ->-0.000157292
min;cross-lower/root401 remains+0.002670833. Shrunk held excess versus GLM:
scene0.300001 vs0.280190,regime0.339601 vs0.331915;lower-root unchanged.
Primary half cost/wait gates also fail;all three frozen controls reproduce.

Next: upgrade the360 singleton training targets to independent future
averages,keeping the240 evaluation states/folds and fixed bound/ridge.
Improve label reliability across existing development prefixes before
expanding model capacity;reserved scenes remain closed.

Limits: Same development prefixes/sampled single-intervention outcomes;
not independent CI or full learned HRL. No promotion before the full gate.
