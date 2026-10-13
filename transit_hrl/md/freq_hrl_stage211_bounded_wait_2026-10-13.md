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

Limits: Same development prefixes/sampled single-intervention outcomes;
not independent CI or full learned HRL. No promotion before the full gate.
