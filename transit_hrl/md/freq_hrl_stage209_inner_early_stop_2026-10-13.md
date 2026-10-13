# Stage209: training-scene checkpoint selection

Keep the Stage208 state-conditioned head, loss, optimizer, six-plan menu,
outer costs/GLM, 600 inputs, 240 mean8 states and 47 outer folds unchanged.
Four inner scene folds use seed200, stratified by root/regime/mean8 status.
Rebuild each inner GLM and all scales on inner training only. Score only
inner-held mean8 states at steps0/50/100/200/400/600. Minimize pooled
magnitude-weighted crossentropy; exact ties choose the earlier checkpoint.
Step0 is the unchanged GLM. Refit on all outer training at the chosen step.

Run: `native_transit_early_stop_wait_stage209_development_20261013_r1`.
188 inner GLMs,188 neural paths,47 outer model assemblies,at most141000
AdamW steps. Report actual steps and zero-step fallbacks. One CPU/3072MB
per task,unbound node001-node006;models/labels stay remote. Compare frozen
GLM,fixed600 and early stop on outer cost/wait,train/held diagnostics and
both future-half partitions. Development gate:negative cost/nonpositive
wait for both roots/all three splits,and lower held excess loss than
fixed600 in all splits. Outer results never select checkpoints.

Completed47 fits`t148191-t148237` and report`t148250`,all exit0;
source`dfb4ca2bb4`;six tests passed. Only458096-byte report pulled.
46 folds select step0;scene/root401 fold77 selects50. Actual112850 AdamW
steps,188 inner GLMs/6442 L-BFGS iterations,only one optimized outer head.
Cost passes,wait fails;both half partitions fail wait. No promotion.
Held excess nats,fixed600 ->early stop:scene0.513596 ->0.291976,
regime0.634518 ->0.331915,lower-root0.585128 ->0.278310. Scene GLM is
still better at0.280190. Failed waits remain regime/root397 +0.000179167
and lower-root/root401 +0.002670833min;frozen controls reproduce exactly.

Next: refit GLM/neural heads using one future half's labels only;evaluate
the other half on the same prefixes versus held-out prefixes,then reverse.
This separates future-target noise from prefix memorization before further
capacity changes;the reference GLM must not see the evaluation-half labels.

Limits: Existing prefixes and noisy eight-draw targets;supervised
single-intervention development,not independent CI or full learned HRL.
