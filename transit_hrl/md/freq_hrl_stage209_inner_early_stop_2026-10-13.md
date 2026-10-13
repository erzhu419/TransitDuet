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

Limits: Existing prefixes and noisy eight-draw targets;supervised
single-intervention development,not independent CI or full learned HRL.
