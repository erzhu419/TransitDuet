# V268: complete-day constrained lower-policy learning

Close the anchor, cumulative-H budget and single-bin pulse branches. Keep V107
as a verified reference,not a frequency-learning claim. The new intervention is
the lower learning objective and optimizer;upper exact360,fleet12,seven bins,
physical simulator,existing point predictors and main stay unchanged.

Minimize complete-day once-only waiting plus onboard person-seconds. Separately
constrain mean actual arrival CV2,mean readiness delay,total positive actual-minus-
original-public launch delay,and unlaunched/incomplete trips to the matched
fixed+bilateral025 reference. No shaped reward,discounted local reward,scalar
penalty sweep,TD critic or predicted single-action benefit enters selection.
Full-day execution includes downstream holding,returns and later dispatches.

Learn a deterministic residual of bilateral025:clip its continuous request plus
45*tanh(theta dot x) to0--45,then use the existing legal seven-bin nearest action
(shorter ties). Six signed coefficients in[-1,1] use current bias,APC load,route
progress,ready stock and overdue plans at the return terminal,and supported gap
imbalance. Missing forward evidence retains zero;missing follower uses the old
forward fallback. No future identity,return label,latent queue or H-as-slack.
The policy can increase some holding and reduce others;there is no global cut.

Compare to a separately optimized uniform residual with only the bias coefficient.
Both get the same four consumed V267 days2071001--2071004,the same8-vector
initialization/one-generation SciPy differential evolution budget(max16 vectors),
and the same full-day objectives/constraints. Seed268,polish off,no restarts or
hyperparameter search. Cache repeated parameter tuples;the zero vector reuses
V267's four fixed-bilateral references,after exact-path preflight. Fixed policies
use runtime seed12011,not four artificially independent actor repetitions.

Unlike V59,there is no noisy cross-vehicle TD learning. Unlike V75/V104,there is
no fitted single-action value or label-selected pulse. Unlike V78,feasibility is
measured from full nonlinear closed-loop paths,not a directional gradient/KL
approximation. This is a bounded low-capacity policy-learning pilot,not proof
that optimization has converged or the original learned hierarchy is repaired.

Budget:three engineering paths(one exact cached zero-policy replay and two
nonzero execution probes),then at most60 new paths per family,total123 starts.
Failures consume slots;no retries,replacements or extensions. Eight simultaneous
rollouts maximum(two family jobs with four day workers each). Preflight checks
old actions,timing,endpoints and physical contracts;its outcomes do not gate
efficacy. No independent confirmation collection in this stage.

Freeze each learned vector after the fixed budget. Report every full-day vector,
constraint tradeoff and same-state increase/decrease count. Only a conditional
policy reducing passenger cost with all five costs nonincreasing on EACH of the
four development days,also jointly improving over the trained uniform control,
merits a separately sealed new-day confirmation. This is a development stopping
rule,not an inferential test. Failure ends this bounded policy family without a
rescue sweep. No main switch or frequency claim. Only after independent service
benefit should matched-budget frequency ablations be considered.

All tests,preflight,optimization and aggregation run via scheduler high/manual
on node001--node006. Raw paths and fitted parameters remain server-side;only
small receipts and final summaries return locally. Original TransitDuet untouched.

Run (2026-10-25):synthetic task t211590 exited0(8 tests). Frozen development
source c00a29013a;high/manual t211591 launched on node004 with8 CPUs,48GiB RAM,
no auto-requeue. Final-source8 tests also pass. At the launch snapshot,one of
three engineering paths has started(neutral cached replay),none is accepted yet.
Only passing all three releases the two four-worker searches and aggregation
within this same job. Confirmation remains closed;no efficacy result yet.
