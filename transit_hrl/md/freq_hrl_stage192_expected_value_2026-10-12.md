# Stage192: Expected-Continuation Targets

Stage191 found unstable single-outcome credit: half-choice agreement10/30,
rank concordance56.6%, and the single-label oracle changes cost from -0.004463
to +0.009249 on new innovations. Prioritize expected-value targets before
changing representation or relaxing the gate.

Keep all600 input states/100 scene groups from Stage188. Only replace the30
Stage191 target records with their eight-draw physical-cost means. Average
quadratic fleet costs, not peaks before squaring. All other targets remain
their original single outcomes. Causal inputs and identity differences stay
unchanged. Target normalization is recomputed within each training fold.

Use ten leave-target-scene, five regime and two lower-root folds, holding all
six callbacks of every test scene out together. Evaluate both representations
on the same30 original and expected target states. Reuse the34 matching frozen
Stage188 fits as controls; train only34 replacement-target fits with unchanged
100-step proper MSE, seed178, architecture and optimizer. Both selectors share
each model; action choices cannot depend on evaluation-label kind.

Preparation also decomposes cost-gap innovation variance into fleet/nonfleet
variance and twice covariance. Full mean-label maps, models and predictions
stay remote. Scheduler nodes001-006 are unbound, one CPU/3GB per task.
Budget:34 new fits/3,400 updates,34 reused fits, no new native episodes or actor
updates. This changes targets and their training-only normalization, not inputs.

## Limitations
Only5% of training states receive mean targets; these eight-draw estimates still
have sampling noise. The30 expected test states cover ten development prefixes,
not all600 states or new scene confirmation. No reserved-label/deployment gate
opens even if this small expected-value screen improves.

Run `native_transit_expected_value_stage192_fit_20261012_r1`, source `e737a0d540`.
Prepare `t145141` and fits `t145157-t145190` all completed exit0. Six focused
tests pass. Preparation decomposes cost-gap variance as fleet0.006122,
nonfleet0.000222, twice covariance-0.000333, total0.006011. Thus fleet-peak
variation, rather than only wait/nonfleet noise, dominates this diagnostic.

## Result and Next Step
Report `t145226` qualified34 new and34 reused fits,3,400 new updates, zero
native episodes/actor updates. All expected cost/wait joint gates still fail.
Scalar scene/root401 wait changes from +0.017925 to +0.001000 and cost from
-0.004492 to -0.005901, but regime/root401 still worsens both metrics.
Sequence scene/root401 instead changes cost from -0.001712 to +0.002731 and
wait from +0.013825 to +0.018350. Summed fit time is34s scalar versus699s
sequence. Do not select the favorable scalar cell as general superiority.

The experiment only denoises5% of training targets; it does not reject full
expected-value training. Next prioritize a physics-prior residual/explicit
fleet-risk critic and adequate conditional target coverage, rather than
additional encoder capacity, gate weakening or another global seed sweep.
One3,648-byte prepare receipt and one159,912-byte report were manually pulled.
All full labels, features, models and prediction arrays remain server-only.
