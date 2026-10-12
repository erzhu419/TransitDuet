# Stage194: Reference Risk and Action Credit

Stage193 improved absolute fleet MSE without improving selection. Test credit
separation before spending native rollouts on a larger target cohort.

Reuse the same600 states,30 mean8 targets and17 grouped folds. Freeze Stage192
nonfleet/wait heads. A two-by-two contrast tests shared versus separated credit
and absolute-RMS versus pair-RMS loss units. Reuse Stage193's shared/absolute
control; fit the other three versions. Shared uses98-16-1; separated uses two
98-8-1 networks, with no unidentifiable contrast-output bias. Both have1601
parameters and start from the same16 hidden rows and zero predictions.

Separated risk predicts the original action's future fleet cost. Contrast
predicts candidate-minus-original effects. Pair loss detaches reference risk;
there are no shared parameters between risk and contrast. Clamp the resulting
future costs at zero above the historical floor. All output coordinates use
the same training-only absolute RMS; pair-scaled losses alone use training-only
15-pair RMS. Original-only anchor supervision is part of the separation change.

Fit100 AdamW steps per version, seed178, unchanged inputs and groups. Report
every root/split, all factorial effects, absolute and pair MSE, selection bias
and cost/wait. Cached-control predictions must reproduce Stage193 exactly.
Primary remains separated/pair-scaled with negative cost and nonpositive wait
in both roots across all three mean8 splits. Budget:51 new fits/5100 updates,
34 reused fits, zero native episodes/actor updates. Nodes001-006 unbound;
one CPU/3GB per task. Large artifacts stay remote.

## Limitations and Next Step
Only30 targets are conditional means; the same ten development prefixes are
reused. This does not establish repeated learned-upper control or open reserved
labels. Use the factorial result to choose a fixed training specification, then
broaden conditional target coverage; do not equate a better MSE with authority.
