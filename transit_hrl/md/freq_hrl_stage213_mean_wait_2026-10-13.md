# Stage213: fixed-budget wait heads with mean8 training targets

Stage212 collection completed:2880 draws/23040 native episodes;no actor
updates or new prefixes. Prepare means on servers,retain original240
evaluation states and47 outer folds. Keep240 sample weights8 and360
weights1 despite actual draw_count8 everywhere.

Compare three frozen Stage211 controls with a newly fitted symmetric
GLM and fixed B1/ridge0-or0.1 bounded residual heads. Refit outer and
inner GLMs/scales on their own training only. Keep original four inner
scene partitions and original240 validation membership;newly averaged
states are training targets,not extra checkpoint-validation data.
Cost heads,menu,threshold0.5,3393 neural parameters,seed and optimizer
unchanged. Primary:mean8_shrunk. Gate:cost<0/wait<=0 for both roots/all
three splits,and lower held excess than the new GLM across all splits.
Original future-half outcomes remain separate diagnostics.

47 fits/one report:47 outer+188 inner GLMs,376 neural paths,94 outer
assemblies,at most282000 AdamW steps;no cost/actor/native training.
Unbound scheduler,node001-node006,1CPU/3072MB. Prepared labels/models
and predictions stay on servers;small prepare receipt/report only.

Run:`native_transit_mean_wait_stage213_development_20261013_r1`.
Five targeted tests passed:target/weight identity,outer-label exclusion,
inner-reference exclusion,frozen costs/controls,and zero-action gate.

Completed`t151407-t151453` plus report`t151462`,all48 done/exit0;
source`c2a8b50e5f`.235 GLMs,outer1593/inner6363 solver iterations,
231700 AdamW steps. Three old controls reproduce exactly. One840841-byte
report pulled;prepared targets/models remain remote.

New GLM passes cost/wait in scene and regime splits;cross-root401 wait
remains+0.002805208min versus old+0.002670833. Shrunk nonzero refits33/47
versus old13/47,but cross-root397 cost-0.002595199 ->+0.000902642.
New shrunk lower-root train excess0.144915 versus new GLM0.233646;
held excess0.306717 versus0.284118. Neural held excess also exceeds the
new GLM in scene/regime,and half cost/wait gates fail. Full gate fails.

Next: quantify erroneous safety decisions that the fixed residual bound
cannot repair,and separate frozen cost-ranking errors from wait errors.
Label precision alone did not close transfer;do not expand seeds/capacity
before this diagnosis. Cross-root also changes demand/prefix seeds,not
only the lower policy;the present contrast does not isolate policy shift.

Limits: reused development prefixes and supervised one-action effects;
not independent CI or complete learned upper/end-to-end HRL.
