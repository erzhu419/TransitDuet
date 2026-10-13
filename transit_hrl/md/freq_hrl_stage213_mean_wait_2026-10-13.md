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

Next: compare matched old/new heads and half transfers to determine
whether training-target noise explained the previous transfer failure.

Limits: reused development prefixes and supervised one-action effects;
not independent CI or complete learned upper/end-to-end HRL.
