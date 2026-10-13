# Stage210: future noise versus prefix transfer

Use the existing600 inputs/240 mean8 states/47 grouped folds. Split eight
future draws first/last and even/odd;fit in both directions. Each pair fits
a new symmetric GLM on its outer training groups and training-half labels,
then the same3393-parameter neural residual. All scales are training-only.
Half targets have count4;the360 singleton targets retain count1. Report
steps0/50/100/200/400/600 without selecting on diagnostic results.

Compare training-prefix/train-future,training-prefix/other-future,and
held-prefix/both-futures. Scene training reference matches root/regime.
Neural-minus-own-GLM improvements on same-prefix versus held-prefix other
future distinguish transferable prefix effects from sample memorization;
also report stable errors and half-sign disagreement. Primary step600.

Run:`native_transit_future_transfer_stage210_diagnostic_20261013_r1`.
One prepare,47 fit tasks,one report;188 GLMs/188 neural paths/112800
AdamW steps,no cost fits,actors or native episodes. Unbound node001-node006,
1CPU/3072MB;half targets/state diagnostics stay remote,small report only.

Limits: Four-draw targets/reliability weights differ from Stage208/209;
within-run comparisons use the same fit/cohort. Same-prefix tests share
physical/latent conditions,not new seeds. Lower-root changes also change
demand/prefix seeds. Descriptive diagnosis,not CI or native reward proof.
