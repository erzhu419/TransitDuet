# Stage210: future noise versus prefix transfer

Same600 inputs/240 mean states/47 folds;first/last and even/odd,both directions.
Refit each GLM/scales using training groups and training-half labels only;
same3393-parameter residual,checkpoints0/50/100/200/400/600,no selection.
Count4 mean targets/count1 singletons;scene training comparisons match
root/regime. Unbound node001-node006,1CPU/3072MB;raw diagnostics stay remote.

Run:`native_transit_future_transfer_stage210_diagnostic_20261013_r1`.
Prepare`t148279`,fits`t148280-t148326`,report`t148327`,all49 exit0.
Source`29dd47e00b`;13 tests passed;one1060365-byte report pull,no raw/models.
188 GLMs/6655 L-BFGS iterations,188 neural paths/112800 AdamW steps;
no cost fitting,actors or native episodes.

First/last step50 improves same-prefix/new-future excess loss6/6;even/odd1/6.
New-prefix/new-future loss worsens12/12 comparisons at both50 and600.
Scene first/last A-to-B GLM/50/600 excess nats:same-prefix0.320046/0.298632/
0.398595;new-prefix0.337022/0.360501/0.661557. At600 same-prefix stable
false-safe counts469 ->225,new-prefix160 ->175. Half signs disagree386/1200
(first/last),381/1200 (even/odd) candidates. Prefix-specific sign structure
is learnable,but share confidence overfits and prefix transfer remains weak.

Next: bounded/functionally-shrunk GLM residual;grouped training-only validation
on original mean8 targets,unchanged native cost/wait gate. Uniform temperature
scaling preserves the0.5 boundary and cannot repair selected actions.

Limits: Count4 differs from prior count8;same-prefix futures share physical/
latent conditions,not new seeds. Lower-root also changes demand/prefix seeds.
Descriptive diagnosis,not CI or native reward proof;heads remain unpromoted.
