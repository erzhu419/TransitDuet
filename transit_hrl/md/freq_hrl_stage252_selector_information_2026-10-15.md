# Stage252: OOF Validation Information With Fixed Full Models

Stage251 key swaps attribute most pooled regression to changed selected keys.
Test validation information rather than increasing the head penalty again.

| Selector | Validation Futures Per Inner Block |
|---|---|
| old_A2 | Same old A2 labels and unchanged OOF rule |
| matched2 | First 2 futures of the other half of the training bank |
| expanded8 | All 8 futures of that same half |

Cross these selectors with the frozen free_old and shrunk models. Primary:
expanded8-minus-matched2 separately within each family. Secondary: new2-minus-old2
and new8-minus-old2. No model/selector winner or scenario-specific gate is selected.

Validation prefix is excluded from each inner fit; new validation draw indices
also exclude its 8 training futures and the opposite16 evaluation bank. New2 is
nested in new8, and the two validation halves are disjoint. Keep the 17-candidate
rank/strength grid, score, nonharm and half-safe rules, scales and raw references.

Read Stage251 full models verbatim. Its inner coefficients were not persisted:
reconstruct only those same 8-future inner predictions, with unchanged free and
shrinkage formulas, and require old_A2 candidate summaries and keys to reproduce
Stage250/251. Persist OOF candidate actions server-side for future selector work.
Freeze chosen full-model actions before accessing the opposite16 outcomes.

Budget: 32 reconstruction/selection tasks + one report, 10,240 small inner solves,
zero full/raw-reference fits, native episodes, actor updates or new prefixes.
One successful end-to-end preflight before the remaining 31. Scheduler
node001-node006, one core/task, unbound. Pull only receipts and compact report.

Report both banks and all four views: cost/wait/service/fleet contrasts, selected
predictor errors, admission and key histograms. Full model accuracy is unchanged;
selected-predictor error changes reflect different keys/mixtures.

## Results

All 33 tasks completed with exit code 0 and no retries: t162384 preflight,
t162390-t162420 remaining cells, t162421 report. All 69,120 cached action exposures
reproduced; old_A2 candidate summary maximum absolute difference was zero.
Only the 3,299-byte preflight receipt and 1,732,956-byte report were pulled.

Mixed-pooled held-prefix means relative to raw; negative is better:

| Fixed Model | Selector | Cost | Wait / min |
|---|---|---:|---:|
| free_old | old_A2 | -0.000126306 | -0.000281217 |
| free_old | matched2 | -0.000013673 | +0.000057357 |
| free_old | expanded8 | -0.000336602 | -0.000546061 |
| shrunk | old_A2 | -0.000065757 | +0.000015951 |
| shrunk | matched2 | -0.000013821 | +0.000057422 |
| shrunk | expanded8 | -0.000278643 | -0.000405599 |

Expanded8 improves cost and wait versus matched2 in both banks for both model
families. Merely replacing old_A2 with matched2 does not help pooled outcomes.
This is evidence that OOF validation information affects selection, not evidence
that full-model fleet expectations became accurate: those models are unchanged,
and free_old selected-predictor fleet MSE rises from 0.000955252 to 0.000989431.

## Limitations

Free expanded8 worsens cost and wait versus old_A2 in bank0, despite improving
both in bank1. Seen-prefix wait worsens versus matched2; dedicated root401 also
worsens on both metrics. Shrunk expanded8 harms both metrics in mixed_b.
Free expanded8 is not a joint improvement over Stage247: cost is lower but wait
is worse (-0.000546061 versus -0.000766732 min). Its pooled cost gain versus raw
is 79.8% fleet cost, and changed-action rate is only 1.875%.

These are reused Stage249 development futures/prefixes, not fresh confirmation
or seed CI. Old versus matched2 also changes future realizations, not just vintage.
The two mixtures are Transit views, not two domains. Stage248 and Stage169
reserves stayed closed; no default, deployment or candidate-winner change.

## Next

Freeze all six compared protocols and the Stage247 control before new paired
futures. Confirm expanded8-minus-matched2 separately within each family, report
both banks and prefix-level fleet events, and retain seen-prefix wait and root401
failures. Diagnose OOF-to-full selection transfer separately; do not tune another
head penalty from these outcomes. No next-stage jobs were launched in Stage252.
