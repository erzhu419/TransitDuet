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
One successful end-to-end preflight before the remaining31. Scheduler
node001-node006, one core/task, unbound. Pull only receipts and compact report.

Report both banks and all four views: cost/wait/service/fleet contrasts, selected
predictor errors, admission and key histograms. Full model accuracy is unchanged;
selected-predictor error changes reflect different keys/mixtures.

Limitations: reused Stage249 development futures/prefixes, not fresh confirmation
or seed CI. Old versus new2 compares future realizations as well as vintage; it
does not by itself identify a changed distribution. Stage248 and Stage169 reserves
stay closed. No default/deployment change or post-hoc candidate promotion.
