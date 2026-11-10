# Frozen Calibration Score-Source Control

Reuse the eight fixed heads, saved OOF predictions, and original training A/B
effects. No fold reconstruction, fitting, new prefixes or native simulation.
Retain the original numeric quantile cut; do not compute a new full-score quantile.

For each head/root, independently select cuts on OOF scores and full-model scores
using the SAME A-only G objective, empirical W floor and tie rules as the previous
experiment. Reproduce the saved OOF choice. Freeze all four cuts/root before B.
Cross both calibration sources with both readout score views; primary contrast is
full-selected minus OOF-selected on the SAME frozen full-model scores. Also retain
both versus the unchanged quantile policy and actual C0, so baseline action changes
cannot masquerade as an improvement in that direct contrast.

Report A, B1/B2/all4, paired W/G, conditional MC SE, descriptive prefix intervals,
and action counts, including collapsed solutions. Four unpinned 1CPU/2GiB scheduler
jobs reuse server-private caches; only public aggregates return locally. No head
selection, B retuning, confirmation or automatic follow-up. PILOT_RECIPE_HOLD stays.

## Limitations

Full-score A calibration is in-sample with respect to the head. OOF cuts are not
nested policy validation. B is previously revealed development data on known
training prefixes. This diagnostic can isolate score-source mismatch, not establish
fresh-prefix, complete-episode, lifecycle-budget or end-to-end HRL improvement.
