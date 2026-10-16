# Stage257: Frozen Old-Key Response Path

Fixed old keys across validation, opposite futures, full seen and full held.
Task t165497: node006, exit 0, no retries; five tests passed. All 32 cached cells
reproduced means within 3.47e-18. Pulled only the 3,847,518-byte scalar report;
no fits, inference, simulation or policy changes.

Mixed new8 / pooled-safe removed-category conditional wait means (minutes,
relative to full raw), 12 bank-cell exposures per family:

| Family | Inner Validation | Inner Opposite16 | Full Seen | Full Held |
|---|---:|---:|---:|---:|
| free_old | +0.000177691 | +0.000312413 | +0.000372743 | -0.001424306 |
| shrunk | +0.000776389 | +0.000334722 | +0.000563368 | -0.001050434 |

Both opposite halves retain positive group waiting harm. Only 3/8 point-harm
exposures repeat the same violation in both halves; none clears all original
violations in both. Four no-strict-gain exposures change no inner/full-seen
actions; held cost is +0.000042026 with almost zero wait effect. Looser strict-gain
filtering is not supported as the remedy.

Limitations: selection-conditioned development strata, not intrinsic domain-shift
or causal proof. Half signs are not CI. Inner/full changes both training future
count and prefixes; seen/held is unpaired. Stage253/248/169 remain closed.

Next: freeze keys/full models, compare cached inner8 with prefix-excluded inner16
in the same full basis/prior/cap, on common opposite16 futures. This isolates
training count without new selection, raw/full fits or episodes. Stage258 not submitted.
