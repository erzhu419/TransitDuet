# Stage249: independent training-future information

Stage248 replicated service/wait benefit, not total-cost superiority. Preserve it
as confirmation evidence; no Stage248 outcome is an input to this experiment.

- Freeze the Stage247 representation, six actions, raw references, TSVD ranks,
  A-scale/RMS caps, cached head difference, state-GLS loss and OOF selection rule.
- Collect 32 independent futures for every existing state and all six actions.
  Seeds: 1490000000 + Stage195 state index * 32 + draw. No reserved prefix opens.
- Crossfit banks 0-15 and 16-31. Matched-count arm uses 2 inner/4 full training
  futures; expanded arm uses 8 inner/16 full. Matched samples are nested in expanded.
- Keep the old Stage247 state-GLS model as a frozen third arm. Compare replacement
  versus old, expansion versus matched. Do not select between these arms.
- Select rank/strength using the unchanged old A2 OOF validation labels only.
  Diagnose each fit on the opposite 16-future bank, never its training futures.
- Compare train60 and held60 target-mean errors, predictor service/fleet/total MSE,
  feasible-pair sign errors, action coverage and cost/wait/service/fleet outcomes.
- Primary contrast: expanded pooled-safe minus matched pooled-safe. Both bank
  directions and four views stay visible; no root/scenario-specific gating.
- This isolates future information, not independent-head representation. Equal
  service/risk sum updates remain unchanged for a later separate hypothesis.

Scheduler node001-node006, one core/task, 1.5 GiB native task RAM, no node binding.
Ten one-draw preflights -> preflight report -> 1,920 four-draw chunks -> panel
preparation -> one fit preflight -> remaining31 fits -> report. Preflight draws
are not recollected. Budget: 7,680 draws, 61,440 episodes, zero raw-reference fits.
Only small receipts/aggregates are pulled; panels, models and rollouts stay remote.

Next: freeze all three arms, then pre-register genuinely new prefixes for transfer.
Current crossfit is conditional on existing development prefixes, not a seed-level
performance CI or proof of full learned HRL. No default changes are planned here.

Preflight t160200-t160209 and summary t160218 completed with exit zero: 80 qualified
episodes, maximum draw time 221 s, peak RSS 734 MB, zero training updates. Seven new
tests and 11 existing regressions passed. Only the 2,835-byte summary was pulled.
Remaining collection is t160223-t162142 (1,920 chunks, 7,670 draws excluding the
ten preflights). Fits were pending at that submission snapshot.

## Completed Fitting And Analysis

Preparation t162192, fit preflight t162199, remaining 31 fits t162205-t162235,
report t162239 and compact analysis t162245 completed with exit zero. Including
collection: 1,966/1,966 records done, no failed/cancelled attempts. All 23,040 frozen
raw/old action exposures reproduce exactly. This continuation added no native
episodes, raw-reference fits or actor updates. Only scalar summaries were pulled.

Mixed-pooled nonanchor fleet-cost-difference MSE (lower is better):

| Quantity | Matched4 futures | Expanded16 futures |
|---|---:|---:|
| Sample mean versus opposite16 mean | 0.002056715 | 0.000734343 |
| Selected model, seen prefixes / new futures | 0.000743993 | 0.000741258 |
| Selected model, held prefixes / new futures | 0.000944178 | 0.000942938 |
| Full-strength rank16, seen prefixes / new futures | 0.000747392 | 0.000739611 |
| Full-strength rank16, held prefixes / new futures | 0.001022890 | 0.001093041 |

Sample-mean error falls 64.30%, but selected-model error falls only 0.37% on seen
prefixes and 0.13% on held prefixes. Estimated opposite16 MC noise floor is
0.000436357; subtracting that common floor gives an 81.61% reduction in the target
mean-error estimate, not a confidence interval. Training-bank model MSE drops
0.002095303 -> 0.000742011 on seen prefixes, but these are different-precision
targets; the new-future rows show why that drop alone is not learning progress.

The held-prefix zero-difference baseline has MSE 0.000877426: the expanded selected
model is 7.47% worse on this all-nonanchor metric. On feasible-pair MSE it is 10.44%
better than zero, so there is useful local signal; expansion itself improves that
metric only0.054%. Do not equate sparse zero labels with complete risk learning.

Full-strength rank16 improves seen-prefix fleet MSE 1.04% but worsens held-prefix
MSE 6.86%; ranks 1/3 change very little. This is not solely OOF rejection hiding a
successful critic. At every fixed rank, service-minus-risk prediction is frozen:
the inferred difference MSE, 2*(service MSE + fleet MSE) - total MSE, changes by
at most 6.51e-19 between arms across all reported views/banks/scopes. This confirms
the parameterization restriction, not that independent heads will necessarily win.
Post-hoc full-fit ledger inspection also finds active old A-RMS caps in 31/32
expanded rank8 fits and 26/32 rank16 fits; max KKT error remains below 2e-12. Simply
removing caps is not justified when higher-rank held-prefix error already worsens.

Held-prefix policy means versus frozen raw:

| Policy | Total cost delta | Wait delta (min) |
|---|---:|---:|
| Frozen Stage247 | -0.000227372 | -0.000766732 |
| Matched4 | +0.000005808 | +0.000154557 |
| Expanded16 | -0.000064318 | +0.000030924 |

Expansion improves both pooled means versus matched, but not versus frozen old,
and wait-versus-matched has opposite signs across the two banks. Expanded wait
versus raw worsens in both mixed-pooled banks. Admission falls from 16/32 old to
14/32 matched and 12/32 expanded model-bank exposures. Dedicated root401 has a
better pooled point result after expansion, but bank1 still worsens wait.

Next decision: a separately registered fixed-data contrast of head-difference
freezing and correction-budget choice, followed by genuinely new-prefix evaluation.
Do not just add seeds, relax the cap, or choose rank from these diagnostic outcomes.

Saved under `results/native_transit_training_information_stage249_20261015_r1/`:
`analysis/result.json`, `completion_receipt.json`, `diagnostic_addendum.json`,
`constraint_summary.json` and `findings.json`. The compact analysis is 640,256 bytes;
panels, model files and rollouts remain on the server. Ten focused tests passed.

## Limitations

The target is an action-relative expected peak-fleet penalty, not peak vehicle
count. Observed MSE includes finite evaluation noise. These are reciprocal-bank
crossfit means on existing development prefixes, not independent training-seed CI
or new-prefix confirmation; the existing Stage248 claim boundary is unchanged.
