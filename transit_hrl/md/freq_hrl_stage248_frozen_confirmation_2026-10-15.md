# Stage248: frozen-policy conditional-future confirmation

Stage247 state-balanced GLS improved mixed held-prefix cost/wait on eight reused
development futures, but mixed_b and dedicated single-root models still harmed.
The next experiment freezes actions rather than retuning against those outcomes.

## Protocol

- Compare frozen full-model held60 raw, all-MSE, pair-GLS and state-GLS actions.
- Preserve all 32 cells, four views and the existing 240 states/40 prefixes.
- Seal divergent state/action unions before any new outcome; same-action contrasts
  stay exactly zero in the original full-cohort denominators.
- Use 64 new independent future seeds/state, split 32+32, CRN across actions.
- Seed = 1480000000 + Stage195 state index * 64 + draw; no adjacent-probe overlap.
- Reproduce the original source, then resampled observer and original identity;
  replay only the sealed candidate union, with original-greedy continuation.
- Require unchanged prefix/RNG/context, frozen weights, physical clock and ledger.
- No raw/reference fitting, critic updates, action inference or reselection.
- Primary: mixed-pooled cost and wait versus raw, conditional normal MC upper 95%
  bounds below zero with negative means in both 32-draw halves.
- Paired comparisons with all-MSE/pair-GLS and root/view/regime results are secondary.

## Execution

Scheduler only, node001-node006, one core/task, no node binding.
Seal -> ten native preflight cells -> qualified preflight report -> remaining
draws -> final report. Raw labels/rollouts remain server-only; sync small receipts.

Seal t155893 completed with maximum Stage247 reproduction error 2.17e-19.
There are 66 active states, 4,224 draw tasks and 22,528 native episodes;
100,352 exhaustive-replay episodes are avoided. The 17 fixture tests passed.
Native preflight t155896-t155905 and its report t155914 completed with exit zero:
51 qualified episodes, maximum task time 166 s, maximum process RSS 691 MB,
zero training updates. Remaining 4,214 draws are t155915-t160128.
Bulk registration completed normally. The submitter now skips a redundant quadratic
identity scan after its exact preceding-phase gate; failed or already-expanded
phases still block submission. No scheduler code or research protocol changed.

## Completed result, 2026-10-15

All 4,224 draws and final report t160164 completed with exit zero. Including seal
and preflight report: 4,227/4,227 done, no failed/cancelled attempts, 22,528 native
episodes, zero training updates. Final coverage is qualified. The analysis pulled
only 602,532 bytes of final aggregate and 11,608 bytes of state attribution;
no raw labels, rollouts, CSVs or checkpoints. The original submission snapshot is
preserved; `completion_receipt.json` records final status.

**Wait confirms; joint cost/wait confirmation fails.** Mixed primary versus raw:

| Component | Stage247 reused fresh8 | Stage248 independent fresh64 | Stage248 conditional MC 95% interval |
|---|---:|---:|---:|
| Total cost | -0.000227328 | -0.000052109 | [-0.000131529, +0.000027310] |
| Fleet cost | -0.000162760 | +0.000013563 | [-0.000061852, +0.000088978] |
| Nonfleet cost | -0.000064568 | -0.000065673 | [-0.000083433, -0.000047912] |
| Wait (min) | -0.000584896 | -0.000570573 | [-0.000732027, -0.000409119] |

Negative is better. Both 32-draw halves improve nonfleet cost and wait; total cost
is -0.000132757 in half0 but +0.000028539 in half1. Wait retains 97.55% of its old
gain and nonfleet retains 101.71%; total cost retains 22.92%. Fleet's deterioration
accounts for all the total-cost regression, with slightly better nonfleet outcomes.
The mixed primary has 32 fleet-event state/draws: 15 beneficial and 17 harmful,
versus only three sampled events supporting the old estimate. This supports a
replicated service effect and unresolved fleet expectation, not universal failure.

State-GLS also improves wait relative to frozen all-MSE by -0.000452539 min
(95% [-0.000581234, -0.000323844]) and pair-GLS by -0.000247689 min
([-0.000383479, -0.000111899]). Neither paired total-cost interval excludes zero.

## Where the effect holds and fails

- Persistent shift accounts for about 72% of pooled net wait and nonfleet benefit.
  One prefix, root401/scene914010303 (probes8 and23), contributes 70.89% of pooled
  net wait gain and 71.58% of nonfleet gain. Its cost contribution is -0.000067351;
  all other states together contribute +0.000015242 under the original weights.
  This is descriptive concentration, not a new leave-prefix-out model experiment.
- OOD period worsens nonfleet cost by +0.000017449 (95% [+0.000008658,
  +0.000026241]) and wait by +0.000178792 min ([+0.000120791, +0.000236793]);
  both halves have the same adverse signs. Fleet sampling alone cannot explain this.
- Mixed-trained models improve wait on both roots. The dedicated root401 view
  instead worsens wait by +0.000564551 min and nonfleet cost by +0.000058606,
  with adverse signs in both halves and both 95% intervals above zero. These are
  different training populations, not contradictory statements about root401.
- Only 19/960 mixed model-state decisions differ from raw, spanning18/240 states.
  Pooled wait gain is 0.0342 seconds; conditional on changed model-state decisions,
  it is 1.73 seconds. Statistical replication does not establish a large overall effect.

## Diagnosis and next decision

The inspected sparse denominators, shared-state MC variance, paired GLS covariance,
head units and fixed-clock CRN are correct. No implementation defect was identified
that justifies rerunning Stage248. Stage245-247 still estimated new residual targets
and uncertainty from only2 inner/4 full-fit B futures; they changed objectives and
weights without adding independent training outcomes. `pooled_safe` selects OOF
point estimates, not a confidence-bound guarantee. The reused fresh8 was not fed
directly into selection, but repeated development against it is not independent replication.

Earlier short-horizon results should be retained: Stage167/168 found real service
improvements obscured by scoring mostly rewritten timetable tails and by full-day
peak-fleet cost. Stage240/242 separately demonstrated reference-refit drift and
held-prefix transfer failure. Those findings explain why repeated local improvements
have not accumulated into general performance; repaired encoders need not be reopened.

The next informative comparison should hold the current architecture, selector and
action space fixed and change independent training-future information, particularly
the noisy fleet expectation, then evaluate frozen policies on genuinely new prefixes.
Stage248 remains a confirmation set, not a source for scenario gates or another
loss-weight search. Existing service/risk heads are not independent in the current
sum-only correction: their difference is frozen and corrections are equal. Changing
that parameterization is a separate hypothesis after the information test, not an
established fix. No next experiment or default change was made in this analysis.

Evidence: `results/native_transit_frozen_confirmation_stage248_20261015_r1/report/`
contains the official `result.json` and the server-reduced `state_attribution.json`.
Attribution sums reproduce all four official primary means within1e-12.

## Limitations

This confirms conditional-future replication on existing development prefixes,
not new training-seed CI, unobserved fleet-tail guarantees, or complete end-to-end
learned Freq-HRL. Raw is the learned reference selector; the native greedy policy
provides the common continuation after one forced action. Subgroup intervals are
descriptive, without multiplicity correction. Default and deployment eligibility remain unchanged.
