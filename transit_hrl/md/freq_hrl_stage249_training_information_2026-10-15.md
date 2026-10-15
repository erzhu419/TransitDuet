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
ten preflights). Fits await complete qualified panels; no new performance claim.
