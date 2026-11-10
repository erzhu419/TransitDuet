# Decision after V402 (2026-11-10)

Recommendation: one bounded redesign of budget decisions, not another retention
sweep, binary-upper training run, or immediate superiority-paper freeze.

## What the existing evidence says

V402's independent delay benefit is real, but both CV2 protections fail. The
pressure policy has lower CV2 on only 2/16 static comparisons and 1/16 cap
comparisons. More confirmation days alone do not address this directional harm.

Read-only server task t214166 additionally examined the four OLD calibration
days. At pressure's mean CV2 (.077787), the lowest-delay convex combinations
of the recorded simple settings have 4,115.93 s/day (static) and 3,585.63 s/day
(cap) more delay. Pressure also has lower development journey cost. The geometry
therefore supports a development opportunity, not an independently proven frontier.

These CV2-matched mixtures use the same component pairs as V402; their high-setting
probabilities differ from the original holding-matched probabilities by only
.00314 and .01129. Changing the matching axis is not a remedy for the held-out
CV2 loss. No new comparator has been sealed or independently evaluated here.

The budget formula uses estimated normal holding and supply pressure, with no
regularity term (`base_law.py:195`). Normal holding itself comes from this
policy's completed-trip command history, not an external compensation requirement.
The fast controller's feasibility reference is ALREADY budget-projected
(`../v314_dynamic_upper/attempt03/coupled_model.py:106`). Its CV2 check protects
current-action selection relative to that reference, not the budget restriction
itself or a whole-day simple policy. This is a design boundary, not a new wiring bug.
Saved pressure probes changed selected actions in 831/896 supported own-budget
removals; constraints are active, but these probes do not label harmful actions.

## One proposed development stage

Treat new credit issuance/tightening as a control decision, evaluated jointly
with its legal holding continuation against a common no-new-tightening reference.
Preserve useful regularity compensation before accepting resource savings. Keep
fleet, legal bins, completed-prefix inputs, forecasts, costs and runtime fixed.
Unsupported evidence admits no NEW tightening; existing issued/spent credit and
irreversible commands remain accounted for. No fictitious slack or quota reset.
An already-imposed restriction cannot be retrospectively claimed to be harmless.

This needs a frozen design and engineering checks before traffic, then one
four-new-day development screen: the candidate, frozen pressure, and the four
already selected simple components (24 physical paths, plus separately sealed
minimal engineering paths). The mixtures stay fixed. Compare delay with CV2
protection against both simple policies; report passenger cost and the tradeoff
against old pressure. Do not require dominance of every old operating point.
Only a useful screen justifies fresh confirmation; no automatic training expansion.

If this still only exchanges regularity for turnover, close this method branch.
The paper can retain the operational tradeoff and V107/V400 evidence, but cannot
claim proven pressure-allocation, learned-upper or full frequency-separation
superiority. Its method must describe the actual slow rule and model fast layer.

## Evidence And Limits

The added calculation uses existing scalar summaries only; it starts no traffic
or optimization and changes no frozen gate, parameter or main policy. Calibration
has four days, and its frontier is exploratory. The forecast-level explanation
above has not quantified its contribution to actual CV2 loss. Data:
`review/decision_review_short/results.json`. Task t214161 failed before remote
execution on inline-command length; the short-entrypoint t214166 completed with
zero physical starts. V402's 147-path accounting is unchanged.
