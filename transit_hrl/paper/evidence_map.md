# Active Manuscript Evidence Map

Date: 2026-10-22. Main manuscript: [manuscript.md](manuscript.md).
Compact numerical source: [plan_validity_evidence.json](plan_validity_evidence.json).

## Central argument

Supervised renewal value can guide a fixed planner-call budget and improve
PointMaze closed-loop control. The independent-seed claim is confined to that
task family. Transit supplies separate positive local-control evidence and
a negative repeated-decision boundary, not same-algorithm domain transfer.

## Main evidence and comparators

| Manuscript location | Evidence | Unit and exact comparison | Interpretation |
| --- | --- | --- | --- |
| Method / Table1 | Stage9 independent confirmation | Eight fresh optimizer roots, each 16 held-out paths; shared controller, history trigger versus fixed/random-offset/current-only timing | All five frozen gates pass; central positive result |
| Results5.1 | Stage9 development | Eight development roots, distinct from confirmation | Supporting design result; not pooled with confirmation |
| Results5.2 | Stages88/89/95/100/129/131/135 | Their own teacher/root and frozen-upper contracts | Genuine learned-upper effects, not replications of the Stage9 trigger |
| Table2 | Stage298 and301 independent E32 | Two separate 400-prefix cohorts; paired single-intervention outcomes, fixed profiles averaged within prefixes | Raw-relative cost/wait gains; learned fleet increment is not established |
| Table3 / Results5.4 | Stage304 | 320 complete scenes, four paired controllers, one fixed profile, two upper decisions | Registered dual-win and statistical half-base retention fail; no extra joint cost benefit |
| Limitations | Stage10 learned termination | Revealed Stage9 confirmation paths; mostly deadline-only deployment | Failed development baseline, not competitive learned-termination superiority |

The current-only and history predictors have different feature maps and sizes
(170 versus 39). Timing controls share jitter-trained actor weights. These
comparisons identify schedule utility under that controller, not superiority
to all representation-matched or separately trained timing baselines.

## Executable and result sources

- Stage9 implementation: [pointmaze_budgeted_trigger.py](../freq_hrl/experiments/pointmaze_budgeted_trigger.py).
- Branch targets/query acquisition: [pointmaze_plan_validity_branching.py](../freq_hrl/experiments/pointmaze_plan_validity_branching.py).
- Predictor feature maps: [pointmaze_compact_plan_validity.py](../freq_hrl/experiments/pointmaze_compact_plan_validity.py).
- Frozen constants: [pointmaze_budgeted_trigger_stage9_spec.py](../scripts/pointmaze_budgeted_trigger_stage9_spec.py).
- Independent confirmation registration: [preregistration.json](../results/pointmaze_budgeted_trigger_stage9_v2_confirmation_20260926_r1/preregistration.json).
- Original confirmation analysis: [analyze_pointmaze_budgeted_trigger_stage9.py](../scripts/analyze_pointmaze_budgeted_trigger_stage9.py).
- Stage298: [qualified report](../results/native_transit_service_confirmation_stage298_20261019_r1/report/result.json).
- Stage301: [qualified report](../results/native_transit_service_comparison_stage301_20261020_r1/report/result.json).
- Stage304: [qualified report](../results/native_transit_closed_loop_confirmation_stage304_20261021_r2/report/result.json), [completion receipt](../results/native_transit_closed_loop_confirmation_stage304_20261021_r2/evaluation_completion_receipt.json), [descriptive event audit](../results/native_transit_closed_loop_confirmation_stage304_20261021_r2/report/descriptive_audit.json).

`build_plan_validity_evidence.py` reproduces Stage9 root effects and Student-t
intervals from the eight existing cell summaries and selects unchanged Transit
report contrasts. It does not refit predictors, resample outcomes, change gates
or contact servers. The old analyzer's ISE key contains `candidate_minus_fixed`,
but its executable subtraction is fixed minus candidate; the manuscript and
compact artifact name the arithmetic explicitly. No historical sign is changed.

## Historical positive results and their place

The [complete historical inventory](../md/freq_hrl_stage304_closed_loop_and_paper_evidence_2026-10-21.md)
preserves MuJoCo responsibility restoration, mixed Quant baselines, learned
PointMaze upper policies, early Transit service gains and cached information
probes. They remain separate from the active timing claim:

- MuJoCo v12/v14.29 support responsibility restoration with a reward floor;
  38/47 accepted v14.29 transactions preserve physical behavior exactly.
- Quant v7.4 has 8/12 favorable Holm-controlled contrasts and one supported
  return harm versus recurrent generic HRL.
- PointMaze Stages54-57 and 87-100 establish bounded learning/execution effects;
  Stage135 compact-upper gains coexist with Stage136's harmful joint update.
- Transit Stage168's 0.20-0.32min waiting benefit costs extra fleet resources.
- Old247's new-future service gains coexist with Stage287's failed new-prefix
  confirmation. Stage281's useful query information is not a deployable model.
- Stage288 confirms some held cost gains, not a wide-over-dense causal benefit;
  Stage292/294 waiting gains lack joint total-cost support; Stage296 retains
  only 29.50% of base waiting gain and fails its primary despite raw-relative gains.

The old `evidence/authoritative_registry_v1.json` and generated historical
claim ledger remain the sources for their registered mechanisms. This map
defines the active manuscript's source selection; it does not overwrite or
reclassify those historical decisions. A number of research stages is not
a number of independent confirmations of one algorithm.
