# Active Manuscript Evidence Map

Date: 2026-11-12. Main manuscript: [manuscript.md](manuscript.md).
Compact numerical source: [plan_validity_evidence.json](plan_validity_evidence.json).
Core figures and traceability: [figure evidence audit](figure_evidence_audit.md).

## Central argument

Supervised renewal value can guide a fixed planner-call budget and improve
PointMaze closed-loop control. The independent-seed claim is confined to that
task family. Reacher and cart-pole separate waypoint freshness from timing
opportunity. Transit supplies separate positive local-control evidence and a
negative repeated-decision boundary. None is same-algorithm domain transfer.

## Main evidence and comparators

| Manuscript location | Evidence | Unit and exact comparison | Interpretation |
| --- | --- | --- | --- |
| Method / Table1 | Stage9 independent confirmation | Eight fresh optimizer roots, each 16 held-out paths; shared controller, history trigger versus fixed/random-offset/current-only timing | All five frozen gates pass; central positive result |
| Results5.1 | Stage9 development | Eight development roots, distinct from confirmation | Supporting design result; not pooled with confirmation |
| Results5.2 | Stages88/89/95/100/129/131/135 | Their own teacher/root and frozen-upper contracts | Genuine learned-upper effects, not replications of the Stage9 trigger |
| Table3 | Stage298 and301 independent E32 | Two separate 400-prefix cohorts; paired single-intervention outcomes, fixed profiles averaged within prefixes | Raw-relative cost/wait gains; learned fleet increment is not established |
| Table4 / Results5.4 | Stage304 | 320 complete scenes, four paired controllers, one fixed profile, two upper decisions | Registered dual-win and statistical half-base retention fail; no extra joint cost benefit |
| Limitations | Stage10 learned termination | Revealed Stage9 confirmation paths; mostly deadline-only deployment | Failed development baseline, not competitive learned-termination superiority |
| Follow-up boundary | [Frozen pilot episode diagnostic](../md/freq_hrl_local_episode_result_2026-11-10.md) | Two existing optimizer roots/task, 16 held-out development paths/root; selected threshold and learned termination | PointMaze directional support; mixed Reacher and failed best-clock eligibility on one root; no independent strong-baseline claim |
| Follow-up boundary | [Fresh-prefix conditioning](../md/freq_hrl_boundary_conditioning_confirmation_2026-11-11.md) | 16 fresh prefixes/root, four frozen roots; boundary tail at `t + 50`, zero fits | No conditioning upgrade; Wide beats zero on three roots, not an episode-policy result |
| Strong timing follow-up | [Stage9 development and build gates](../md/freq_hrl_stage9_strong_timing_2026-11-11.md#build-outcome) | Two pilot roots, then eight fresh controller builds | Development passes; formal evaluation held before test access because one learned termination misses best-clock competence |
| Results5.1 follow-up | [Distance diagnostic](../results/pointmaze_stage9_distance_diagnostic_20261112/summary.json) | All eight frozen builds, 32 new paired paths/root, 24 upper calls per episode | 7.28% lower mean ISE versus the selected distance threshold; not learned-termination confirmation |
| Table2 | [Frozen timing portfolio](../results/pointmaze_stage9_timing_portfolio_20261112/summary.json) | Same eight builds, 64 further paired paths/root, 24 calls and shared CPU/steps/latency caps | 5.81% lower ISE than distance; 21.56% lower than the selection-based portfolio; original learned-only HOLD retained |
| Results5.1 mechanism ablation | [Matched timing features](../md/freq_hrl_stage9_scale_ablation_2026-11-12.md#completed-result) | Same eight controllers, three equal-capacity heads on cached labels, 64 further paired paths/root | Short history improves ISE 6.44%; the 500ms/100ms substitution worsens ISE 5.85% versus 100ms/100ms; no general multiscale attribution |
| Table5 | [Reacher applicability precheck](../md/freq_hrl_reacher_timing_applicability_2026-11-12.md#completed-result) | Two frozen roots, 24 paired paths/root, matched call counts | Waypoint authority supported; all four privileged timing intervals cross zero; formal transfer remains on hold |
| Table5 | [Native cart-pole precheck](../md/freq_hrl_cartpole_timing_applicability_2026-11-12.md#completed-result) | Two fixed LQR configurations, 24 paired paths each, no policy fits | Visible timing signal passes, waypoint contrast reverses, joint qualification fails; not a Stage9 result |
| Results5.5 | [Cart-pole waypoint factorial](../md/freq_hrl_cartpole_waypoint_factorial_2026-11-12.md#completed-result) | Two LQR weights and two clocks crossed on the same 32 paths | Timely-waypoint gain reverses by weight at both offsets; the prior qualification remains failed |

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
intervals from eight compact episode files in `paper/data/stage9/` and selects unchanged Transit
report contrasts. It does not refit predictors, resample outcomes, change gates
or contact servers. The old analyzer's ISE key contains `candidate_minus_fixed`,
but its executable subtraction is fixed minus candidate; the manuscript and
compact artifact name the arithmetic explicitly. No historical sign is changed.
The artifact also includes the first registered confirmation path's saved
call schedule, score-check counts by mode and the existing Stage304 descriptive
event audit. These additions supply figures, not new performance experiments.
Table 5 uses the linked applicability reports and their compact summaries
directly; its comparisons use different rules and are not included in Figure 2.

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
