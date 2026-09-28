# Stage-30 Separate Motion Inference

Reuse Stage-29 fit observations and frozen action-response predictions for
roots 209011/209061 (preflight 208001). No controller or physical-model fitting,
weight download, environment replay or policy deployment. Regenerate external
tapes and match every cached observed prefix and one-step external label.

Generic motion forecaster takes only external observations: current six
channels plus target-channel causal velocities at lags 1/10/25/50. Unit ridge,
training-only normalization with the existing 1e-8 constant-feature threshold,
unpenalized intercept and direct displacement-rate heads at 1/10/25/50/100
steps are frozen. Three equal-width views: history, current repetition and
within-prefix shuffled history preserving the current frame. Lag-one velocity
extrapolation and zero displacement are additional controls. No parameter,
epoch, horizon or query-roster search follows outcome access.

Fit rows reuse Stage-29 seeds at steps 64,69,..., excluding endpoints whose
100-step label exceeds the episode. Fresh query bases 3299000/3300000/3301000,
offsets 101-108 (101-102 preflight), use the same bounded step grid. Exclude
source fit/evaluation and inherited controller training/evaluation paths.
One-step gate: history target-rate MSE below current repetition. Planning gate:
average target-rate MSE over 10/25/50/100 below current, shuffled, lag-one and
zero. Both gates must pass on both roots; ties fail. Other-channel errors and
individual horizons are diagnostic, not alternate qualification endpoints.

Each full root: 3328 fit/1664 fresh query/1664 cached bridge rows, three linear
fits and three multi-RHS solves (90 scalar RHS), 38432 generated tape points.
Preflight: 56/56/56 rows, three fits/90 RHS, 1806 generated tape points. New
environment steps and controller/physical-model updates are zero. The bridge
replaces only cached external means; physical predictions stay identical.
Report root/path metrics and a small audit sample; full arrays stay remote.
Scheduler node001-node006 dynamically, one CPU/1536 MB, compact JSON only.
Twenty-four focused tests passed, covering causal time units, equal capacity,
label isolation, unchanged physical prediction, source labels, disjoint paths,
all-control gates and accounting. Implementation frozen at `7d87dd7e8d`.
Preflight `t101738` completed on node006: 56/56/56 rows, three fits/90 RHS,
1806 generated tape points and zero environment steps. Server-side independent
label, forecast, normalization and metric recomputation matched. One-step
gate passed; planning/joint gates failed. Keep this negative result; full
settings remain unchanged. Retrieved 81590 bytes of JSON, no raw arrays.
Full tasks `t101739/101740` completed on node006/node005 at the same frozen
revision. The [full result](freq_hrl_stage30_separate_motion_result_2026-09-28.md)
passes both gates on both roots: planning MSE drops 58.11%/63.12% versus current
and 55.39%/56.56% versus lag-one. All 16 new paths beat every planning control.
Server-only independent recomputation matches labels, predictions, scales and
metrics. Three fits per root and zero environment steps are accounted for.
This qualifies target-motion forecasts, not a deployed control policy.

## Limitations

Fresh synthetic external paths test motion inference, not native control or
reward improvement. Cached bridge comparisons are development diagnostics.
This does not establish shared-latent interference as the Stage-29 failure
cause, calibrated uncertainty, or a plan-value policy. Stage-9 remains the
performance reference.
