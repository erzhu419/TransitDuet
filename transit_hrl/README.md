# Transit HRL

This directory is the isolated workspace for the general Freq-HRL direction.
Existing `FreqDuet/` and `transit_duet/` code should be treated as read-only
references unless a file is intentionally copied into this tree first.

The active research mainline is now **plan-validity qualification for
goal-conditioned HRL**:

- the upper policy emits a state-space goal or plan at a physical-time macro
  interval;
- the lower policy alone emits the actuator action and retains full physical
  feedback;
- the Stage-7 evidence supports the ordinary hierarchy more strongly than any
  fixed frequency representation or routing rule;
- Stage 8 found strong plan dependence but rejected immediate regime-event
  timing and did not support privileged regime input;
- Stage 8B found positive paired `keep`/`renew` plan value and usable causal
  ranking, but causal history did not significantly beat current plan/state;
- deployed belief, plan-validity trigger, and joint training therefore remain
  disabled.

The completed four-grid Stage-1 protocol was run with:

```bash
python3 scripts/run_multiscale_goal_stage1.py --methods flat_history flat_multiscale hrl_history hrl_multiscale flat_causal_filter --scenarios clean slow_target_fast_force slow_signal_fast_observation_noise band_swap --output results/multiscale_goal_stage1/result.json
```

Use `--dry-run` to write the fully resolved protocol without training. The
checkpoint objective is mean episode return, and every result records `dt`,
window durations, signal RMS, saturation, system response time, and the
actor-observability contract. Its 160-cell development result was negative;
see `md/freq_hrl_multiscale_goal_stage1_v1_result_2026-09-19.md`.

The current Stage-2 gate tests ordinary goal-conditioned HRL on PointMaze
before any multiscale mechanism is admitted:

```bash
python3 scripts/run_pointmaze_goal_stage2.py --methods flat_goal_ppo hrl_goal_ppo --iterations 768 --horizon 300 --upper-period-seconds 0.25 --maximum-subgoal-delta 0.75 --output results/pointmaze_goal_stage2/result.json
```

The upper policy emits a relative XY waypoint and the lower policy alone emits
physical acceleration. The lower policy cannot observe the final task goal.
Capacity and environment transitions are matched against flat PPO; checkpoint
selection is lexicographic by validation success rate and then dense return.
The completed V1 development gate was not supported: hierarchical success was
0.359 with root-level 95% CI [0.197, 0.522], below the registered 0.50 lower-CI
threshold. Multiscale enhancement therefore remains blocked while the
ordinary-HRL credit path is repaired. V2 terminates lower GAE at waypoint
changes, restores progress-based intrinsic reward, and uses a larger, less
frequently reused validation set with entirely fresh seeds. V2 also failed the
gate: both methods reached 0.297 mean success, and post-hoc analysis exposed a
positive-dense-return/early-termination conflict that must be corrected before
further hierarchy tuning. V3 uses the official fixed-horizon PointMaze mode so
earlier success and larger dense return are directionally aligned for both
methods. V3 passed the ordinary-HRL gate with hierarchical success 0.789 and
root-level 95% CI [0.710, 0.868]; its paired success advantage over flat PPO
remained inconclusive. Multiscale factorial testing is now admitted, but this
is not yet a positive Freq-HRL result. See
`md/freq_hrl_pointmaze_stage2_protocol_2026-09-19.md`,
`md/freq_hrl_pointmaze_stage2_v1_result_2026-09-19.md`,
`md/freq_hrl_pointmaze_stage2_v2_protocol_2026-09-19.md`,
`md/freq_hrl_pointmaze_stage2_v2_result_2026-09-19.md`,
`md/freq_hrl_pointmaze_stage2_v3_protocol_2026-09-19.md`,
`md/freq_hrl_pointmaze_stage2_v3_result_2026-09-19.md`, and
`md/freq_hrl_reorientation_2026-09-19.md` for the research boundary.

The active Stage-3 V2 experiment is a preregistered PointMaze factorial:

```bash
python3 scripts/submit_pointmaze_multiscale_stage3_scheduleurm.py --run-name pointmaze_multiscale_stage3_v2_preflight_20260919_r1 --preflight
```

Raw-history, causal-filter, and Haar flat policies use the same 32
actor-visible samples. Both HRL levels retain current physical feedback;
multiscale features augment it, with slow+mid routed upward and mid+high routed
downward. Observation noise, continuous action stress, and persistent mode
shift are separate scenarios. The auxiliary causal-filter baseline and the
factorial interaction prevent generic smoothing gains from being relabeled as
Freq-HRL gains. See
`md/freq_hrl_pointmaze_multiscale_stage3_v2_protocol_2026-09-19.md`.

The V1 preflight was software-valid but design-invalid because its upper
multiscale state removed current physical feedback. All 64 V1 development
tasks were cancelled, and no partial output is evidence. See
`md/freq_hrl_pointmaze_multiscale_stage3_preflight_2026-09-19.md`.

The corrected 20-cell V2 preflight passed its software, state-contract,
capacity, seed-pairing, option-boundary, runtime, and independent-stress
checks. It authorizes the frozen 160-cell development matrix but is not
performance evidence. See
`md/freq_hrl_pointmaze_multiscale_stage3_v2_preflight_2026-09-19.md`.

The 160-cell V2 development matrix is complete. Multiscale routing improved
the raw-history HRL baseline under both primary stresses, but the registered
Freq-HRL-specific gate failed: the factorial interactions and causal-filter
comparison were not supported, and flat multiscale PPO was better under
observation noise. See
`md/freq_hrl_pointmaze_multiscale_stage3_v2_result_2026-09-20.md`.

The active registered experiment is a within-HRL routing attribution: raw
history, causal filtering, all bands to both levels, the intended routing, and
the swapped routing are compared under clean, observation-noise, and action-
stress conditions. It tests whether band-to-level assignment matters beyond
generic Haar features or compression. See
`md/freq_hrl_pointmaze_routing_stage4_protocol_2026-09-20.md`.

Its 15-cell scheduler preflight passed the frozen software, PPO-update,
capacity, seed-role, state-feedback, timing, runtime, and independent-stress
checks. This is not performance evidence; it authorizes the unchanged
120-cell development matrix. See
`md/freq_hrl_pointmaze_routing_stage4_preflight_2026-09-20.md`.

The 120-cell matrix is complete. The intended slow+mid-upper / mid+high-lower
routing beat raw history in clean and observation-noise conditions, but the
registered attribution gate failed: routed did not significantly beat the
all-band control and was significantly worse than swapped routing in all three
scenarios. Post-hoc design audit found that unequal Haar band sizes also
swapped the upper/lower input-layer parameter allocation, so this run does not
cleanly identify frequency semantics. A fresh equal-shape masked-routing
protocol is required. See
`md/freq_hrl_pointmaze_routing_stage4_result_2026-09-20.md`.

Stage-4 V2 repairs that attribution confound with fixed 134-dimensional states
at both levels. Routed and swapped controls now zero excluded Haar blocks while
retaining identical network shapes, per-level parameter counts, and initial
weights for each root. It uses fresh seeds and the unchanged V1 claim gate.
See `md/freq_hrl_pointmaze_routing_stage4_v2_protocol_2026-09-20.md`.

The 15-cell V2 preflight passed equal-shape, identical-initialization,
learned-update, fresh-seed, timing, runtime, and stress-channel checks. It is
software evidence only and authorizes the unchanged 120-cell matrix. See
`md/freq_hrl_pointmaze_routing_stage4_v2_preflight_2026-09-20.md`.

The full V2 matrix is complete. Equal-shape routed history significantly beat
raw history in clean and both stresses and beat causal filtering in both
stresses. The selective-routing gate still failed: routed did not significantly
beat all-band HRL in either stress and was significantly worse than swapped
routing under action stress. This closes endogenous PointMaze history as
evidence for the central exogenous-stream assignment claim. See
`md/freq_hrl_pointmaze_routing_stage4_v2_result_2026-09-21.md`.

Stage 5 now separates current endogenous physical state from an
action-independent, actor-visible external stream containing a slow moving
target and a fast measured force. Frequency routing is disabled until ordinary
HRL learns this dynamic tracking task. The frozen two-cell scheduler preflight
passed state-shape, parameter-budget, causal-observability, PPO-update,
option-boundary, runtime, and dynamic-placement checks. It is software evidence
only and authorizes the 16-cell development matrix. See
`md/freq_hrl_pointmaze_exogenous_stage5_protocol_2026-09-21.md` and
`md/freq_hrl_pointmaze_exogenous_stage5_preflight_2026-09-21.md`.

The 16-cell development matrix is complete. HRL significantly improved over
its own paired untrained policy, but its mean tracking success of 0.616 had a
root-level 95% interval [0.424, 0.808]. The lower endpoint missed the frozen
0.50 absolute threshold, so the substrate gate failed and frequency routing
remains blocked. See
`md/freq_hrl_pointmaze_exogenous_stage5_result_2026-09-21.md`.

A bounded stability screen selected eight training rollout roots, and the
fresh-seed Stage-5 V2 confirmation then passed the full substrate gate: HRL
tracking success was 0.887 [0.860, 0.915], with positive success and return
gains over its paired untrained policy. HRL-versus-flat remained inconclusive.
External-stream frequency-routing attribution is now admitted, but no routing
claim follows from the substrate result itself. See
`md/freq_hrl_pointmaze_exogenous_stage5_v2_result_2026-09-22.md`.

The admitted 64-cell Stage-6 attribution matrix is complete. All-band
multiscale HRL improved success over HRL history by 0.050 [0.003, 0.096], and
intended routing improved over history and causal filtering. The strict
selective-routing claim nevertheless failed: intended routing was
inconclusive against both all-band and swapped routing, and the
hierarchy-by-multiscale interaction crossed zero. Fixed slow-upper/high-lower
masking is therefore closed as the mainline mechanism. The next independent
test is a fresh confirmation of the simpler flat/HRL by history/all-band
factorial, not another repair of this failed gate. See
`md/freq_hrl_pointmaze_exogenous_routing_stage6_result_2026-09-22.md`.

Stage 7 freezes that fresh confirmation at 16 optimizer roots using only the
flat/HRL by history/all-band factorial. Its four-cell preflight passed on four
independent CPU nodes and is software evidence only. The fixed 64-cell
confirmation has been dispatched without interim analysis or sequential root
extension. See
`md/freq_hrl_pointmaze_exogenous_multiscale_stage7_protocol_2026-09-22.md` and
`md/freq_hrl_pointmaze_exogenous_multiscale_stage7_preflight_2026-09-22.md`.

The full Stage-7 confirmation is now complete and **not supported**. HRL
all-band versus HRL history was +0.0018 [-0.0289, 0.0326], HRL all-band versus
flat all-band was +0.0584 [-0.000041, 0.1168], and the hierarchy-by-multiscale
interaction was +0.0236 [-0.0488, 0.0960]. The Stage-6 representation signal
did not replicate, and no sequential root extension is allowed. Ordinary
goal-conditioned HRL retains supported return/RMSE improvements, but the
current PointMaze evidence does not support a confirmed frequency-specific
algorithm. See
`md/freq_hrl_pointmaze_exogenous_multiscale_stage7_result_2026-09-22.md`.

Stage 8 is a new development task, not an extension of Stage 7. It introduces
a 12-second PointMaze task with hidden persistent target-motion regimes,
measured short force pulses, and a reward-irrelevant distractor. The frozen
goal-conditioned controller is evaluated under stale/perturbed plans and under
privileged event schedules that preserve the fixed 2 Hz upper-call budget. A
separate oracle controller sees only the current true regime, never future
regimes. The primary endpoint is integrated squared tracking error. Learned
belief and event-triggered replanning are authorized only if plan refresh,
plan content, current regime information, matched-budget timing, and a usable
causal identification window are all supported across the fixed eight roots.
The two-cell preflight passed runtime, state, capacity, update, path-pairing,
budget, variable-duration, and compact-artifact checks on `node004` and
`node006`. It authorizes the fixed 16-cell development matrix but supplies no
performance evidence. See
`md/freq_hrl_stage8_plan_value_protocol_2026-09-22.md` and
`md/freq_hrl_stage8_plan_value_preflight_2026-09-22.md`.

The full 16-cell Stage-8 matrix is complete and **does not authorize Stage
9**. The learned history controller, plan refresh, and plan content checks were
supported, but current-regime oracle information was inconclusive. Zero-delay
event replanning significantly worsened integrated tracking error, and waiting
250 ms improved rather than degraded it. The audit also found that the nominal
500-ms schedule realized only 0.307 s mean first-response delay and that event
relocation exposed a fixed-duration planner to 1-112-step options. The failed
run is retained; no roots will be appended. The next admissible work is a fresh
counterfactual `keep` versus `renew` plan-value qualification with corrected
delay and variable-duration semantics, not a learned trigger. See
`md/freq_hrl_stage8_plan_value_result_2026-09-22.md`.

Stage 8B is the independent repair. It replaces event-time relocation with
paired `keep current waypoint` / `renew now` simulator branches sharing an
exact deterministic prefix and common 0.50-second downstream rule. A causal
history predictor is qualified against plan-age, current-plan/state, and
change-magnitude baselines; true regime context is diagnostic only. The fixed
eight-root conjunction can authorize only later trigger development, not a
closed-loop claim. See
`md/freq_hrl_stage8b_counterfactual_plan_validity_protocol_2026-09-22.md`.

The one-cell Stage-8B preflight passed runtime, exact-prefix replay, feature
causality, branch-call isolation, balanced opportunity coverage, predictor,
extra-transition accounting, analyzer, and compact-artifact checks on
`node004`. It authorizes the unchanged eight-root development matrix but is
not performance evidence. See
`md/freq_hrl_stage8b_counterfactual_plan_validity_preflight_2026-09-22.md`.

The fixed eight-root Stage-8B matrix is complete and **does not authorize Stage
9**. Seven of eight registered checks passed: delayed regime-change renewal had
positive paired local value, exceeded immediate regime, force-pulse, and
distractor opportunities, and causal history had positive rank and selected
utility. The required history-versus-plan/state selected-utility contrast was
+0.004908 [-0.001556, 0.011371], so it remained inconclusive. The completed run
is retained and no roots will be appended. See
`md/freq_hrl_stage8b_counterfactual_plan_validity_result_2026-09-22.md`.

Stage 8C is the fresh predictor repair. The revealed Stage-8B data are used
only to select a frozen 39-feature causal interaction structure; all Stage-8C
evidence will use new optimizer and path seeds. The candidate is compared with
a 170-feature nonlinear current-only baseline and a 170-feature generic causal
dynamic model, with ridge regularization selected only by grouped branch-fit
cross-validation. No predictor receives event labels, regime context, future
values, or distractor features. See
`md/freq_hrl_stage8c_compact_plan_validity_protocol_2026-09-22.md`.

The first Stage-8C preflight failed before evidence because one frozen path had
no force pulse within its 240-step implementation horizon. The repaired
300-step preflight completed as `t100474` on `node002` and passed runtime,
balanced-opportunity, grouped-CV, causal-feature, exact-prefix, call-isolation,
branch-budget, analyzer, and compact-artifact checks. It authorizes the
unchanged eight-root matrix but supplies no performance observation. See
`md/freq_hrl_stage8c_compact_plan_validity_preflight_2026-09-22.md`.

The package also retains the earlier components for:

- causal exogenous stream encoders;
- causal fixed-bin stream adapters;
- experimental upper/lower frequency routing masks;
- high-frequency to low-frequency promotion;
- action-effect leakage regularization;
- frequency responsibility diagnostics;
- Transit and Trading action-effect adapters.

This is not yet a full replacement runner for FreqDuet.  It is the shared core
that future `FreqTransitDuet` and `FreqTradeDuet` experiments should import.

Current domain entry points:

- `freq_hrl.domains.transit.TransitFrequencyTracker`: FreqDuet-compatible
  `update`, `upper_features`, `lower_features`, and `summary` API backed by the
  shared core encoders and promotion gate.
- `freq_hrl.domains.trading.TradingFrequencyTracker`: causal market-bar
  frequency features for portfolio/execution experiments.
- `freq_hrl.domains.trading.PortfolioExecutionEnv`: a minimal portfolio target
  plus execution-speed environment for early FreqTradeDuet tests.
- `freq_hrl.domains.mujoco`: goal-observation and physical-time adapters for
  PointMaze/AntMaze, plus the historical constrained-action adapter.

The following historical MuJoCo entry point is the **spectral action-constraint
side branch**, not the multiscale goal-conditioned mainline:

```bash
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.mujoco.control_validation --method freq_hrl --env-id HalfCheetah-v5 --disturbance-mode standard --train-seeds 31013 31019 31033 --selection-seeds 32003 32009 32027 --eval-seeds 33013 33023 33029 33037 33049 --steps 500 --iterations 64 --optimizer-seed 34019 --output-dir transit_hrl/results/mujoco_control/halfcheetah/freq_hrl/replicate_34019
```

The optional pinned runtime is listed in `requirements-mujoco.txt`. This side
branch compares capacity-matched `flat_ppo`, `generic_hrl`,
`freq_hrl_no_leakage`, and `freq_hrl`; a short smoke cell validates only the
software path and is not performance evidence.

Run the current smoke tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m unittest discover -s transit_hrl/tests
```

Run the current synthetic performance validation with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.trading.performance_validation --seeds 42 123 456 789 2026 --steps 720 --assets 3 --output-dir transit_hrl/results/trading_performance
```

The default validation uses the tuned promotion setting from the sweep:
`threshold=0.00035`, `persistence_ratio=0.40`, `mid_gain=0.5`, and
`adapt_gain=0.25`. Leakage regularization is also applied online to the
learner-facing reward with `leakage_reward_scale=0.00005`; the `no_leakage`
baseline disables only that reward shaping path.
The report also includes frequency-aware reward attribution columns for
low-frequency cost, high-frequency cost, leakage cost, and promotion adaptation
cost.
The trading baseline matrix currently covers `vanilla_rl`, `hrl_raw`,
`raw_history`, `freq_single_policy`, `lf_upper_only`, `hf_lower_only`,
`allfreq_alllayers`, `swapped`, `no_promotion`, `no_leakage`, and `freq_hrl`.
Diagnostic columns include `PromotionDelay`, `ShockResponseTime`,
regime-promotion accuracy, and 120-bar recovery cost.

Run the Phase-0 logging-only audit with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.trading.phase0_audit --seed 42 --steps 180 --assets 3 --output-dir transit_hrl/results/trading_phase0_audit
```

It writes `phase0_trace.jsonl` and verifies that logged causal bins reconstruct
the logged frequency state.

Run the Transit config-isolation audit with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.transit.config_isolation --config-dir transit_hrl/freq_transitduet/configs_freqduet --output-dir transit_hrl/results/transit_config_isolation
```

Run the minimal pluggable-policy entry point with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.trading.policy_entry --mode eval --policy heuristic --seeds 42 123 --steps 360 --assets 3 --output-dir transit_hrl/results/trading_policy_entry
```

Run the promotion sweep with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.trading.promotion_sweep --seeds 42 123 456 789 2026 --steps 720 --assets 3 --thresholds 0.00035 0.00050 0.00065 0.00080 --ratios 0.20 0.30 0.40 --mid-gains 0.0 0.5 1.0 --adapt-gains 0.0 0.10 0.25 0.50 --output-dir transit_hrl/results/trading_promotion_sweep
```

The validation artifacts are written to:

- `transit_hrl/results/trading_performance/per_seed.csv`
- `transit_hrl/results/trading_performance/summary.csv`
- `transit_hrl/results/trading_performance/summary.json`
- `transit_hrl/results/trading_performance/report.md`

Run the current Transit simulator validation pilot from
`transit_hrl/freq_transitduet` with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/erzhu419/mine_code/TransitDuet/transit_hrl FREQDUET_TORCH_THREADS=1 python3 scripts/run_freqduet_ablation.py --configs T_nofreq_terminal,T_rawhistory_terminal,T_lf_upper_terminal,T_hf_lower_terminal,T_freqhrl_terminal,T_allfreq_terminal,T_swapped_terminal,T_nopromotion_terminal,T_noleakage_terminal --seeds 42 --episodes 1 --last-k 1 --logs-dir logs_freqhrl_validation --out-dir results_freqhrl/transit_validation_smoke --upper-warmup-eps 0 --worker-threads 1 --clean
```

The pilot report is written to
`transit_hrl/freq_transitduet/results_freqhrl/transit_validation_smoke/report.md`.
