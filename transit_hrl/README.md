# Transit HRL

This directory is the isolated workspace for the general Freq-HRL direction.
Existing `FreqDuet/` and `transit_duet/` code should be treated as read-only
references unless a file is intentionally copied into this tree first.

The active research mainline is now **multiscale goal-conditioned HRL**:

- the upper policy emits a state-space goal or plan at a physical-time macro
  interval;
- the lower policy alone emits the actuator action and retains full physical
  feedback;
- causal history and multiscale policies use the same trailing samples;
- projector, promotion, leakage loss, and responsibility gauge are disabled in
  the stage-1 mainline until a task-level conflict justifies them.

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
This is an ordinary-HRL admission gate, not yet a positive Freq-HRL result. See
`md/freq_hrl_pointmaze_stage2_protocol_2026-09-19.md` and
`md/freq_hrl_reorientation_2026-09-19.md` for the research boundary.

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
