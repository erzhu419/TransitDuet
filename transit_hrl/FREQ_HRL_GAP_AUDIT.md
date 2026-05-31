# Freq-HRL Gap Audit

This audit compares the current `transit_hrl/freq_hrl` implementation against
`md/freq_hrl_dev_manual.md` and `md/freq_hrl_gpt.md`.

## Current Status

Implemented:

- Domain-agnostic core package under `freq_hrl`.
- Causal EMA, causal Fourier, causal state-space, and trailing-window Haar
  wavelet encoders.
- Causal stream binning adapter.
- Upper/lower frequency router with mask tests.
- Promotion gate with persistence, hysteresis, and cooldown.
- Generic action-effect leakage regularizer.
- Frequency diagnostics with FocusScore and leakage metrics.
- Generic Bernstein plan curve.
- Causal rolling plan-curve state for vector-valued upper plans; the synthetic
  trading loop can now route `freq_hrl` target weights through this plan curve.
- Transit and Trading action-effect operators.
- Transit-compatible frequency tracker.
- Trading market-bar tracker and toy portfolio/execution environment.
- Synthetic trading performance validation with task metrics and frequency diagnostics.
- Causal HF-utility gate for the trading lower controller, so high-frequency
  execution-speed modulation is reduced when recent HF signals have not
  predicted next-bar returns.
- Conservative promotion handling in trading: shorter cooldown, smaller
  residual absorption, and no additional LF absorption during cooldown-only
  promotion visibility.
- Regime-buffer promotion filter: persistent HF residuals must also accumulate
  enough mid-frequency buffer before they can be promoted.
- Optional promotion warm-up and activation-strength filters are implemented
  and exposed to the validation/sweep CLIs; they are not enabled in the default
  trading setting because the current pressure matrix shows a cross-regime
  tradeoff.
- Online leakage reward shaping and reward attribution accounting.
- Phase-0 JSONL logging and offline causal-bin reconstruction audit.
- Transit config-isolation diff audit.
- Pluggable high-level planner and low-level controller policy interfaces.
- Lightweight learned linear trading policy and cross-entropy training entry.
- On-policy Gaussian policy-gradient trading policy (`pg_linear`) that samples
  upper targets and lower execution speeds and trains with reward-to-go
  REINFORCE updates.
- Leakage-constrained `pg_linear` training path with policy-loss leakage
  penalties and a Lagrange-style constraint multiplier.
- Linear Gaussian actor-critic trading policy (`ac_linear`) with separated
  upper low-frequency and lower high-frequency TD(0) critics. The current
  held-out run reaches return `0.2804`, objective `0.4436`, and total leakage
  `1.6475`, improving return/objective and leakage versus the previous
  `pg_linear` runs while giving up Sharpe.
- Shared dual-level PPO actor-critic trainer under `freq_hrl.rl`, with
  separate upper/lower Gaussian actors, separate value functions, clipped PPO
  updates, GAE, entropy/value losses, gradient clipping, and a trading domain
  adapter. The current persistent-shift held-out run reaches Sharpe `15.402`,
  return `0.2351`, and `LowerLFDrift=1.716`; PPO updates reduce lower drift in
  training checkpoints but currently trade away return/Sharpe, so this is
  trainer-completeness evidence rather than the strongest learned-policy
  performance result.
- Shared `train_dual_ppo` loop under `freq_hrl.rl.training` now drives both
  Trading and Transit adapters through the same upper/lower trajectory batch,
  objective, summary, checkpoint-selection, and held-out evaluation path.
- Transit PPO surrogate adapter using `TransitFrequencyTracker` features and
  the same shared PPO loop as Trading. The current persistent-shift held-out
  run reaches reward `-4.460`, wait proxy `4.203`, and
  `LowerLFDrift=1.000`; this closes the shared-core plumbing gap but is still a
  surrogate, not the copied Transit runner's native simulator loop.
- Learned Bernstein plan-curve actions for PPO upper policies. The upper actor
  can now output plan coefficients instead of one-step targets, and a shared
  mapper converts those coefficients into executable portfolio/headway targets
  for lower control. The trading plan-action run reaches return `0.2991`,
  Sharpe `14.750`, `LowerLFDrift=1.548`, and plan smoothness
  `0.000008`; the Transit surrogate plan-action run reaches reward `-4.500`,
  wait proxy `4.200`, and `LowerLFDrift=1.000`.
- Promotion-triggered high-level replanning for deterministic plan-curve
  control. Promotion can now force `CausalPlanCurveState` to replan immediately
  instead of waiting for the regular replan interval, and the validation logs
  forced replan counts. On the promotion-recovery trading scenario,
  promotion-triggered replanning improves return by `+0.0014`, Sharpe by
  `+0.075`, post-shift-120 PnL by `+0.00075`, recovery regret by `-0.00075`,
  and `LowerLFDrift` by `-0.0326` versus interval-only plan reuse over
  10 paired seeds.
- Shared PPO now supports loss-level primal-dual constraints via trajectory
  cost fields, fixed constraint coefficients, and a dual multiplier. Trading
  and Transit surrogate adapters feed online lower-LF leakage costs into this
  path. The constrained trading plan-PPO run reduces `LowerLFDrift` to `0.856`
  versus `1.548` for unconstrained plan-PPO and `1.716` for direct PPO, but
  trades off return (`0.1649`) and Sharpe (`8.756`). The Transit surrogate
  constrained run activates the dual multiplier but does not materially reduce
  `LowerLFDrift`, so the strong leakage claim is still clearest on trading.
- Lower-LF drift can now be constrained explicitly at two levels: learned
  `pg_linear`/`ac_linear` policy losses have separate `LowerLFDrift` penalty
  and Lagrange controls, and the synthetic trading controller has an optional
  drift-speed constraint that increases lower execution speed when the lower
  effect persistently lags the upper plan.
- Trading pressure-test matrix across six synthetic market regimes, including
  the persistent reversal recovery shock.
- Promotion recovery sweep with sharded scheduler execution and merge output.
- Dedicated promotion-recovery validation on a persistent reversal shock,
  including oracle-regime `recovery_regret_120`.
- Trading decomposer ablation across causal EMA, Fourier, state-space, and
  trailing-window Haar wavelet encoders.
- Trading plan-curve validation: enabling the upper portfolio plan curve on
  persistent-shift `freq_hrl` raises Sharpe from `16.062` to `16.106`, reduces
  turnover from `5.76` to `5.34`, and reduces total leakage from `1.822` to
  `0.969`, with a `0.667` plan reuse ratio.
- Lower-LF constraint validation: combining the portfolio plan curve with
  `lower_lf_drift_speed_gain=0.1` keeps `freq_hrl` as the best Sharpe baseline
  (`16.080`) while reducing `LowerLFDrift` to `0.916` versus the original
  default `1.821`.
- Public Level-1 ETF daily-bar validation path using local CSV inputs.
- Public ETF encoder ablation across EMA, state-space, and Haar wavelet
  decomposers.
- Diagnostic plot generation for signal decomposition, promotion, ablations,
  FocusScore, NoLeakage drift comparison, pressure matrix, and promotion
  recovery.
- 27-seed x 10-episode copied-Transit ablation validation on the 9-config matrix.
- Paired copied-Transit performance report with bootstrap confidence intervals
  and per-metric winners for the 27-seed matrix.
- Unit/smoke tests for causality, router masks, promotion, leakage, plan curves, stream adapter, domain trackers, and validation harness.

## MVP-Critical Checklist

1. Real FreqTransitDuet validation.
   - Done: `freq_transitduet` contains a copied Transit runner/env under `transit_hrl`.
   - Done: copied `frequency` entry point is bridged to `TransitFrequencyTracker` from the shared core.
   - Done: a 27-seed x 10-episode validation was run for the 9-config Transit matrix.
   - Done: paired statistical report
     `freq_transitduet/results_freqhrl/transit_performance_validation/report.md`
     summarizes bootstrap CIs and per-metric winners from the copied runner logs.
   - Current evidence: `T_freqhrl_terminal` has the best composite score
     (`1.695`) in the 9-config matrix and wins paired composite deltas against
     all baselines. The strongest paired composite deltas are against
     `T_hf_lower_terminal` (`-0.346`, CI95 `[-0.643, -0.087]`),
     `T_rawhistory_terminal` (`-0.338`, CI95 `[-0.621, -0.070]`), and
     `T_allfreq_terminal` (`-0.325`, CI95 `[-0.518, -0.133]`).
     `T_swapped_terminal` still has slightly lower raw wait (`6.817` vs
     `6.917`), so this is positive performance validation but not clean
     dominance on every metric.

2. Promotion validation.
   - Done for the current synthetic trading validation: the tuned promotion setting
     (`threshold=0.00035`, `persistence_ratio=0.50`, `mid_gain=0.5`,
     `cooldown_s=600`, `regime_threshold=0.00003`, `min_age_s=0`,
     `activation_strength_threshold=0`, `startup_strength_age_s=0`,
     `startup_strength_threshold=0`, `adapt_gain=0.05`) makes `freq_hrl` the best Sharpe baseline and improves
     Sharpe against `no_promotion`.
   - Done: Promotion delay and post-shift cumulative PnL are reported in the
     validation output.
   - Done: a larger promotion recovery sweep was run and merged.
   - Done: a dedicated `promotion_recovery` reversal-shock validation now shows
     the recovery-tuned gate improving both headline and recovery metrics
     against `no_promotion` over 20 paired seeds: Sharpe delta `+0.430`
     (CI95 `[+0.315, +0.536]`), return delta `+0.0115`
     (CI95 `[+0.0087, +0.0145]`), post-shift-120 PnL delta `+0.00854`
     (CI95 `[+0.00622, +0.01090]`), and oracle-regime recovery-regret delta
     `-0.00839` (CI95 `[-0.01075, -0.00613]`).
   - Done for deterministic plan-curve control: promotion can force immediate
     upper replan and improves recovery metrics versus interval-only plan reuse.
   - Still partial: promotion-triggered replanning is not yet embedded in the
     learned PPO/off-policy runners or low-model process-noise adaptation.

3. Leakage regularization in learned rewards/losses.
   - Done for the shared core and trading harness: `CausalLeakageRewardShaper`
     computes causal leakage from online action effects and subtracts a scaled
     penalty from the learner-facing reward.
   - Done in the copied Transit runner path: `runner_v3.py` already applies
     lower drift and upper high-frequency penalties to lower/upper rewards when
     `leakage.enable` is true.
   - Done in the policy-gradient trading path: `pg_linear` now supports
     explicit policy-loss leakage penalties plus a Lagrange-style constraint
     multiplier driven by causal action-effect leakage.
   - Done in learned trading paths: `pg_linear` and `ac_linear` now expose
     separate `LowerLFDrift` policy-loss and Lagrange constraint controls.
   - Done in shared PPO: `TrajectoryBatch` now carries constraint costs and the
     clipped PPO objective supports a primal-dual constraint term. The trading
     constrained plan-PPO run cuts `LowerLFDrift` from `1.548` to `0.856`,
     with a large performance tradeoff.
   - Done in the deterministic trading control loop: an optional
     lower-drift-speed constraint directly boosts lower execution speed when
     the lower state drifts from the upper plan.
   - Current evidence: the leakage-regularized PG run reaches Sharpe `15.993`
     and return `0.2495` versus the refreshed unregularized PG run at Sharpe
     `15.915` and return `0.2487`; it also reduces turnover (`5.98` vs
     `6.41`) and upper HF action leakage (`UpperHFPower=0.000677` vs
     `0.000776`).
   - Current lower-drift evidence: the lower-drift-speed validation reduces
     `LowerLFDrift` from `1.821` to `0.916` while keeping Sharpe above the
     original default (`16.080` vs `16.062`). The lower-LF constrained AC run
     gives a smaller learned-policy improvement (`1.647` to `1.640`) while
     improving Sharpe (`15.528` to `15.619`).
   - Still partial: the strongest lower-drift reduction is now available in
     both deterministic control and PPO loss-level training, but the PPO
     constraint is a clear performance/leakage tradeoff and did not yet improve
     the Transit surrogate.

4. Reward and credit attribution are incomplete.
   - Done for the shared core and trading harness:
     `RewardAttributionAccumulator` logs low-frequency attributable cost,
     high-frequency attributable cost, leakage cost, promotion adaptation cost,
     and upper/lower credit summaries.
   - Done in the trading validation output: LF/HF/leakage/promotion cost columns
     are reported per baseline.
   - Still partial: copied Transit runner has frequency wait-credit diagnostics,
     but it does not yet use the shared attribution accumulator.
   - Existing diagnostics are only partly used for learned high/low policy credit.

5. Full baseline matrix.
   - Done in trading validation: `vanilla_rl`, `hrl_raw`, `raw_history`,
     `freq_single_policy`, `lf_upper_only`, `hf_lower_only`,
     `allfreq_alllayers`, `swapped`, `no_promotion`, `no_leakage`,
     `freq_hrl`.
   - Done for copied Transit configs: `T_rawhistory_terminal`,
     `T_lf_upper_terminal`, and `T_hf_lower_terminal` were added next to the
     existing nofreq/allfreq/swapped/nopromotion/noleakage/freqhrl configs.
   - Done: copied Transit matrix now has a 27-seed x 10-episode run for the
     nine configs, merged from two scheduler shards.

6. Diagnostics.
   - Implemented: `UpperHFPower`, `LowerLFDrift`, `FocusScore`.
   - Done in trading validation output: `PromotionDelay`, `ShockResponseTime`,
     balanced regime-promotion accuracy, precision/recall, and 120-bar recovery
     cost.
   - Done in copied Transit runner output: shock response, demand attribution,
     leakage ratios, and promotion summaries are present in diagnostics.
   - Done: `freq_hrl.experiments.trading.diagnostic_plots` generated seven
     figures under `transit_hrl/results/trading_diagnostic_plots`.
   - Still partial: these plots are generated by a separate experiment script,
     not by the main validation command automatically.

7. Phase 0 logging-only audit.
   - Done for the shared core and trading harness:
     `Phase0TraceLogger` writes schema-versioned JSONL records for `x_raw`,
     `x_bin`, `z_t`, `a_U`, `a_L`, plan curves, action effects, rewards, and
     domain entity IDs.
   - Done: `freq_hrl.experiments.trading.phase0_audit` replays logged causal
     bins and verifies offline reconstruction of frequency state. The current
     180-step audit passes with zero reconstruction error.
   - Still partial: copied Transit runner does not yet emit this shared Phase-0
     schema.

8. Config isolation tests.
   - Done: `freq_hrl.experiments.transit.config_isolation` resolves YAML
     `_extends`, diffs ablation configs against `T_freqhrl_terminal`, and
     verifies only allowed paths change.
   - Done: unit test `test_config_isolation.py` covers the audit.
   - Current generated audit passes for allfreq, swapped, nopromotion, and
     noleakage configs.

## Not Done: Required For Full Freq-HRL Claim

1. Learned-policy integration.
   - Partial but stronger: `freq_hrl.experiments.trading.policy_entry` plus the
     shared PPO trainer now support four learned policy paths. `linear` trains
     shared frequency-routing coefficients with cross-entropy search; the
     refreshed run reaches held-out Sharpe `15.419` and return `0.280`.
     `pg_linear`
     trains an on-policy Gaussian actor with REINFORCE over upper targets and
     lower execution speeds; the current held-out run reaches Sharpe `15.915`
     and return `0.249`, above the same held-out heuristic run (`15.663`
     Sharpe, `0.244` return). A leakage-constrained `pg_linear` variant reaches
     Sharpe `15.993` and return `0.2495` with lower turnover and lower upper-HF
     action leakage. `ac_linear` adds explicit upper/lower value functions and
     TD(0) bootstrapped actor updates; the current held-out run reaches return
     `0.2804`, objective `0.4436`, and leakage penalty `1.6475`. The shared
     `ppo_dual_actor_critic` path adds a domain-agnostic clipped-PPO/GAE
     trainer with separated upper/lower actors and critics; its current
     held-out result is Sharpe `15.402`, return `0.2351`, and
     `LowerLFDrift=1.716`.
   - Still missing: SAC/TD3-style off-policy actor-critic with replay,
     separate high/low replay buffers, plan-coefficient actions, and a real
     domain runner training loop. PPO exists now, but it is not yet the best
     performance path.

2. HighLevelPlanner and LowLevelController policy interfaces.
   - Done: `HighLevelPlanner`, `LowLevelController`, `HighLevelDecision`, and
     `LowLevelDecision` interfaces are implemented under `freq_hrl.policies`.
   - Done: trading heuristic planner/controller implementations exercise the
     interface and can be replaced by learned policies.

3. High-level plan curve deployment.
   - Done in copied Transit: `runner_v3` uses `TimetableCurvePlanner` in the
     actual dispatch/headway callback, with executable terminal launch plans,
     plan reuse, smoothness penalties, and plan diagnostics.
   - Done in shared core: `TimetableCurvePlanner` now delegates Bernstein basis,
     clipping, and smoothness semantics to the generic `BernsteinPlanCurve`.
   - Done in Quant synthetic validation: `freq_hrl` can route desired portfolio
     targets through `CausalPlanCurveState`; the current validation improves
     Sharpe and strongly reduces lower LF drift/leakage at a small return cost.
   - Done in shared PPO adapters: upper actors can optimize Bernstein plan
     coefficients directly for both Trading and Transit surrogate control.
   - Still partial: public-data evaluation does not yet expose the plan-curve
     switch, and the copied Transit simulator still uses hand-coded timetable
     replanning rather than shared PPO plan-coefficient training.

4. Cross-domain validation is partial.
   - Quant has synthetic validation, pressure testing, learned-policy
     validation, and Level-1 public daily-bar validation.
   - Transit now has a copied-runner 27-seed x 10-episode performance report
     on a 9-config matrix, not just tracker smoke coverage.
   - Done for shared training-core plumbing: Trading PPO and Transit surrogate
     PPO now use the same `freq_hrl.rl.train_dual_ppo` loop.
   - Still partial: the copied Transit runner still trains through its copied
     RESAC/runner stack; shared PPO is not yet embedded as the native trainer
     for that simulator.

5. Environment pressure-test matrix.
   - Done: pressure matrix now covers persistent shift, promotion recovery,
     stationary low-noise, stationary high-noise, localized burst, and OOD
     period.
   - Done: the matrix includes both conservative `freq_hrl` and
     `freq_hrl_recovery_tuned` so the recovery gate can be evaluated as a
     scenario-specific variant instead of silently replacing the robust default.
   - Current evidence: conservative `freq_hrl` wins persistent shift
     (`16.062` Sharpe), stationary high-noise (`-0.679`), and OOD period
     (`13.901`); `freq_hrl_recovery_tuned` wins promotion recovery (`21.133`)
     and localized burst (`2.312`); stationary low-noise is an effective tie
     where LF-only/no-promotion/frequency variants all report `8.440`.
   - Boundary: the recovery-tuned gate hurts stationary high-noise
     (`-1.788` vs conservative `-0.679`), so it should remain a
     scenario-specific promotion profile rather than the global default.

6. Required figures.
   - Done for trading diagnostics: raw/LF/HF signal decomposition, promotion
     gate timeline, ablation Sharpe bars, FocusScore scatter, pressure heatmap,
     promotion recovery scatter, and NoLeakage drift comparison are generated.
   - Still partial: plot generation is a separate command and is not yet
     automatically included in every experiment report.

7. Advanced encoders.
   - Done: Level-2 causal state-space encoder with uncertainty.
   - Done: Level-3 trailing-window causal Haar wavelet-style encoder.
   - Done: trading encoder ablation was run for EMA, Fourier, state-space, and
     Haar wavelet on the persistent-shift scenario.
   - Current evidence: EMA is still strongest in that ablation
     (Sharpe `16.062` vs Fourier `7.011`, state-space `8.931`, and Haar
     wavelet `12.712`), so the new encoders are not the source of the
     persistent-shift synthetic gains.
   - Done: public SPY/QQQ/IWM daily-bar encoder ablation now gives the Haar
     wavelet encoder the best Sharpe/return on the 1500-bar public CSV slice
     (`0.596` Sharpe, `0.7667` return) versus EMA (`0.406`, `0.3957`) and
     state-space (`0.066`, `-0.0214`).
   - No learnable wavelet or neural state-space encoder.
   - No PINN-constrained encoder.

8. Public/real market data experiments.
   - Done: Level-1 public ETF daily-bar validation path is implemented and run
     on SPY/QQQ/IWM CSVs covering 2016-06-01 through 2026-05-29. The refreshed
     EMA 1500-bar evaluation reports return `0.3957`, Sharpe `0.406`, and max
     drawdown `0.279`.
   - Done: public-data encoder ablation on the same SPY/QQQ/IWM slice shows
     Haar wavelet as the best Level-1 public-data encoder.
   - No Level-2 minute crypto/stock validation.
   - No Level-3 order-book/market-making validation.

## Current Blocking Evidence

The current validation package supports frequency routing, but the evidence is
now mixed rather than uniformly positive:

- In the default persistent-shift trading regime, `freq_hrl` beats
  `vanilla_rl`, `hrl_raw`, `raw_history`, `freq_single_policy`,
  `lf_upper_only`, `hf_lower_only`, `allfreq_alllayers`, `swapped`,
  `no_promotion`, and `no_leakage` on Sharpe. The edge over `lf_upper_only`
  is small (`16.062` vs `16.047`) and should not be overstated.
- In the wider pressure matrix, conservative `freq_hrl` wins persistent shift,
  stationary high-noise, and OOD period, while the explicit
  `freq_hrl_recovery_tuned` profile wins promotion recovery and localized
  burst. Stationary low-noise remains an effective tie. This makes the pressure
  result stronger than the earlier five-scenario matrix, but also shows that
  promotion tuning should be scenario-aware.
- The new Quant plan-curve path is a positive control-loop result on the
  default persistent-shift validation: it gives `freq_hrl` the best Sharpe
  (`16.106`) while lowering turnover and lower LF drift, but total return
  drops slightly (`0.2402` vs `0.2436` without plan smoothing).
- Adding the lower-drift speed constraint to the plan-curve control loop
  further reduces `LowerLFDrift` (`0.916`) and keeps `freq_hrl` best on Sharpe
  (`16.080`). This supports the lower-drift constraint mechanism, but the
  return tradeoff remains (`0.2403` vs original `0.2436`).
- Filter ablations found the remaining localized-burst loss comes from early
  promotion during EMA startup in one seed, not from the localized burst itself.
  A 130-minute hard warm-up or a startup activation-strength threshold can
  remove that loss, but those settings also reduce either OOD or stationary
  high-noise performance, so the default keeps them disabled.
- In copied Transit, `T_freqhrl_terminal` is best on the composite metric over
  27 seeds x 10 episodes. The paired report shows negative composite deltas
  against every baseline and statistically clearer improvements against
  `T_allfreq_terminal`, `T_hf_lower_terminal`, `T_lf_upper_terminal`, and
  `T_rawhistory_terminal`; the comparison to `T_swapped_terminal` remains close
  (`-0.037` composite, CI95 `[-0.198, +0.135]`) and `T_swapped_terminal` has
  slightly lower raw wait.
- Against `no_promotion`, tuned promotion improves persistent-shift Sharpe by
  about `+0.015` over 5 seeds. This conservative default is still chosen for
  the broad pressure matrix.
- A separate recovery-tuned promotion gate now supports the shock-recovery
  claim on a controlled persistent reversal shock: it triggers earlier
  (`16.6` bars mean delay vs `45.9` for the conservative default), improves
  post-shift-120 PnL by `+0.00854`, and reduces oracle-regime recovery regret
  by `-0.00839` against `no_promotion` over 20 paired seeds. This is evidence
  for the promotion mechanism.
- Promotion-triggered plan replanning now supports the stronger replan claim at
  the deterministic controller level: forced replan improves return by
  `+0.0014`, post-shift-120 PnL by `+0.00075`, recovery regret by `-0.00075`,
  and `LowerLFDrift` by `-0.0326` against interval-only plan reuse over
  10 paired seeds. It is still not yet a learned-runner replanning result.
- Leakage shaping is now in the online reward path, learned policy paths have
  explicit policy-loss / Lagrange-style constraints for total leakage and lower
  LF drift, and the deterministic lower-drift-speed constraint now cuts lower
  LF drift by about half in the main synthetic validation. The learned
  lower-drift constraint is positive but still much weaker than the
  control-level result.
- Reward attribution is now logged in the trading validation as LF cost, HF
  cost, leakage cost, promotion adaptation cost, upper credit, and lower credit.
- The synthetic decomposer ablation still shows that replacing EMA with Fourier,
  state-space, or Haar wavelet reduces Sharpe on the persistent-shift synthetic
  task. However, the public SPY/QQQ/IWM encoder ablation now gives the Haar
  wavelet encoder the best daily-bar Sharpe/return, so encoder sophistication
  has real-data upside but is not a universal default.
- HF speed and residual-order lower actions did not improve the default
  high-cost trading environment; the default now disables them, while keeping
  them as explicit sweep parameters for lower-cost execution settings.

Therefore the implementation should currently be described as:

```text
frequency-routed HRL protocol prototype with copied-Transit and trading validation
```

It should not yet be described as:

```text
fully validated, domain-general Frequency-Separated HRL
```

## Recommended Next Steps

1. Calibrate the new warm-up/activation-strength promotion filters on a larger
   out-of-sample matrix before deciding whether to enable them by default.
2. Investigate the Transit `swapped` raw-wait edge despite `freq_hrl` winning
   composite score.
3. Carry promotion-triggered replanning into learned PPO/off-policy runners and
   copied Transit native training, plus low-model process-noise adaptation.
4. Extend the shared PPO path with plan-coefficient actions and/or add
   off-policy SAC/TD3-style replay so the current control-level lower-drift
   constraint can be learned end-to-end instead of hand-coded.
5. Add Level-2 public minute data and Level-3 order-book/market-making validation.
6. Add automatic plot/report generation to the main validation commands.
7. Tune state-space and Haar wavelet encoder hyperparameters by domain, and add
   Level-2 minute data to check whether the public daily-bar Haar result
   survives at higher frequency.
