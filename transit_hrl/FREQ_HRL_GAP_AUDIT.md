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
- Trading pressure-test matrix across five synthetic market regimes.
- Promotion recovery sweep with sharded scheduler execution and merge output.
- Trading decomposer ablation across causal EMA, Fourier, state-space, and
  trailing-window Haar wavelet encoders.
- Public Level-1 ETF daily-bar validation path using local CSV inputs.
- Diagnostic plot generation for signal decomposition, promotion, ablations,
  FocusScore, NoLeakage drift comparison, pressure matrix, and promotion
  recovery.
- 27-seed x 10-episode copied-Transit ablation validation on the 9-config matrix.
- Unit/smoke tests for causality, router masks, promotion, leakage, plan curves, stream adapter, domain trackers, and validation harness.

## MVP-Critical Checklist

1. Real FreqTransitDuet validation.
   - Done: `freq_transitduet` contains a copied Transit runner/env under `transit_hrl`.
   - Done: copied `frequency` entry point is bridged to `TransitFrequencyTracker` from the shared core.
   - Done: a 27-seed x 10-episode validation was run for the 9-config Transit matrix.
   - Current evidence: `T_freqhrl_terminal` has the best composite score
     (`1.695`) in the merged 9-config summary, but `T_swapped_terminal` has
     slightly lower raw wait (`6.817` vs `6.917`), so this is positive but not
     a clean dominance result on every metric.

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
   - Still partial: the best merged sweep setting improves headline Sharpe
     (`+0.230`) and total return (`+0.0032`) against `no_promotion`, but its
     immediate post-shift 120-bar PnL delta is still slightly negative
     (`-0.00015`), so shock-recovery tuning remains open.
   - Promotion does not yet trigger high-level replanning or low-model process-noise adaptation in a learned runner.

3. Leakage is not integrated into learned rewards/losses.
   - Done for the shared core and trading harness: `CausalLeakageRewardShaper`
     computes causal leakage from online action effects and subtracts a scaled
     penalty from the learner-facing reward.
   - Done in the copied Transit runner path: `runner_v3.py` already applies
     lower drift and upper high-frequency penalties to lower/upper rewards when
     `leakage.enable` is true.
   - Still partial: leakage is reward-shaped into Bellman targets, not yet added
     as an explicit differentiable policy-loss regularizer.

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
   - Partial but stronger: `freq_hrl.experiments.trading.policy_entry` now
     supports two learned policy paths. `linear` trains shared frequency-routing
     coefficients with cross-entropy search; the refreshed run reaches held-out
     Sharpe `15.419` and return `0.280`. `pg_linear` trains an on-policy
     Gaussian actor with REINFORCE over upper targets and lower execution
     speeds; the current held-out run reaches Sharpe `15.915` and return
     `0.249`, above the same held-out heuristic run (`15.663` Sharpe,
     `0.244` return).
   - Still missing: actual SAC/PPO/TD3-style actor-critic implementation with
     replay/advantage estimators and explicit lower/upper value functions.

2. HighLevelPlanner and LowLevelController policy interfaces.
   - Done: `HighLevelPlanner`, `LowLevelController`, `HighLevelDecision`, and
     `LowLevelDecision` interfaces are implemented under `freq_hrl.policies`.
   - Done: trading heuristic planner/controller implementations exercise the
     interface and can be replaced by learned policies.

3. High-level plan curve is not deployed in real runners.
   - Generic Bernstein curve exists.
   - Transit timetable/headway spline and Quant target portfolio curve are not yet connected to training.

4. Cross-domain validation is partial.
   - Quant has synthetic validation.
   - Transit has a tracker adapter only.
   - The same core has not yet been used in two real training/evaluation loops.

5. Environment pressure-test matrix.
   - Done: pressure matrix now covers persistent shift, stationary low-noise,
     stationary high-noise, localized burst, and OOD period.
   - Current evidence is mixed but improved after conservative promotion and
     HF-utility gating: `freq_hrl` wins persistent shift, stationary high-noise,
     and OOD period; it ties stationary low-noise and remains slightly behind
     `lf_upper_only` on localized burst by Sharpe (`2.301` vs `2.306`).
   - Additional filter probes show that hard warm-up or high activation-strength
     gates can eliminate the localized-burst gap, but they also reduce OOD or
     stationary high-noise gains, so they remain sweep parameters rather than
     defaults.

6. Required figures.
   - Done for trading diagnostics: raw/LF/HF signal decomposition, promotion
     gate timeline, ablation Sharpe bars, FocusScore scatter, pressure heatmap,
     promotion recovery scatter, and NoLeakage drift comparison are generated.
   - Still partial: plot generation is a separate command and is not yet
     automatically included in every experiment report.

7. Advanced encoders are missing.
   - Done: Level-2 causal state-space encoder with uncertainty.
   - Done: Level-3 trailing-window causal Haar wavelet-style encoder.
   - Done: trading encoder ablation was run for EMA, Fourier, state-space, and
     Haar wavelet on the persistent-shift scenario.
   - Current evidence: EMA is still strongest in that ablation
     (Sharpe `16.062` vs Fourier `7.011`, state-space `8.931`, and Haar
     wavelet `12.656`), so the new encoders are implemented baselines, not a
     performance upgrade yet.
   - No learnable wavelet or neural state-space encoder.
   - No PINN-constrained encoder.

8. Public/real market data experiments.
   - Done: Level-1 public ETF daily-bar validation path is implemented and run
     on SPY/QQQ/IWM CSVs covering 2016-05-31 through 2026-05-29. The current
     1500-bar evaluation reports return `0.639`, Sharpe `0.558`, and max
     drawdown `0.210`.
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
- In the wider pressure matrix, `freq_hrl` now wins persistent shift,
  stationary high-noise, and OOD period; it ties stationary low-noise and is
  still slightly behind `lf_upper_only` on localized burst. The regime-buffer
  filter reduced the localized gap from about `-0.016` Sharpe to about
  `-0.005`.
- Filter ablations found the remaining localized-burst loss comes from early
  promotion during EMA startup in one seed, not from the localized burst itself.
  A 130-minute hard warm-up or a startup activation-strength threshold can
  remove that loss, but those settings also reduce either OOD or stationary
  high-noise performance, so the default keeps them disabled.
- In copied Transit, `T_freqhrl_terminal` is best on the composite metric over
  27 seeds x 10 episodes, but `T_swapped_terminal` has slightly lower raw wait.
- Against `no_promotion`, tuned promotion improves persistent-shift Sharpe by
  about `+0.015` over 5 seeds. This is more conservative than the earlier
  setting but avoids more false promotion in pressure tests.
- The larger promotion sweep improves headline Sharpe/return further, but the
  best headline setting is still slightly worse on immediate post-shift 120-bar
  cumulative PnL, so promotion is not yet a fully optimized shock-recovery
  mechanism.
- Leakage shaping is now in the online reward path. The synthetic trading
  report shows nonzero leakage reward penalties for leakage-enabled baselines
  and zero penalty for `no_leakage`.
- Reward attribution is now logged in the trading validation as LF cost, HF
  cost, leakage cost, promotion adaptation cost, upper credit, and lower credit.
- The first decomposer ablation shows that replacing EMA with Fourier,
  state-space, or Haar wavelet currently reduces Sharpe on the persistent-shift
  synthetic task, so encoder sophistication should not be claimed as the source
  of the current gains.
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
3. Continue promotion tuning specifically for persistent-shift recovery, or add
   explicit high-level replanning in learned runners.
4. Replace the lightweight linear policy search with actual learned policy training, then
   decide whether reward-shaped leakage is sufficient or explicit policy-loss
   regularization is needed.
5. Add Level-2 public minute data and Level-3 order-book/market-making validation.
6. Add automatic plot/report generation to the main validation commands.
7. Tune state-space and Haar wavelet encoder hyperparameters, or keep them as
   causal ablation baselines rather than default methods.
