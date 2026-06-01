# Freq-HRL Paper Diagnostics

## Formal Objects

- Exogenous stream: causal bins `x_t` emitted by a domain adapter.
- Encoder: `z_t = (x_low, x_mid, x_high, energy, persistence)` with no access to future bins.
- Upper policy: low-frequency plan action `a_U`, optionally Bernstein coefficients over a horizon.
- Lower policy: high-frequency execution/control action `a_L` conditioned on the active upper plan.
- Promotion gate: persistent high-frequency residual detector that can promote regime evidence into the upper plan.
- Leakage: action-effect mismatch `UpperHFPower + LowerLFDrift`, computed causally from upper and lower effects.

## Diagnostic Bounds

For shaped rewards `r'_t = r_t - lambda * L_t`, cumulative shaped-return deviation from task return is bounded by `lambda * sum_t L_t`. With `L_t >= 0`, optimizing shaped return is a conservative lower bound on task return when leakage is treated as a constraint cost. The primal-dual PPO path makes this explicit by adding `eta * (cost_t - c)` to the clipped policy objective and updating `eta` from observed cost excess.

Promotion false positives and false negatives are controlled by the persistence window, residual threshold, regime buffer, and strength threshold. Lower thresholds reduce detection delay but raise stationary/high-noise false positives; the pressure matrix and promotion-replan validation should be reported together.

## Claim Matrix

| claim | status | metric | remaining gap |
|---|---|---|---|
| C1: frequency-separated HRL can share one training core | supported native loop | trading plan return=0.2991; transit composite=1.6950337432937435; native bridge=supported_interface U=20x4 L=43x1; native loop=supported_native_episode_loop, wait=6.9890 | Native shared-PPO episode loop exists; multi-seed native performance validation remains. |
| C2: high-level plan variables can be learned as curves | supported synthetic | plan-PPO return=0.2991, LowerLFDrift=1.5481 | Public-data and copied-Transit learned plan-coefficient training remain open. |
| C3: promotion should trigger replanning after persistent shocks | supported learned; native replan | return delta=0.0014, recovery regret delta=-0.0007; learned transit reward=+0.0047 [+0.0016, +0.0076], wait=-0.0079 [-0.0096, -0.0062], replans=+20.1000 [+14.9000, +25.5000]; native reward=+331.9682 [-4100.6044, +4771.6282], native replans=+54.8750 [+39.5000, +69.8844] | Native promotion-replan is now 8-seed positive-mixed on reward/wait and supported on replan count; learned native gate and larger off-policy theorem-grade validation remain. |
| C4: leakage can be constrained at loss level | supported | trading drift delta=-1.0782 [-1.2590, -0.8371]; return delta=-0.0003 [-0.0011, +0.0006] | Projected and raw lower-drift constraints are supported in surrogate diagnostics; native and real-data confirmation remain. |
| C5: advanced causal encoders can be swapped by domain | supported path | adaptive Sharpe=13.0749; neural Sharpe=6.8422; EMA Sharpe=16.0625 | Neural/PINN encoder path exists; larger cross-domain performance validation is still needed. |
| C6: public-data validation covers more than daily bars | supported path | best intraday encoder=adaptive_wavelet, Sharpe=-9.2200; best order-book encoder=state_space, Sharpe=299.9851 | Order-book adapter exists with deterministic CI fixture; larger real L2/L3 feeds remain for the strongest data claim. |
| C7: integrated native Transit Freq-HRL closes the copied-runner gap | supported | reward delta=+0.0565 [+0.0485, +0.0608]; wait delta=-0.0758 [-0.0815, -0.0651]; drift delta=-0.0199 [-0.0199, -0.0198]; native-loop samples=4970 | Supported on surrogate performance plus a native shared-PPO episode loop; still needs multi-seed native performance and real-demand validation. |
| C8: passenger waiting-time frequency credit improves control quality | supported | wait delta vs no-wait=-0.1251 [-0.1468, -0.1020] | Supported on the small surrogate gate; still needs larger seed coverage and native timetable validation. |
| C9: leakage constraints achieve no-tradeoff responsibility separation | supported | trading drift=supported, trading return=supported, transit drift=supported, transit reward=supported; raw drift trading=supported, raw drift transit=supported | Supported on surrogate Trading/Transit with raw-drift diagnostics; still needs native Transit and real-data confirmation. |
| C10: dynamic harmonic count-state demand estimator is competitive | supported | MSE delta=-0.7217 [-1.2633, -0.2617] | The count-state path now covers local OD-derived traces; larger real AFC/APC/GTFS demand feeds remain for the strongest claim. |

## Statistical Claim Gates

Deltas are `treatment - control`; `direction=decrease` means negative raw delta is the desired effect. Bootstrap intervals are paired by seed where possible.
No-tradeoff gates use a small noninferiority margin: 0.01 total-return for trading and 0.005 reward-mean for Transit.

| check | status | metric | n | delta CI95 | win rate | sign p |
|---|---|---|---:|---:|---:|---:|
| transit_full_reward_vs_base | supported | reward_mean | 3 | +0.0565 [+0.0485, +0.0608] | 1.00 | 0.2500 |
| transit_full_wait_vs_base | supported | wait_proxy | 3 | -0.0758 [-0.0815, -0.0651] | 1.00 | 0.2500 |
| transit_full_lower_lf_vs_base | supported | RawLowerLFDriftAbs | 3 | -0.0199 [-0.0199, -0.0198] | 1.00 | 0.2500 |
| transit_wait_credit_vs_no_wait | supported | wait_proxy | 3 | -0.1251 [-0.1468, -0.1020] | 1.00 | 0.2500 |
| transit_learned_promotion_reward_vs_interval | supported | reward_mean | 10 | +0.0047 [+0.0016, +0.0076] | 0.70 | 0.3438 |
| transit_learned_promotion_wait_vs_interval | supported | wait_proxy | 10 | -0.0079 [-0.0096, -0.0062] | 1.00 | 0.0020 |
| transit_learned_promotion_replans_vs_interval | supported | promotion_replan_count | 10 | +20.1000 [+14.9000, +25.5000] | 1.00 | 0.0020 |
| transit_learned_promotion_raw_lf_vs_interval | supported | RawLowerLFDriftAbs | 10 | -0.0003 [-0.0003, -0.0003] | 1.00 | 0.0020 |
| transit_native_promotion_reward_vs_interval | positive_mixed | ep_reward | 8 | +331.9682 [-4100.6044, +4771.6282] | 0.62 | 0.7266 |
| transit_native_promotion_wait_vs_interval | positive_mixed | avg_wait_min | 8 | -4.4798 [-13.1886, +0.8364] | 0.62 | 0.7266 |
| transit_native_promotion_replans_vs_interval | supported | upper_plan_decisions | 8 | +54.8750 [+39.5000, +69.8844] | 1.00 | 0.0078 |
| demand_nb_vs_fourier_mse | supported | mse | 17 | -0.7217 [-1.2633, -0.2617] | 0.82 | 0.0127 |
| demand_nb_vs_fourier_mae | supported | mae | 17 | -0.0494 [-0.0759, -0.0296] | 1.00 | 0.0000 |
| demand_nb_vs_fourier_poisson_nll_no_const | supported | poisson_nll_no_const | 17 | -0.1322 [-0.1620, -0.1047] | 1.00 | 0.0000 |
| trading_constraint_lower_lf | supported | LowerLFDrift | 5 | -1.0782 [-1.2590, -0.8371] | 1.00 | 0.0625 |
| trading_constraint_return_tradeoff | supported | total_return | 5 | -0.0003 [-0.0011, +0.0006] | 0.40 | 1.0000 |
| trading_constraint_raw_lower_lf | supported | RawLowerLFDriftAbs | 5 | -0.0000 [-0.0000, -0.0000] | 1.00 | 0.0625 |
| transit_constraint_lower_lf | supported | LowerLFDrift | 5 | -0.3090 [-0.3382, -0.2834] | 1.00 | 0.0625 |
| transit_constraint_reward_tradeoff | supported | reward_mean | 5 | +0.0316 [+0.0308, +0.0322] | 1.00 | 0.0625 |
| transit_constraint_raw_lower_lf | supported | RawLowerLFDriftAbs | 5 | -0.0192 [-0.0193, -0.0192] | 1.00 | 0.0625 |

## Paper Boundary

The current evidence supports a frequency-routed HRL protocol prototype with trading, surrogate Transit, and native Transit shared-PPO validation paths. It does not yet justify a fully validated domain-general algorithm claim because learned native promotion gates, larger real intraday/order-book and AFC/APC/GTFS feeds, and broader seed-level statistical tests remain open.
