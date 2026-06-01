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
| C1: frequency-separated HRL can share one training core | supported native loop | trading plan return=0.2991; transit composite=1.6950337432937435; native bridge=supported_interface U=20x4 L=43x1; native loop=supported_native_episode_loop, wait=6.9890; offpolicy native=supported_native_episode_loop, replay_updates=3 | Native shared-PPO episode loop exists; multi-seed native performance validation remains. |
| C2: high-level plan variables can be learned as curves | supported synthetic | plan-PPO return=0.2991, LowerLFDrift=1.5481 | Public-data and copied-Transit learned plan-coefficient training remain open. |
| C3: promotion should trigger replanning after persistent shocks | supported learned; native learned-gate score | return delta=0.0014, recovery regret delta=-0.0007; learned transit reward=+0.0047 [+0.0016, +0.0076], wait=-0.0079 [-0.0096, -0.0062], replans=+20.1000 [+14.9000, +25.5000]; native reward=+0.0168 [-11.2749, +16.2435], native replans=+49.4167 [+36.4146, +62.0000]; native learned reward=+11.1923 [-9.1302, +33.6636], native learned score=+0.0385 [+0.0047, +0.0752], native learned gate replans=+15.6667 [+11.0000, +20.4167] | Native learned gate runs end-to-end with CI-supported control score/gate replans, but episode reward remains positive-mixed; larger off-policy/native training remains. |
| C4: leakage can be constrained at loss level | supported | trading drift delta=-1.0782 [-1.2590, -0.8371]; return delta=-0.0003 [-0.0011, +0.0006] | Projected and raw lower-drift constraints are supported in surrogate diagnostics; native and real-data confirmation remain. |
| C5: advanced causal encoders can be swapped by domain | supported path | adaptive Sharpe=13.0749; neural Sharpe=6.8422; EMA Sharpe=16.0625 | Neural/PINN encoder path exists; larger cross-domain performance validation is still needed. |
| C6: public-data validation covers more than daily bars | supported path | best intraday encoder=adaptive_wavelet, Sharpe=-9.2200; best order-book encoder=state_space, Sharpe=299.9851 | Order-book adapter exists with deterministic CI fixture; larger real L2/L3 feeds remain for the strongest data claim. |
| C7: integrated native Transit Freq-HRL closes the copied-runner gap | supported | reward delta=+0.0565 [+0.0485, +0.0608]; wait delta=-0.0758 [-0.0815, -0.0651]; drift delta=-0.0199 [-0.0199, -0.0198]; native-loop samples=4970 | Supported on surrogate performance plus a native shared-PPO episode loop; still needs multi-seed native performance and real-demand validation. |
| C8: passenger waiting-time frequency credit improves control quality | supported native | surrogate wait delta=-0.1251 [-0.1468, -0.1020]; native final wait delta=-8.5908 [-20.4276, -0.9074]; native score delta=+8.6724 [+1.0542, +20.4807]; native reward delta=+2037.3278 [-66.8598, +6165.1210] | Native wait-credit path is supported in the shared-PPO loop; still needs real AFC/APC demand. |
| C9: leakage constraints achieve no-tradeoff responsibility separation | supported | trading drift=supported, trading return=supported, transit drift=supported, transit reward=supported; raw drift trading=supported, raw drift transit=supported | Supported on surrogate Trading/Transit with raw-drift diagnostics; still needs native Transit and real-data confirmation. |
| C10: dynamic harmonic count-state demand estimator is competitive | supported | MSE delta=-93.5937 [-139.1670, -48.3963] | The count-state path now covers public GTFS schedule proxies; true AFC/APC passenger-demand feeds remain for the strongest claim. |

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
| transit_native_promotion_reward_vs_interval | inconclusive | ep_reward | 12 | +0.0168 [-11.2749, +16.2435] | 0.25 | 0.1460 |
| transit_native_promotion_wait_vs_interval | not_supported | avg_wait_min | 12 | +0.0092 [-0.0373, +0.0716] | 0.50 | 0.5078 |
| transit_native_promotion_replans_vs_interval | supported | upper_plan_decisions | 12 | +49.4167 [+36.4146, +62.0000] | 1.00 | 0.0005 |
| transit_native_learned_gate_reward_vs_interval | positive_mixed | ep_reward | 12 | +11.1923 [-9.1302, +33.6636] | 0.58 | 0.7744 |
| transit_native_learned_gate_wait_vs_interval | inconclusive | avg_wait_min | 12 | -0.0066 [-0.0523, +0.0438] | 0.33 | 1.0000 |
| transit_native_learned_gate_score_vs_interval | supported | score | 12 | +0.0385 [+0.0047, +0.0752] | 0.58 | 0.1797 |
| transit_native_learned_gate_replans_vs_interval | supported | upper_plan_decisions | 12 | +4.3333 [+3.2500, +5.5000] | 1.00 | 0.0005 |
| transit_native_learned_gate_gate_replans_vs_interval | supported | shared_ppo_gate_replans | 12 | +15.6667 [+11.0000, +20.4167] | 1.00 | 0.0005 |
| transit_native_wait_credit_final_wait_vs_no_wait | supported | final_avg_wait_min | 5 | -8.5908 [-20.4276, -0.9074] | 0.80 | 0.3750 |
| transit_native_wait_credit_mean_wait_vs_no_wait | supported | avg_wait_min_mean | 5 | -6.1649 [-16.1955, -0.4631] | 0.80 | 0.3750 |
| transit_native_wait_credit_reward_vs_no_wait | positive_mixed | final_ep_reward | 5 | +2037.3278 [-66.8598, +6165.1210] | 0.60 | 1.0000 |
| transit_native_wait_credit_score_vs_no_wait | supported | final_score | 5 | +8.6724 [+1.0542, +20.4807] | 0.80 | 0.3750 |
| transit_native_wait_credit_active_vs_no_wait | supported | freq_wait_upper_credit_std | 5 | +0.4500 [+0.4500, +0.4500] | 1.00 | 0.0625 |
| demand_nb_vs_fourier_mse | supported | mse | 41 | -93.5937 [-139.1670, -48.3963] | 0.78 | 0.0004 |
| demand_nb_vs_fourier_mae | supported | mae | 41 | -1.8545 [-2.7563, -0.9781] | 0.88 | 0.0000 |
| demand_nb_vs_fourier_poisson_nll_no_const | not_supported | poisson_nll_no_const | 41 | +24.4245 [+13.9743, +36.8789] | 0.44 | 0.5327 |
| trading_constraint_lower_lf | supported | LowerLFDrift | 5 | -1.0782 [-1.2590, -0.8371] | 1.00 | 0.0625 |
| trading_constraint_return_tradeoff | supported | total_return | 5 | -0.0003 [-0.0011, +0.0006] | 0.40 | 1.0000 |
| trading_constraint_raw_lower_lf | supported | RawLowerLFDriftAbs | 5 | -0.0000 [-0.0000, -0.0000] | 1.00 | 0.0625 |
| transit_constraint_lower_lf | supported | LowerLFDrift | 5 | -0.3090 [-0.3382, -0.2834] | 1.00 | 0.0625 |
| transit_constraint_reward_tradeoff | supported | reward_mean | 5 | +0.0316 [+0.0308, +0.0322] | 1.00 | 0.0625 |
| transit_constraint_raw_lower_lf | supported | RawLowerLFDriftAbs | 5 | -0.0192 [-0.0193, -0.0192] | 1.00 | 0.0625 |

## Paper Boundary

The current evidence supports a frequency-routed HRL protocol prototype with trading, surrogate Transit, native Transit shared-PPO validation, and public GTFS schedule-proxy data paths. It does not yet justify a fully validated domain-general algorithm claim because native learned-promotion reward, larger real intraday/order-book feeds, true AFC/APC passenger-demand feeds, and broader seed-level statistical tests remain open.
