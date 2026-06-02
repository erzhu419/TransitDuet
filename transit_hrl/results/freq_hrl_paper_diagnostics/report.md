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
| C3: promotion should trigger replanning after persistent shocks | supported learned; native guarded-gate path | return delta=0.0014, recovery regret delta=-0.0007; learned transit reward=+0.0047 [+0.0016, +0.0076], wait=-0.0079 [-0.0096, -0.0062], replans=+20.1000 [+14.9000, +25.5000]; native reward=+0.0168 [-11.2749, +16.2435], native replans=+49.4167 [+36.4146, +62.0000]; native learned reward=+3.6935 [-12.3283, +17.8474], native learned score=-0.0068 [-0.0904, +0.0555], native learned gate replans=+1.5000 [+1.0813, +1.9167] | Native learned gate now preserves the closed native action policy and has CI-supported gate replans; episode reward/wait remain inconclusive, so larger off-policy/native training remains. |
| C4: leakage can be constrained at loss level | supported | trading drift delta=-1.0782 [-1.2590, -0.8371]; return delta=-0.0003 [-0.0011, +0.0006] | Projected and raw lower-drift constraints are supported in surrogate diagnostics; native and real-data confirmation remain. |
| C5: advanced causal encoders can be swapped by domain | supported path | adaptive Sharpe=13.0749; neural Sharpe=6.8422; EMA Sharpe=16.0625 | Neural/PINN encoder path exists; larger cross-domain performance validation is still needed. |
| C6: public-data validation covers more than daily bars | supported path | best intraday encoder=adaptive_wavelet, Sharpe=-9.2200; best order-book encoder=state_space, Sharpe=299.9851; best stress encoder=adaptive_wavelet under stale_book, Sharpe=243.4833; stress adaptive Sharpe=+0.2807 [-0.3942, +0.9004]; best L2 matching encoder=state_space latency=0, Sharpe=599.2800; L2 adaptive Sharpe=+1.0908 [-0.2300, +2.4692] | Order-book adapter, stress matrix, and L2 matching simulator exist; larger real L2/L3 feeds and exchange queue-priority modeling remain for the strongest data claim. |
| C7: integrated native Transit Freq-HRL closes the copied-runner gap | supported | reward delta=+0.0565 [+0.0485, +0.0608]; wait delta=-0.0758 [-0.0815, -0.0651]; drift delta=-0.0199 [-0.0199, -0.0198]; native-loop samples=4970 | Supported on surrogate performance plus a native shared-PPO episode loop and native AFC/APC-profile passenger generation; exact AFC/APC OD geometry and alighting-throughput support remain open. |
| C8: passenger waiting-time frequency credit improves control quality | supported native; real-demand positive-mixed | surrogate wait delta=-0.1251 [-0.1468, -0.1020]; native final wait delta=-8.5908 [-20.4276, -0.9074]; native score delta=+8.6724 [+1.0542, +20.4807]; native reward delta=+2037.3278 [-66.8598, +6165.1210]; real-demand objective delta=+1.8114 [+1.2826, +2.4539]; real-demand wait delta=-1.6741 [-2.2818, -1.1728]; native real-demand wait delta=-0.0830 [-0.2248, +0.0567] | Native wait-credit path is supported in the shared-PPO loop; native real-demand passenger loop has score/reward support but wait is not fully CI-supported and exact AFC/APC OD validation remains open. |
| C9: leakage constraints achieve no-tradeoff responsibility separation | supported | trading drift=supported, trading return=supported, transit drift=supported, transit reward=supported; raw drift trading=supported, raw drift transit=supported | Supported on surrogate Trading/Transit with raw-drift diagnostics; still needs native Transit and real-data confirmation. |
| C10: causal count-state demand estimators support Transit demand validation | supported afc+apc-calibrated+native-score | all-demand MSE delta=+4202.4659 [+1172.4726, +8434.5613]; AFC NB MSE delta=+16121.7347 [+6009.5751, +31280.0939]; AFC profile MSE delta=-7186235.1828 [-10935870.1938, -4496470.4747]; APC profile MSE delta=-5034.9993 [-8843.9598, -2369.0635]; real-control objective delta=+1.8114 [+1.2826, +2.4539]; native real-control score delta=+99.6725 [+62.3044, +137.0299]; native board-wait delta=-0.0833 [-0.2251, +0.0564]; native alighted delta=-4.8333 [-9.8333, -0.8333] | Real AFC/APC profiles now drive native passenger generation with boarding/alighting/onboard-load metrics; score is supported, board-wait is positive_mixed, alighted throughput is not_supported, and exact public OD/onboard occupancy feeds remain open. |

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
| transit_native_learned_gate_reward_vs_interval | inconclusive | ep_reward | 12 | +3.6935 [-12.3283, +17.8474] | 0.42 | 1.0000 |
| transit_native_learned_gate_wait_vs_interval | inconclusive | avg_wait_min | 12 | -0.0153 [-0.0530, +0.0060] | 0.17 | 0.4531 |
| transit_native_learned_gate_score_vs_interval | not_supported | score | 12 | -0.0068 [-0.0904, +0.0555] | 0.33 | 1.0000 |
| transit_native_learned_gate_replans_vs_interval | not_supported | upper_plan_decisions | 12 | +0.0000 [+0.0000, +0.0000] | 0.00 | 1.0000 |
| transit_native_learned_gate_gate_replans_vs_interval | supported | shared_ppo_gate_replans | 12 | +1.5000 [+1.0813, +1.9167] | 0.83 | 0.0020 |
| transit_native_wait_credit_final_wait_vs_no_wait | supported | final_avg_wait_min | 5 | -8.5908 [-20.4276, -0.9074] | 0.80 | 0.3750 |
| transit_native_wait_credit_mean_wait_vs_no_wait | supported | avg_wait_min_mean | 5 | -6.1649 [-16.1955, -0.4631] | 0.80 | 0.3750 |
| transit_native_wait_credit_reward_vs_no_wait | positive_mixed | final_ep_reward | 5 | +2037.3278 [-66.8598, +6165.1210] | 0.60 | 1.0000 |
| transit_native_wait_credit_score_vs_no_wait | supported | final_score | 5 | +8.6724 [+1.0542, +20.4807] | 0.80 | 0.3750 |
| transit_native_wait_credit_active_vs_no_wait | supported | freq_wait_upper_credit_std | 5 | +0.4500 [+0.4500, +0.4500] | 1.00 | 0.0625 |
| transit_real_demand_control_objective_vs_base | supported | control_objective | 6 | +1.8114 [+1.2826, +2.4539] | 1.00 | 0.0312 |
| transit_real_demand_control_reward_vs_base | supported | reward_mean | 6 | +1.6835 [+1.1721, +2.3034] | 1.00 | 0.0312 |
| transit_real_demand_control_wait_vs_base | supported | wait_proxy | 6 | -1.6741 [-2.2818, -1.1728] | 1.00 | 0.0312 |
| transit_real_demand_control_lower_lf_vs_base | supported | LowerLFDrift | 6 | -0.3026 [-0.3978, -0.1969] | 1.00 | 0.0312 |
| transit_real_demand_control_raw_lower_lf_vs_base | supported | RawLowerLFDriftAbs | 6 | -0.0185 [-0.0191, -0.0179] | 1.00 | 0.0312 |
| transit_native_real_demand_control_score_vs_interval | supported | control_score | 6 | +99.6725 [+62.3044, +137.0299] | 1.00 | 0.0312 |
| transit_native_real_demand_reward_vs_interval | supported | ep_reward | 6 | +98.7658 [+59.9997, +137.1515] | 1.00 | 0.0312 |
| transit_native_real_demand_wait_vs_interval | positive_mixed | avg_wait_min | 6 | -0.0830 [-0.2248, +0.0567] | 0.67 | 0.6875 |
| transit_native_real_demand_board_wait_vs_interval | positive_mixed | native_avg_board_wait_min | 6 | -0.0833 [-0.2251, +0.0564] | 0.67 | 0.6875 |
| transit_native_real_demand_alighted_vs_interval | not_supported | native_alighted_pax | 6 | -4.8333 [-9.8333, -0.8333] | 0.17 | 0.3750 |
| demand_nb_vs_fourier_mse | not_supported | mse | 89 | +4202.4659 [+1172.4726, +8434.5613] | 0.57 | 0.2031 |
| demand_nb_vs_fourier_mae | positive_mixed | mae | 89 | -0.0693 [-0.9110, +0.7818] | 0.61 | 0.0558 |
| demand_nb_vs_fourier_poisson_nll_no_const | not_supported | poisson_nll_no_const | 89 | +70.6436 [+45.4303, +98.7533] | 0.27 | 0.0000 |
| demand_afc_nb_vs_fourier_mse | not_supported | mse | 24 | +16121.7347 [+6009.5751, +31280.0939] | 0.08 | 0.0000 |
| demand_afc_nb_vs_fourier_mae | not_supported | mae | 24 | +3.8859 [+2.7080, +5.3789] | 0.00 | 0.0000 |
| demand_afc_nb_vs_fourier_poisson_nll_no_const | not_supported | poisson_nll_no_const | 24 | +168.7536 [+97.1531, +257.5680] | 0.12 | 0.0003 |
| demand_afc_profile_vs_fourier_mse | supported | mse | 24 | -7186235.1828 [-10935870.1938, -4496470.4747] | 1.00 | 0.0000 |
| demand_afc_profile_vs_fourier_mae | supported | mae | 24 | -1487.0751 [-1840.2423, -1211.6740] | 1.00 | 0.0000 |
| demand_afc_profile_vs_fourier_poisson_nll_no_const | supported | poisson_nll_no_const | 24 | -6612.4993 [-8039.1114, -5475.0716] | 1.00 | 0.0000 |
| demand_apc_nb_vs_fourier_mse | supported | mse | 24 | -377.7009 [-767.6740, -71.2334] | 0.71 | 0.0639 |
| demand_apc_nb_vs_fourier_mae | positive_mixed | mae | 24 | -0.9749 [-2.4697, +0.6107] | 0.75 | 0.0227 |
| demand_apc_nb_vs_fourier_poisson_nll_no_const | not_supported | poisson_nll_no_const | 24 | +51.4912 [+18.7151, +94.3915] | 0.12 | 0.0003 |
| demand_apc_profile_vs_fourier_mse | supported | mse | 24 | -5034.9993 [-8843.9598, -2369.0635] | 0.96 | 0.0000 |
| demand_apc_profile_vs_fourier_mae | supported | mae | 24 | -30.3136 [-41.5414, -20.9693] | 1.00 | 0.0000 |
| demand_apc_profile_vs_fourier_poisson_nll_no_const | supported | poisson_nll_no_const | 24 | -136.5008 [-180.4619, -100.6444] | 1.00 | 0.0000 |
| trading_constraint_lower_lf | supported | LowerLFDrift | 5 | -1.0782 [-1.2590, -0.8371] | 1.00 | 0.0625 |
| trading_constraint_return_tradeoff | supported | total_return | 5 | -0.0003 [-0.0011, +0.0006] | 0.40 | 1.0000 |
| trading_constraint_raw_lower_lf | supported | RawLowerLFDriftAbs | 5 | -0.0000 [-0.0000, -0.0000] | 1.00 | 0.0625 |
| transit_constraint_lower_lf | supported | LowerLFDrift | 5 | -0.3090 [-0.3382, -0.2834] | 1.00 | 0.0625 |
| transit_constraint_reward_tradeoff | supported | reward_mean | 5 | +0.0316 [+0.0308, +0.0322] | 1.00 | 0.0625 |
| transit_constraint_raw_lower_lf | supported | RawLowerLFDriftAbs | 5 | -0.0192 [-0.0193, -0.0192] | 1.00 | 0.0625 |
| order_book_depth_adaptive_wavelet_vs_ema_sharpe | positive_mixed | sharpe | 25 | +0.2807 [-0.3942, +0.9004] | 0.68 | 0.1078 |
| order_book_depth_adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 25 | -0.0000 [-0.0000, -0.0000] | 0.40 | 0.4244 |
| order_book_depth_adaptive_wavelet_vs_ema_max_drawdown | supported | max_drawdown | 25 | -0.0000 [-0.0000, -0.0000] | 0.68 | 0.1078 |
| order_book_depth_adaptive_wavelet_vs_ema_turnover | not_supported | turnover | 25 | +0.0011 [+0.0006, +0.0016] | 0.00 | 0.0000 |
| order_book_depth_neural_state_space_vs_ema_sharpe | not_supported | sharpe | 25 | -1.1349 [-2.4475, +0.0254] | 0.48 | 1.0000 |
| order_book_depth_neural_state_space_vs_ema_total_return | not_supported | total_return | 25 | -0.0000 [-0.0000, -0.0000] | 0.04 | 0.0000 |
| order_book_depth_neural_state_space_vs_ema_max_drawdown | supported | max_drawdown | 25 | -0.0000 [-0.0000, -0.0000] | 0.84 | 0.0009 |
| order_book_depth_neural_state_space_vs_ema_turnover | not_supported | turnover | 25 | +0.0055 [+0.0050, +0.0061] | 0.00 | 0.0000 |
| order_book_depth_state_space_vs_ema_sharpe | not_supported | sharpe | 25 | -71.3384 [-128.2308, -16.8018] | 0.40 | 0.4244 |
| order_book_depth_state_space_vs_ema_total_return | not_supported | total_return | 25 | -0.0003 [-0.0004, -0.0001] | 0.28 | 0.0433 |
| order_book_depth_state_space_vs_ema_max_drawdown | not_supported | max_drawdown | 25 | +0.0003 [+0.0001, +0.0004] | 0.28 | 0.0433 |
| order_book_depth_state_space_vs_ema_turnover | not_supported | turnover | 25 | +3.4668 [+3.4136, +3.5239] | 0.00 | 0.0000 |
| order_book_matching_adaptive_wavelet_vs_ema_sharpe | inconclusive | sharpe | 15 | +1.0908 [-0.2300, +2.4692] | 0.47 | 0.7744 |
| order_book_matching_adaptive_wavelet_vs_ema_total_return | supported | total_return | 15 | +0.0000 [+0.0000, +0.0000] | 0.47 | 0.7744 |
| order_book_matching_adaptive_wavelet_vs_ema_max_drawdown | supported | max_drawdown | 15 | -0.0000 [-0.0000, -0.0000] | 0.67 | 0.0386 |
| order_book_matching_adaptive_wavelet_vs_ema_avg_slippage_bps | inconclusive | avg_slippage_bps | 15 | -0.0000 [-0.0000, +0.0000] | 0.40 | 1.0000 |
| order_book_matching_adaptive_wavelet_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 15 | +0.0000 [+0.0000, +0.0000] | 0.00 | 1.0000 |
| order_book_matching_neural_state_space_vs_ema_sharpe | inconclusive | sharpe | 15 | +0.8548 [-0.8653, +2.4053] | 0.47 | 0.7744 |
| order_book_matching_neural_state_space_vs_ema_total_return | supported | total_return | 15 | +0.0000 [+0.0000, +0.0000] | 0.60 | 0.1460 |
| order_book_matching_neural_state_space_vs_ema_max_drawdown | supported | max_drawdown | 15 | -0.0000 [-0.0001, -0.0000] | 0.67 | 0.0386 |
| order_book_matching_neural_state_space_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 15 | +0.0000 [-0.0000, +0.0000] | 0.27 | 1.0000 |
| order_book_matching_neural_state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 15 | +0.0000 [+0.0000, +0.0000] | 0.00 | 1.0000 |
| order_book_matching_state_space_vs_ema_sharpe | supported | sharpe | 15 | +483.0914 [+407.1948, +559.5459] | 1.00 | 0.0001 |
| order_book_matching_state_space_vs_ema_total_return | supported | total_return | 15 | +0.0012 [+0.0006, +0.0018] | 0.87 | 0.0074 |
| order_book_matching_state_space_vs_ema_max_drawdown | supported | max_drawdown | 15 | -0.0012 [-0.0017, -0.0006] | 0.93 | 0.0010 |
| order_book_matching_state_space_vs_ema_avg_slippage_bps | positive_mixed | avg_slippage_bps | 15 | -0.0000 [-0.0000, +0.0000] | 0.53 | 1.0000 |
| order_book_matching_state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 15 | +0.0000 [+0.0000, +0.0000] | 0.00 | 1.0000 |

## Paper Boundary

The current evidence supports a frequency-routed HRL protocol prototype with trading, surrogate Transit, native Transit shared-PPO validation, public GTFS schedule-proxy data, real AFC station-hour passenger-demand paths, real APC route-boarding passenger-demand paths, real-demand shared-PPO control replay, native AFC/APC-profile passenger generation, calibrated causal profiles, order-book spread/depth/latency stress checks, and L2 matching simulation with CSV input support. It does not yet justify a fully validated domain-general algorithm claim because native learned-promotion reward/wait CIs, larger real intraday/L2-L3 order-book feeds, exchange queue-priority matching, APC onboard-load/alighting feeds, true OD demand feeds, and broader seed-level statistical tests remain open.
