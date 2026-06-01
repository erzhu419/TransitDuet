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
| C1: frequency-separated HRL can share one training core | partial | trading plan return=0.2991; transit composite=1.6950337432937435 | Copied Transit native simulator still uses copied RESAC runner. |
| C2: high-level plan variables can be learned as curves | supported synthetic | plan-PPO return=0.2991, LowerLFDrift=1.5481 | Public-data and copied-Transit learned plan-coefficient training remain open. |
| C3: promotion should trigger replanning after persistent shocks | supported deterministic | return delta=0.0014, recovery regret delta=-0.0007 | Not yet embedded in learned PPO/off-policy runner. |
| C4: leakage can be constrained at loss level | supported with tradeoff | trading drift delta=-0.6920 [-0.7946, -0.5380]; return delta=-0.1342 [-0.1773, -0.0929] | Constraint trades off return/Sharpe and did not improve Transit surrogate drift. |
| C5: advanced causal encoders can be swapped by domain | mixed | adaptive Sharpe=13.0749; EMA Sharpe=16.0625 | Neural state-space and PINN-constrained encoders remain open. |
| C6: public-data validation covers more than daily bars | supported path | best intraday encoder=adaptive_wavelet, Sharpe=-9.2200 | Short Level-1 intraday slice only; no order book or execution simulator. |
| C7: integrated native Transit Freq-HRL closes the copied-runner gap | supported | reward delta=+0.0350 [+0.0288, +0.0391]; wait delta=-0.0596 [-0.0659, -0.0501]; drift delta=-0.0011 [-0.0044, +0.0054] | Supported on the small Transit surrogate gate; still needs larger native Transit and real-demand validation. |
| C8: passenger waiting-time frequency credit improves control quality | supported | wait delta vs no-wait=-0.1250 [-0.1466, -0.1021] | Supported on the small surrogate gate; still needs larger seed coverage and native timetable validation. |
| C9: leakage constraints achieve no-tradeoff responsibility separation | not_supported | trading drift=supported, trading return=not_supported, transit drift=not_supported, transit reward=not_supported | The constraint path can reduce drift in trading, but no-tradeoff is not supported and Transit drift is not improved. |
| C10: dynamic harmonic count-state demand estimator is competitive | supported | MSE delta=-2.2628 [-2.9776, -1.6011] | The count-state path is present; it must beat or match Fourier on larger real Transit demand data before becoming a headline claim. |

## Statistical Claim Gates

Deltas are `treatment - control`; `direction=decrease` means negative raw delta is the desired effect. Bootstrap intervals are paired by seed where possible.

| check | status | metric | n | delta CI95 | win rate | sign p |
|---|---|---|---:|---:|---:|---:|
| transit_full_reward_vs_base | supported | reward_mean | 3 | +0.0350 [+0.0288, +0.0391] | 1.00 | 0.2500 |
| transit_full_wait_vs_base | supported | wait_proxy | 3 | -0.0596 [-0.0659, -0.0501] | 1.00 | 0.2500 |
| transit_full_lower_lf_vs_base | positive_mixed | LowerLFDrift | 3 | -0.0011 [-0.0044, +0.0054] | 0.67 | 1.0000 |
| transit_wait_credit_vs_no_wait | supported | wait_proxy | 3 | -0.1250 [-0.1466, -0.1021] | 1.00 | 0.2500 |
| demand_nb_vs_fourier_mse | supported | mse | 5 | -2.2628 [-2.9776, -1.6011] | 1.00 | 0.0625 |
| demand_nb_vs_fourier_mae | supported | mae | 5 | -0.1043 [-0.1604, -0.0542] | 1.00 | 0.0625 |
| demand_nb_vs_fourier_poisson_nll_no_const | supported | poisson_nll_no_const | 5 | -0.0975 [-0.1315, -0.0671] | 1.00 | 0.0625 |
| trading_constraint_lower_lf | supported | LowerLFDrift | 5 | -0.6920 [-0.7946, -0.5380] | 1.00 | 0.0625 |
| trading_constraint_return_tradeoff | not_supported | total_return | 5 | -0.1342 [-0.1773, -0.0929] | 0.00 | 0.0625 |
| transit_constraint_lower_lf | not_supported | LowerLFDrift | 5 | +0.0001 [+0.0001, +0.0001] | 0.00 | 0.0625 |
| transit_constraint_reward_tradeoff | not_supported | reward_mean | 5 | -0.0002 [-0.0003, -0.0000] | 0.20 | 0.3750 |

## Paper Boundary

The current evidence supports a frequency-routed HRL protocol prototype with copied-Transit and trading validation. It does not yet justify a fully validated domain-general algorithm claim because copied Transit native training, larger intraday/order-book data, neural/PINN encoders, and broader statistical tests remain open.
