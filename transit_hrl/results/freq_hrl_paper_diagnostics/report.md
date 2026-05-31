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
| C4: leakage can be constrained at loss level | supported with tradeoff | constrained LowerLFDrift=0.8560, return=0.1649 | Constraint trades off return/Sharpe and did not improve Transit surrogate drift. |
| C5: advanced causal encoders can be swapped by domain | mixed | adaptive Sharpe=13.0749; EMA Sharpe=16.0625 | Neural state-space and PINN-constrained encoders remain open. |
| C6: public-data validation covers more than daily bars | supported path | best intraday encoder=adaptive_wavelet, Sharpe=-9.2200 | Short Level-1 intraday slice only; no order book or execution simulator. |

## Paper Boundary

The current evidence supports a frequency-routed HRL protocol prototype with copied-Transit and trading validation. It does not yet justify a fully validated domain-general algorithm claim because copied Transit native training, larger intraday/order-book data, neural/PINN encoders, and broader statistical tests remain open.
