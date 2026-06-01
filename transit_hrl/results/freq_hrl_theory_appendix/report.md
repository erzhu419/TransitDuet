# Freq-HRL Theory Appendix

## Formal Setup

Freq-HRL assumes an endogenous state `z_t`, an exogenous time-series stream `x_t`, and a causal encoder `E_phi(x_<=t)` that emits low-frequency trend, middle-frequency regime buffer, high-frequency residual, uncertainty, energy, and persistence summaries.

The upper policy `pi_U` consumes low-frequency trend/forecast plus bounded high-frequency summaries and emits a plan action. The lower policy `pi_L` consumes the active upper plan, local endogenous state, and high/middle-frequency residual context and emits high-frequency control actions.

## Assumptions

- A1: the encoder reads only current and past exogenous bins.
- A2: the upper action remains active across multiple lower decisions unless a scheduled or promoted replan occurs.
- A3: leakage costs are nonnegative and computed causally from action effects.
- A4: under stationary noise, residual-threshold events are conditionally bounded by a Bernoulli rate p.

## Theorem 1: Leakage-Shaped Return Bound

For shaped rewards `r'_t = r_t - lambda L_t`, where `L_t >= 0`, the absolute deviation between task return and shaped return over an episode is bounded by `lambda * sum_t L_t`. Therefore, enforcing a leakage budget controls the maximum reward-shaping distortion while penalizing responsibility violations.

Example bound with `lambda=0.30`: `0.0870`.

## Theorem 2: Stationary Promotion False-Positive Bound

If residual threshold events occur with stationary probability `p < rho`, and promotion requires a trailing-window event share of at least `rho`, Hoeffding's inequality gives `P(false promote) <= exp(-2 n (rho-p)^2)` for window length `n`.

Example `n=10`, `rho=0.35`, `p=0.10`: `0.286505`.

## Theorem 3: Persistent-Shock Detection Delay

If every residual event after a regime shift exceeds threshold, the causal trailing-window gate detects the shift after at most one full persistence window. This is conservative and avoids future leakage.

Example delay bound: `600.0s`.

## Empirical Anchors

| check | status | delta CI95 |
|---|---|---:|
| transit_learned_promotion_wait | supported | -0.009634420896740811 to -0.006195855116323389 |
| native_learned_gate_reward | positive_mixed | -15.637375000000247 to 105.49962500000032 |
| trading_leakage_constraint | supported | -1.2589940220658022 to -0.8371210591584954 |
| transit_leakage_constraint | supported | -0.33821777083336013 to -0.2833995091642844 |

## Boundary

These results formalize the Freq-HRL protocol claims. They do not replace large-scale performance validation: native Transit, real AFC/APC/GTFS demand, and deeper order-book feeds still need broader seed and data coverage.
