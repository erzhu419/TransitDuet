# Freq-HRL Theory Appendix

## Formal Setup

Freq-HRL assumes an endogenous state `z_t`, an exogenous time-series stream `x_t`, and a causal encoder `E_phi(x_<=t)` that emits low-frequency trend, middle-frequency regime buffer, high-frequency residual, uncertainty, energy, and persistence summaries.

The upper policy `pi_U` consumes low-frequency trend/forecast plus bounded high-frequency summaries and emits a plan action. The lower policy `pi_L` consumes the active upper plan, local endogenous state, and high/middle-frequency residual context and emits high-frequency control actions.

## Assumptions

- A1: the encoder reads only current and past exogenous bins.
- A2: the upper action remains active across multiple lower decisions unless a scheduled or promoted replan occurs.
- A3: leakage costs are nonnegative and computed causally from action effects.
- A4: under stationary noise, residual-threshold events are conditionally bounded by a Bernoulli rate p.
- A5: paired validation compares treatment/control on the same seed and source window.
- A6: frequency credit residuals are explicitly measurable from the same causal rollout.

## Theorem 1: Leakage-Shaped Return Bound

For shaped rewards `r'_t = r_t - lambda L_t`, where `L_t >= 0`, the absolute deviation between task return and shaped return over an episode is bounded by `lambda * sum_t L_t`. Therefore, enforcing a leakage budget controls the maximum reward-shaping distortion while penalizing responsibility violations.

Example bound with `lambda=0.30`: `0.0870`.

## Theorem 2: Stationary Promotion False-Positive Bound

If residual threshold events occur with stationary probability `p < rho`, and promotion requires a trailing-window event share of at least `rho`, Hoeffding's inequality gives `P(false promote) <= exp(-2 n (rho-p)^2)` for window length `n`.

Example `n=10`, `rho=0.35`, `p=0.10`: `0.286505`.

## Theorem 3: Persistent-Shock Detection Delay

If every residual event after a regime shift exceeds threshold, the causal trailing-window gate detects the shift after at most one full persistence window. This is conservative and avoids future leakage.

Example delay bound: `600.0s`.

## Theorem 4: Hierarchical Credit Residual Bound

Let `c_t` be the total causal wait credit and let `c_t^U + c_t^L` be the upper/lower frequency attribution used by the policy losses. The episode-level attribution error is bounded by `sum_t |c_t - c_t^U - c_t^L|`. When diagnostics keep this residual small, the learned objectives are close to the intended passenger-wait objective.

Example residual bound: `0.1000`.

## Theorem 5: Paired Mean Evidence Width

For paired seed/source deltas with empirical standard deviation `s` and `n` pairs, the normal-approximation half-width is `z s / sqrt(n)`. This gives an explicit target for the larger-seed native promotion and real-demand control validations: increasing independent paired seeds shrinks inconclusive CIs at the standard square-root rate.

Example `s=0.18`, `n=36`, `z=1.96`: `0.0588`.

## Empirical Anchors

| check | status | delta CI95 |
|---|---|---:|
| transit_learned_promotion_wait | supported | -0.009634420896740811 to -0.006195855116323389 |
| native_learned_gate_reward | inconclusive | -12.328347916665471 to 17.84738958333338 |
| native_learned_gate_wait | inconclusive | -0.0530041666666669 to 0.006000000000000116 |
| real_demand_control_objective | supported | 1.28256587126095 to 2.4538887158636626 |
| real_demand_control_wait | supported | -2.281750974403621 to -1.1727880039632244 |
| trading_leakage_constraint | supported | -1.2589940220658022 to -0.8371210591584954 |
| transit_leakage_constraint | supported | -0.33821777083336013 to -0.2833995091642844 |

## Boundary

These results formalize the Freq-HRL protocol claims. They do not replace large-scale performance validation: native Transit under true onboard-load/alighting/OD dynamics, learned native promotion reward/wait CIs, and deeper order-book feeds still need broader seed and data coverage.
