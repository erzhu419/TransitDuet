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
- A7: constrained updates use bounded nonnegative dual variables and bounded constraint samples.

## Theorems

### Theorem 1: Causal Frequency Features Are Nonanticipative

Statement: For every decision time t, the feature vector emitted by a causal Freq-HRL encoder is measurable with respect to the observations available up to t.

Assumptions:
- The domain adapter appends an exogenous bin only after that bin has occurred.
- The encoder update is a deterministic or seeded-random function of the previous encoder state and the current bin.
- The feature extractor does not use backward smoothing, centered windows, or future timestamps.

Proof: Use induction on the number of processed bins. The initial encoder state is fixed or seeded independently of future observations. If the state before bin k is a function only of bins 1 through k-1, then the next state is a function only of that state and bin k. Therefore the features after bin k are functions only of bins 1 through k. Mapping k to decision time t gives the nonanticipativity claim.

Limitation: This is an information-flow guarantee. It does not claim that a chosen encoder is statistically optimal for every domain.

Diagnostics: Causal encoder tests cover EMA, Fourier, state-space, Haar/adaptive wavelet, and neural/PINN state-space paths.

### Theorem 2: Leakage-Shaped Return Gap Is Budgeted

Statement: For shaped rewards r'_t = r_t - lambda L_t with lambda >= 0 and causal leakage cost L_t >= 0, the absolute episode-return gap is bounded by lambda sum_t L_t.

Assumptions:
- Leakage is computed from same-trajectory upper and lower action effects.
- The leakage multiplier is nonnegative.
- Task return and shaped return are evaluated on the same rollout.

Proof: Summing the shaped reward gives sum_t r'_t = sum_t r_t - lambda sum_t L_t. Since lambda and L_t are nonnegative, the shaped return is no larger than task return and the exact difference is lambda sum_t L_t. Any enforced leakage budget B therefore bounds the distortion by lambda B.

Limitation: The bound controls reward-shaping distortion and responsibility violations; it is not a guarantee that stronger leakage penalties are performance-neutral.

Diagnostics: Leakage matrices report drift reduction and no-tradeoff gates for Transit and Trading variants.

Numeric example: Example bound with lambda=0.30: 0.0870.

### Theorem 3: Stationary Promotion False Positives Are Exponentially Controlled

Statement: If stationary residual-threshold events have conditional probability p < rho and promotion requires a trailing-window event share of at least rho over n bins, then the false-promotion probability is at most exp(-2 n (rho - p)^2).

Assumptions:
- The gate uses only a finite causal residual-event window.
- Stationary residual events are bounded Bernoulli indicators with rate at most p.
- The detector promotes when the window mean exceeds rho.

Proof: The promotion statistic is the empirical mean of bounded event indicators in the trailing window. Under the stationary null, its expectation is at most p. Hoeffding's inequality bounds the probability that this mean exceeds rho by exp(-2 n (rho - p)^2).

Limitation: The bound is conservative and assumes a stationary null. It should be reported together with empirical promotion false-positive sweeps.

Diagnostics: Promotion sweep and persistent-stress recovery validations test the empirical tradeoff.

Numeric example: Example n=10, rho=0.35, p=0.10: 0.286505.

### Theorem 4: Persistent-Shock Promotion Delay Is Window-Bounded

Statement: If every residual event after a regime shift exceeds the promotion threshold, the causal trailing-window gate promotes within one full persistence window.

Assumptions:
- The gate updates every fixed interval.
- Promotion requires a finite number of positive residual events in the trailing window.
- After the shift, each new event in the window is positive.

Proof: After one full window, all entries in the trailing window are post-shift positive events, so the event share equals one and exceeds any rho <= 1. The implementation's conservative bound reports the full window duration, which avoids any future-looking detection.

Limitation: Real shocks can be intermittent. In that case the false-negative and delay behavior depends on the post-shift event rate and threshold.

Diagnostics: Persistent-stress native promotion runs report replan counts, wait deltas, and recovery metrics.

Numeric example: Example delay bound: 600.0s.

### Theorem 5: Hierarchical Wait-Credit Residual Bounds Attribution Error

Statement: Let c_t be total causal passenger-wait credit, and let c_t^U and c_t^L be the upper and lower frequency-attributed credits on the same rollout. The episode attribution error is bounded by sum_t |c_t - c_t^U - c_t^L|.

Assumptions:
- Total, upper, and lower credits are computed from the same causal rollout.
- The policy losses consume only credits available at their decision times.
- The validation harness logs or reconstructs the residual term.

Proof: At each step, the attribution mismatch is exactly the absolute residual |c_t - c_t^U - c_t^L|. Summing the nonnegative per-step mismatches over the episode gives the stated L1 upper bound on total credit-assignment error.

Limitation: Small residuals certify attribution consistency, not necessarily that the resulting learned policy globally improves wait time.

Diagnostics: Native wait-credit and real-demand control validations report reward/wait/alighting deltas and should keep residual columns in OD/onboard-load runs.

Numeric example: Example residual bound: 0.1000.

### Theorem 6: Paired CI Width Shrinks at the Seed-Count Rate

Statement: For paired seed/source deltas with empirical standard deviation s and n independent pairs, the normal-approximation confidence half-width is z s / sqrt(n).

Assumptions:
- Treatment and control are paired by seed or source window.
- The paired deltas have finite variance.
- The z value matches the reported two-sided confidence level.

Proof: The paired estimator is the sample mean of the deltas. Its standard error is s / sqrt(n). Multiplication by the normal critical value z gives the reported half-width.

Limitation: The statement is an evidence-width calculation. It does not remove bias from nonrepresentative stress regimes or public-data samples.

Diagnostics: The unified matrix records n_common and CI status for every paired claim.

Numeric example: Example s=0.18, n=36, z=1.96: 0.0588.

### Theorem 7: Projected Primal-Dual Leakage Updates Control Average Excess

Statement: For bounded projected dual variables and bounded constraint samples, the standard projected-subgradient bookkeeping term for average constraint excess is O(1 / sqrt(T)) when the dual step is chosen on the 1 / sqrt(T) scale.

Assumptions:
- The dual variable is projected onto a bounded nonnegative interval.
- Constraint samples are uniformly bounded.
- The actor update uses the current multiplier times the causal constraint excess.

Proof: Apply the standard projected subgradient inequality to the one-dimensional dual update. Summing over T steps and dividing by T yields the radius term divided by eta T plus eta times the squared gradient bound. Choosing eta proportional to 1 / sqrt(T) gives the stated average excess rate.

Limitation: This is a constraint-control argument for the multiplier path. It is not a global convergence theorem for nonconvex actor-critic training.

Diagnostics: Dual PPO validation reports leakage budget, multiplier direction, and no-tradeoff status.

Numeric example: Example average-violation bookkeeping term: 0.1250.

## Empirical Anchors

| check | status | delta CI95 |
|---|---|---:|
| transit_learned_promotion_wait | supported | -0.009634420896740811 to -0.006195855116323389 |
| native_learned_gate_reward | positive_mixed | -5.438260526315868 to 16.26581776315782 |
| native_learned_gate_wait | not_supported | -0.0028562500000000575 to 0.04719999999999998 |
| real_demand_control_objective | supported | 1.28256587126095 to 2.4538887158636626 |
| real_demand_control_wait | supported | -2.281750974403621 to -1.1727880039632244 |
| trading_leakage_constraint | supported | -1.2589940220658022 to -0.8371210591584954 |
| transit_leakage_constraint | supported | -0.33821777083336013 to -0.2833995091642844 |

## Boundary

These results formalize the Freq-HRL protocol claims. They do not replace large-scale performance validation: native Transit under true onboard-load/alighting/OD dynamics, learned native promotion reward/wait CIs, and deeper order-book feeds still need broader seed and data coverage.
