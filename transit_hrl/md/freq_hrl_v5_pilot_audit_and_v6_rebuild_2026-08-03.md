# Freq-HRL v5 Pilot Audit And v6 Rebuild

Date: 2026-08-03

## Decision

The v5 pilot is a valid validation-only implementation experiment, but it is
not eligible to freeze the final paper configuration. Freq-HRL remains in
algorithm-development mode. Held-out test seeds have not been loaded.

The next version must correct the experimental unit before increasing the seed
count. In v5, each stress scenario trains a different policy. In particular,
the policy reported under `ood_period` is trained on `ood_period`; it is
therefore a stress-specific result, not an out-of-distribution generalization
result. v6 will train one policy on a declared support mixture and evaluate the
same frozen policy across every validation regime. `ood_period` is excluded
from all parameter updates and checkpoint selection inputs that represent the
training distribution.

## Audited v5 pilot

- protocol: `full_method_nested_hpo_v3`
- code revision: `0c65097c9b86b900c22caf62144fb801532978cb`
- cells: 540 valid of 540 expected
- training replicates: 3
- tuning scenarios: `stationary_low_noise`, `persistent_shift`, `ood_period`
- tuning seeds per cell: 5
- source identity: verified for every cell
- held-out accesses: 0
- task-credit reconstruction error: at numerical precision

The validation-selected full candidate was
`v5_u3_l3_h1_p3_activegate` with mean utility `0.01332044` and replicate-level
bootstrap lower bound `0.01057506`. This is stronger than the PPO and generic
HRL controls in the pilot, but weaker than the validation-selected TD3 control
(`0.02263561`, lower bound `0.01836279`). The comparison is exploratory because
the pilot has only three independent training replicates and trains one model
per scenario.

## Mechanism audit

### Promotion

For the same full-method candidate, full minus no-promotion utility was:

| scenario | paired tuning rows | mean utility delta | full wins |
| --- | ---: | ---: | ---: |
| stationary low noise | 15 | +0.00000978 | 6 |
| persistent shift | 15 | +0.02353872 | 15 |
| OOD period | 15 | -0.00712176 | 1 |

The gate replanned an average of 4.6 times in stationary low noise, 7.0 times
in persistent shift, and 2.07 times in OOD period. Its action rate was 0.554,
1.000, and 0.225 respectively. The mechanism is active and useful under the
persistent-shift generator, but it lacks an acceptable false-positive/reward
boundary. The current gate reward is absolute task return over transitions of
different durations. That can reward a promotion merely because it owns more
positive-return primitive steps. v6 must instead credit the gate with the
incremental value of the replacement plan relative to the still-executable old
plan, net of replan cost.

### High-frequency lower controller

The same-policy zero-residual intervention for the selected full candidate had
mean return delta `-9.26e-7` over 45 paired tuning rows, with 26 positive rows.
The direct effect is economically negligible in this three-scenario pilot.
This pilot omits `localized_burst`, the registered regime with materially
predictive high-frequency residuals, so it cannot establish or reject the HF
claim. v6 must expose a causal historical HF-predictability summary to the
mixed-regime lower policy and require a positive localized-burst intervention
effect before mechanism eligibility.

### Leakage

For the selected candidate, full minus no-leakage mean utility was
`-0.00034468`. Mean absolute lower LF power was `3.7078e-6` for full and
`3.6834e-6` for no-leakage. The regularizer therefore incurred a performance
cost without reducing the registered action-effect endpoint.

The present online cost is a scale-free ratio whose denominator is the current
effect power. It can be near one even when the absolute drift is negligible,
and the dual variable starts positive before any budget violation. v6 will use
a fixed, preregistered RMS action-effect budget. The policy cost is the budget
ratio/excess, while the original spectral ratio remains a diagnostic. This
makes zero or economically negligible drift inexpensive and activates the dual
only for a real constraint violation.

## v6 non-negotiable protocol

1. One trained checkpoint per `(variant, candidate, training replicate)`, not
   one checkpoint per stress scenario.
2. Training and checkpoint selection use a declared mixed support process.
   OOD-period paths never update parameters.
3. Tuning evaluates the same checkpoint on all registered regimes with
   disjoint seeds. Held-out seeds remain inaccessible to HPO.
4. Primary one-factor ablations keep the full architecture and hidden width;
   a mechanism is masked without reallocating its parameters to another level.
5. Promotion eligibility includes stationary false-positive rate, stress
   recall, positive incremental-plan credit, and bounded replan rate.
6. HF eligibility includes nonzero action sensitivity and a positive paired
   localized-burst intervention effect, not merely a non-dead code path.
7. Leakage remains eligible only if it lowers the registered absolute drift
   endpoint without violating a preregistered utility noninferiority margin.
8. Final confirmatory evaluation retrains the frozen configuration on the same
   support distribution using independent optimizer/data replicates, then
   evaluates each checkpoint exactly once on untouched held-out paths.

## Paper boundary

Until v6 passes mixed-regime validation and confirmatory gates, the valid claim
is: "Freq-HRL implements exact asynchronous frequency-responsibility credit
and is undergoing mixed-regime learned-policy validation." The project must not
claim OOD policy generalization, HF benefit, leakage no-tradeoff, or universal
superiority over TD3 from the v5 pilot.
