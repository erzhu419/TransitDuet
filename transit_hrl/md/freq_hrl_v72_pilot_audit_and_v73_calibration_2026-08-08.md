# Freq-HRL v7.2 Pilot Audit And v7.3 Calibration

Date: 2026-08-08

## Decision

The v7.2 advantage-critic pilot confirms that paired counterfactual prediction
contains regime information, but it does not support the selective-promotion
claim. No candidate advances to full HPO. v7.3 adds a role-separated,
checkpoint-specific bias calibration step before untouched tuning validation.

## Audited v7.2 evidence

- protocol: `full_method_support_only_hpo_v7_2`
- code revision: `e36bcde995215d2f394be0d7ff216e32e74e9425`
- cells: 18 valid of 18 expected
- independent optimizer/data replicates: `3001, 3011, 3023`
- support regimes: stationary low noise, stationary high noise, localized
  burst, and persistent shift
- budget: 8 iterations, 3 rollout roots, and 120 steps per episode
- OOD and promotion-recovery access: not loaded
- source identity: verified

All six candidates passed the learning gate. Their mean stress-minus-low
predicted-advantage lift ranged from `+0.0156` to `+0.0284`, and mean
prediction-target correlation ranged from `0.370` to `0.473`. The advantage
head therefore learned useful ordering information.

No candidate passed the mechanism gate. `v72_balanced_strict` was closest: two
of three replicates were selective, with mean decision accuracy `0.645` and
mean correlation `0.394`. Replicate 3023 nevertheless promoted on 83.1% of
stationary-low-noise opportunities and 80.2% of stress opportunities. Its
predicted advantage was shifted upward across regimes, while persistent stress
still had the larger mean prediction. This is an intercept-calibration failure,
not a complete loss of regime ranking.

## v7.3 calibration contract

For a frozen checkpoint, a dedicated support-only calibration set produces
paired critic predictions `A_hat` and counterfactual labels `A_cf`. The robust
checkpoint bias is

```text
b_hat = median(A_hat - A_cf).
```

Each candidate retains a pre-registered economic target margin `tau_target`.
The checkpoint-specific decision threshold is

```text
tau_decision = tau_target + b_hat,
promote(s) = 1[A_hat(s) >= tau_decision].
```

Thus calibration corrects only a scalar critic intercept. It cannot change
state ordering, use stress labels to set a false-positive quota, update model
weights, inspect tuning outcomes, or access OOD/confirmatory paths. The model
state hash is checked before and after calibration. Calibration artifacts
record every scenario/seed, transition count, prediction mean, target mean,
and residual median.

Prediction-side `tau_decision` and label-side `tau_target` remain distinct in
all accuracy diagnostics. This avoids incorrectly comparing economic labels
against a threshold that includes model bias.

## v7.3 seed firewall

Because v7.3 was designed after inspecting v7.2, every development role is
rotated again:

- training rollout roots: `170003, 170021, 170029`;
- promotion-calibration seeds: `180001, 180013, 180023`;
- checkpoint-validation seeds: `190027, 190031, 190051`;
- tuning-validation seeds: `200003, 200009, 200017, 200023, 200029`;
- diagnostic-pilot optimizer seeds: `5003, 5009, 5011`;
- final-HPO optimizer seeds: `6007, 6011, 6029, 6037, 6043`.

All four within-cell data roles must be pairwise disjoint. The final freeze
gate also rejects pilot optimizer reuse. Confirmatory seeds remain absent from
all development artifacts.

## Advancement gate

The v7.2 mechanism thresholds remain unchanged. At least 80% of independent
replicates must jointly satisfy bounded low-noise and stress promotion rates,
positive stress-minus-low rate and advantage lifts, paired-label alignment,
decision accuracy of at least 0.55, active HF/reference/residual paths, and
hard HF budget compliance. Positive utility cannot override a failed mechanism
condition.

## Paper boundary

v7.2 supports the narrow statement that a learned paired advantage head
captures some stress-dependent replanning value. It does not support selective
promotion. v7.3 remains algorithm-development evidence until its fresh-seed
pilot passes and is independently reproduced by the full HPO and confirmatory
protocols.
