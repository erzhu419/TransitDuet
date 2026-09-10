# MuJoCo v22 Uniform Terminal-Reserve Confirmation Result

## Decision

All 288 frozen cells completed successfully and passed provenance and training
audit checks. The unchanged uniform-consistency candidate did not pass the
joint confirmation gate. V22 is therefore a negative confirmatory result, not
manuscript support for reward-preserving terminal-reserve consistency.

## Registered Outcome

Uniform consistency reduced correction relative to zero-consistency reserve in
all three environments:

| Environment | Reward delta | Reward wins | Component reduction | Total reduction | Total correction RMS |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | -3.34% [-12.80%, +6.92%] | 18/32 | 27.26% [12.44%, 38.80%] | 37.54% [21.09%, 49.80%] | 0.0930 |
| Hopper-v5 | -2.85% [-9.18%, +3.33%] | 17/32 | 31.15% [27.09%, 35.06%] | 36.83% [30.60%, 42.77%] | 0.2831 |
| Walker2d-v5 | +4.70% [-6.16%, +16.83%] | 19/32 | 27.25% [16.05%, 36.56%] | 56.57% [41.18%, 69.91%] | 0.0578 |

The pooled paired estimates were reward delta -0.50% [-4.28%, +3.59%],
component correction reduction 28.55% [23.62%, 32.55%], and total correction
reduction 43.65% [38.18%, 48.20%]. Pooled reward noninferiority and both
correction-reduction gates passed.

The joint result nevertheless failed three frozen requirements:

- environment-wise reward noninferiority failed in HalfCheetah and Hopper;
- both projected arms had minimum projection-convergence rates near 92.5%,
  below the registered validity requirement;
- Hopper candidate total correction RMS was 0.2831, above the 0.25 burden cap.

Uniform updates were active with unit weights, certificate violations were
zero, prefix-power budgets passed, and recursive fallback remained bounded.
The failure is therefore not a disabled training path or missing-result error.

## Consequence

V22 rejects the unchanged uniform candidate as a joint confirmatory mechanism.
Its strong correction reductions may be reported only as a negative-result
diagnostic. V21 and v22 roots are retired and cannot be reused to tune the
coefficient, schedule, convergence threshold, burden cap, or checkpoint rule.
Any continuation must use a new development panel and change the mechanism to
improve recursive projection convergence and reward preservation directly.
