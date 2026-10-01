# Stage74: crossed native response result

All root tasks t118693-t118700 and qualification t118701 finished with exit 0: 13,312 native episodes / 15,974,400 steps, 512 cross-execution paired-seed checks and 96 Stage73 geometry/frame reproductions. Root runtime was 526-535 seconds; recorded root RAM peaks were 4593-4732 MiB.

The frozen equal-root, 65536-draw Bonferroni-102 bootstrap gives four positive, three negative and 95 inconclusive endpoints. Positive endpoints are plus-base and plus-minus in the two period-100, zero-history native-MC cells; both minus-base endpoints are negative. Selected results are below; all 102 endpoints and root effects are retained in `results/pointmaze_crossed_direction_stage74_full_20261001_r1/qualification_compact.json`.

| Endpoint | Mean reward difference | Adjusted CI | Result |
|---|---:|---|---|
| 100, zero-history native MC, zero-residual plus-base | +0.5583 | [0.1295, 1.0306] | positive |
| 100, zero-history native MC, normal plus-base | +0.1852 | [0.0154, 0.3752] | positive |
| Same direction: normal minus zero-residual gain | -0.3731 | [-0.7263, -0.0860] | negative |
| 100, normal native MC: joint-history minus zero-history gain | -0.1828 | [-0.4654, 0.0015] | inconclusive |
| 100, native MC fitting/execution interaction | +0.2587 | [-0.1531, 0.7898] | inconclusive |
| 50, zero-history factored GAE, zero-residual plus-base | +0.3972 | [-0.0285, 0.7714] | inconclusive |

The period-100 zero-history MC gain is positive in 8/8 roots under zero-residual and 6/8 under normal execution; attenuation is negative in 8/8 roots. All 12 fitting contrasts and six interactions remain inconclusive. The Stage73 period-50 factored-GAE signal is not independently CI-supported here; retain both results without changing the radius, seeds or correction family.

These are sampled Stage55 teacher-clone probes: the source loader asserts an exactly zero upper mean head, not a trained joint actor. Descriptive base rewards are 959.761/804.808 at period 50 and 862.534/791.914 at period 100 (zero-residual/normal). The supported normal gain is about 0.0234% of its base reward. Source networks/Adam stayed unchanged; Stage67 HOLD remains.

## Next Step

Prioritize the initial sampled upper-residual execution and its reference/context feedback. The execution attenuation is supported for one fixed direction, while fitting-source and interaction attribution are unresolved. Inspect that mechanism before selecting a lower direction or resuming joint training.

## Limitations

Finite-radius stochastic responses on reused teacher-initialized development roots are not full joint learning, deployment/OOD validation or frequency-superiority evidence. Execution comparisons hold parameters, not target-state KL, fixed. Eight-root percentile-bootstrap uncertainty is limited.
