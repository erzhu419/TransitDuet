# MuJoCo v14.22 action-cost critic preflight outcome

## Frozen result

The preregistered all-environment gate failed with two of three environments
supported. Scheduler tasks `t84742-t84744` ran from source revision
`a3712eed013164ed8927d2b3a7f441aa8a5eb335` on `node003`. Every task completed
normally in approximately 77 seconds; peak resident memory was 8.6-8.7 GB.

| Environment | Upper holdout R2 | Lower holdout R2 | Upper action-permutation gain | Lower action-permutation gain | Critic gate | Validation merit change | Reward violations | Result |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| HalfCheetah-v5 | 0.321 | -0.138 | -0.0014 | 0.0008 | fail | not evaluated | not evaluated | not supported |
| Hopper-v5 | 0.193 | 0.326 | 0.0827 | 0.0270 | pass | -1.654% | 0 | supported |
| Walker2d-v5 | 0.080 | 0.147 | 0.0753 | 0.0112 | pass | -0.094% | 0 | supported |

Hopper selected actor RMS step `1e-6`; all five design steps were eligible and
the independent validation reduction was 1.654%. Walker selected `3e-8`; two
design steps were eligible and the independent reduction was 0.094%. Both
preserved the frozen reward guard.

## Mechanism decision

The action-conditioned critic direction is viable on Hopper and Walker, but the
v14.22 mechanism does not advance into the shared core because the frozen gate
required all three environments. HalfCheetah was rejected before actor search:
its lower cumulative-cost critic had negative holdout R2, negligible action
permutation gain, and median ensemble-gradient cosine `-0.0006`. Relaxing the
gate would turn an unidentified action direction into a policy update.

The failure is specific to the long-horizon lower target. All HalfCheetah lower
targets were positive, but their standard deviation was only 0.358 and the
ensemble normalized RMSE was 1.067. The next mechanism should make the native
action effect identifiable, using a short-horizon or residual cost target and
fresh roots, while retaining the independent action-permutation, gradient,
reward, and closed-loop validation gates. The v14.22 roots and thresholds will
not be retuned against these outcomes.

## Evidence boundary

This remains adaptive mechanism development, not confirmatory evidence. It
supports the narrower statement that an occupancy-trained action-cost direction
can transfer under the frozen guards in Hopper and Walker. It does not support
a domain-general action-cost restoration claim.
