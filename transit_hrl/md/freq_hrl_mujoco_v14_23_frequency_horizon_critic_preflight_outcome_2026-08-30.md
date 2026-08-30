# MuJoCo v14.23 frequency-horizon critic preflight outcome

## Frozen result

The preregistered all-environment gate failed with one of three environments
supported. Scheduler tasks `t84796-t84798` ran from source revision
`f787e1d1c700104422f097e0939ee3a868e7291e` on `node004` and completed
normally.

| Environment | Upper holdout R2 | Lower holdout R2 | Upper action-permutation gain | Lower action-permutation gain | Critic gate | Validation merit change | Reward violations | Result |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| HalfCheetah-v5 | -0.067 | 0.190 | -0.0060 | 0.0725 | fail | not evaluated | not evaluated | not supported |
| Hopper-v5 | 0.235 | 0.548 | 0.1415 | 0.0610 | pass | -2.096% | 0 | supported |
| Walker2d-v5 | 0.154 | 0.434 | 0.0848 | 0.0572 | pass | +2.387% | 0 | not supported |

Hopper selected actor RMS step `1e-6` and transferred a 2.096% frequency-merit
reduction with no reward violation. Walker selected `3e-8` on design paths but
reversed to a 2.387% merit increase on fresh validation paths. HalfCheetah's
short-horizon lower critic became identifiable, but its upper critic failed
both holdout R2 and action-permutation gates.

## Mechanism decision

Frequency-window truncation does not advance into the shared core. It moved the
HalfCheetah failure from the lower to the upper critic and did not prevent a
Walker design-to-validation reversal. This rules out target horizon alone as
the missing mechanism.

The common remaining weakness is observational action coverage: each critic is
fit from the action distribution induced by one current policy trajectory per
environment path. Predictive holdout performance can therefore coexist with an
unstable policy derivative. The next preflight must collect explicit paired
action interventions, while retaining the v14.23 target horizons and every
critic, reward, and fresh closed-loop gate. The v14.23 roots, horizons, and
steps will not be retuned against these outcomes.

## Evidence boundary

This adaptive preflight supports only Hopper under the frozen protocol. It does
not establish a domain-general frequency-horizon critic or justify selecting
the Hopper result as a paper claim.
