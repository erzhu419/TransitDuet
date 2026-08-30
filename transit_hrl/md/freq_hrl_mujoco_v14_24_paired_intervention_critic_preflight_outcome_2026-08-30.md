# MuJoCo v14.24 paired-intervention critic preflight outcome

## Frozen result

The preregistered all-environment gate failed with two of three environments
supported. Scheduler tasks `t84805-t84807` ran from source revision
`79b7b273ee6330e72b7d82ab23c6411339b75a16` on `node003`. All tasks completed
normally; peak resident memory was 10.3 GB.

| Environment | Upper holdout R2 | Lower holdout R2 | Upper action-permutation gain | Lower action-permutation gain | Critic gate | Validation merit change | Reward violations | Result |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| HalfCheetah-v5 | 0.653 | 0.966 | 0.252 | 3.281 | pass | no eligible design step | 1-4 on design | not supported |
| Hopper-v5 | 0.597 | 0.911 | 0.173 | 1.923 | pass | -0.046% | 0 | supported |
| Walker2d-v5 | 0.150 | 0.971 | 0.038 | 17.262 | pass | -7.106% | 0 | supported |

The paired intervention data repaired critic identification in all three
environments. Hopper selected full-actor RMS step `1e-6` and transferred a
0.046% merit reduction. Walker selected `1e-6` and transferred a 7.106%
reduction. Both retained the frozen reward guard.

HalfCheetah passed every critic gate, but no full-network actor step was design
eligible. Even `1e-8` raised frequency merit from 0.0554 to 0.6507 and violated
one reward constraint; larger steps violated two to four. The critic is no
longer the failure point.

## Mechanism decision

Paired intervention collection is retained as a successful identification
mechanism, but the v14.24 full-actor update does not advance into the shared
core. The intervention estimates output-action effects by perturbing only actor
mean biases, whereas the update propagates the inferred action derivative
through every actor layer. HalfCheetah shows that this broader parameter update
can amplify into an unstable closed-loop policy despite a well-identified
critic.

The next preflight should match the update estimand to the intervention: compute
the same action-cost gradient but update only upper and lower actor output
biases. It must retain paired critic collection, predictive and gradient gates,
reward guards, and fresh design/validation roots. Reversing the v14.24 direction
or retuning its roots is not justified by this result.

## Evidence boundary

This adaptive result supports paired action interventions as critic-identification
data and supports the resulting full-actor direction only on Hopper and Walker.
It is not a domain-general policy-update result or confirmatory paper evidence.
