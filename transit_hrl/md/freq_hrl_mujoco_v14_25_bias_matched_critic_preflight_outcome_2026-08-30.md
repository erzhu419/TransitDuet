# MuJoCo v14.25 bias-matched critic preflight outcome

## Frozen result

The preregistered all-environment gate failed with two of three environments
supported. Scheduler tasks `t84809-t84811` ran from source revision
`f69d3422be09db1fde0a76d95fc77208b3a00838` on `node003` and completed
normally.

| Environment | Update parameters | Upper holdout R2 | Lower holdout R2 | Lower minimum gradient cosine | Validation merit change | Reward violations | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HalfCheetah-v5 | 12 | 0.668 | 0.951 | -0.659 | no eligible design step | 2-3 on design | not supported |
| Hopper-v5 | 6 | 0.629 | 0.931 | 0.716 | -0.011% | 0 | supported |
| Walker2d-v5 | 12 | 0.271 | 0.960 | 0.841 | -8.916% | 0 | supported |

Hopper and Walker selected output-bias RMS step `1e-4` and passed fresh
validation. HalfCheetah failed every registered bias step. Even `1e-7` raised
frequency merit by 970% and violated three reward constraints.

## Mechanism decision

Matching the actor parameter scope to the intervention is insufficient for
HalfCheetah. Its predictive critic metrics remain strong, but one or more lower
ensemble gradients oppose the others: the minimum pairwise cosine is `-0.659`,
compared with positive minima on Hopper and Walker. A positive median cosine is
therefore too weak to justify averaging the HalfCheetah derivative.

The next preflight should not relax reward guards or shrink the same bias step
registry. The paired trajectories already contain a direct policy-bias
experiment. Their antithetic cost contrasts can estimate one finite-difference
gradient per root and disturbance mode without differentiating an MLP critic.
A robust train-path aggregate should be required to agree with an independent
holdout-path aggregate before exact design and validation. This tests whether
the paired intervention itself provides a stable update direction.

## Evidence boundary

This adaptive result supports bias-matched critic updates on Hopper and Walker,
not on HalfCheetah. It does not establish a domain-general actor update and is
not confirmatory paper evidence.
