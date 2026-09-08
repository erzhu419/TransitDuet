# MuJoCo v21 Reward-Selective Feasible-Action Result

## Decision

The frozen v21 development preflight completed all 48 cells. Every cell exited
successfully, every projected arm satisfied the terminal certificate and
prefix-power checks, and the reward-selective weights were active and correctly
normalized. The reward-selective candidate did not pass the advancement gate.

This is a negative development result. It is not manuscript, confirmatory,
generalization, superiority, or no-tradeoff evidence.

## Registered Outcome

The candidate failed four registered gates:

- HalfCheetah reward was 6.04% below the matched uniform arm, beyond the 5%
  floor;
- component correction regressed by 29.65% in the pooled root analysis versus
  uniform consistency;
- total correction regressed by 35.18% versus uniform consistency;
- Hopper exceeded the registered action-change-rate burden limit.

The candidate beat the uniform reward in only 5 of 12 environment-by-root
pairs. Its positive Walker2d mean normalized ratio was driven by one root whose
uniform reward was unusually low; the arithmetic mean reward was lower and
three of four Walker2d roots favored uniform. The frozen gate remains reported
unchanged, but it must not be paraphrased as stable two-environment reward
superiority.

The mechanism diagnostics were valid: candidate weights had unit minibatch
mean, maxima above 4, and finite positive weighted losses. The failure therefore
rejects positive reward-advantage weighting rather than exposing a disabled
code path or invalid projection.

## Environment Means

| Environment | Reserve reward | Uniform reward | Selective reward | Selective component reduction vs uniform | Selective total reduction vs uniform |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 1536.375 | 1705.939 | 1612.543 | -15.19% | -4.92% |
| Hopper-v5 | 217.540 | 235.539 | 237.347 | -12.83% | -42.84% |
| Walker2d-v5 | 150.345 | 257.349 | 196.998 | -60.93% | -57.79% |

## Uniform-Control Diagnostic

The matched uniform arm was a control, not the registered v21 candidate. A
post-outcome diagnostic found that, relative to zero-consistency terminal
reserve, uniform consistency improved the mean reward in all three environments
and reduced both correction summaries in all three:

| Environment | Reward change | Reward wins | Component reduction | Total reduction |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | +11.04% | 3/4 | 6.02% | 15.34% |
| Hopper-v5 | +8.27% | 4/4 | 38.53% | 46.71% |
| Walker2d-v5 | +71.17% | 3/4 | 48.74% | 84.40% |

Because v21 had only four optimizer roots and did not register uniform as the
candidate, these observations authorize only a separately frozen fresh-root
confirmation. They do not promote the uniform arm or alter the v21 stopping
decision.

## Next Decision

Positive reward-advantage weighting is closed. Its v21 roots cannot be reused
for temperature, clipping, coefficient, schedule, threshold, or sign tuning.
The next experiment must keep the already observed uniform mechanism unchanged,
use fresh roots, and test the joint claim of projected-reward noninferiority and
reduced certificate correction with paired uncertainty estimates.

