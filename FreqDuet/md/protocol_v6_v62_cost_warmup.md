# V62: frozen-actor full-return cost warm-up

Keep each V59 post-64 actor, upper actor and B200 deployment frozen. Collect
four full days per actor for fitting: demand seeds 45007/934011, sampler repeats
0/1. Validate on demand seeds 45107/934111 with repeat 2: two complete demand
days absent from fitting, not a random transition split. Four actors, 24 days.
These validation seeds are reserved for this experiment, not a new policy test.

Save causal 233-dimensional states, executed actions and signed full returns
`CV_day^2 - CV_prefix^2` only on the server. Same-day/rep historical replays must
match V60 exactly; all replays retain stream conservation and frozen upper
checks. No actor or dual updates. Collectors never receive critic fit weights.

Fit only the original cost-network architecture, initialized from V59, using
signed Monte Carlo targets, fresh Adam at the original cost learning rate,
2,048 steps and batch 256. Sample fitting days uniformly; no early stopping or
validation tuning. Save fitted/current and synchronized target cost weights
before loading validation labels. Compare frozen critic, zero prediction and
the causal baseline `mean(training final CV^2) - current CV^2`.

Gate: on each of the two validation days, both MAE and RMSE must improve over
all three baselines. All four actors must pass before a dual/control experiment
is admitted. No multiplier tuning, threshold relaxation or main switch.

## Limitations / Next Step

Chosen-action full-return calibration does not establish unexecuted actions'
ranking or control benefit. Decisions and sampler repeats are correlated;
report whole-day/context metrics. If admitted, next test completed-day dual
feedback before actor updates in a small matched control experiment.
