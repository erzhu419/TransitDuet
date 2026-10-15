# V62: frozen-actor full-return cost warm-up

Keep each V59 post-64 actor, upper actor and B200 deployment frozen. Collect
four full days per actor for fitting: demand seeds 45007/934011, sampler repeats
0/1. Validate on demand seeds 45107/934111 with repeat 2: two complete demand
days absent from fitting, not a random transition split. Four actors, 24 rollouts.
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

## Outcome

Source `d56f5b844d`; tests/smoke `t165272`, data `t165273`-`t165276`, fits
`t165277`-`t165280`, aggregate `t165281`: all exit 0. Sixteen tests pass;
24 rollouts / 124,800 decisions include eight exact V60 matches.
Validation macro MAE: frozen **0.133734**, fitted **0.020678**, zero **0.022610**,
causal final-mean **0.003804**. Gate **0/4**; no actor or dual updates.
The eight validation rollouts share two demand seeds across four actors, not
eight independent days. Small results stay in `freqduet/results_freqduet/protocol_v6_v62_cost_warmup/summary/status.json`.

## Limitations / Next Step

Chosen-action full-return calibration does not establish unexecuted actions'
ranking or control benefit. Decisions and sampler repeats are correlated;
report whole-day/context metrics. Control is not admitted. Next factor the
known prefix analytically: `Q_cost(s,a) = E[CV_day^2 | s,a] - CV_prefix^2`;
fit the unknown final component, validate on new demand days, then check action
differences separately. V62 validation becomes development evidence, not a
reusable confirmation gate. Keep the actor, dual and main frozen.
