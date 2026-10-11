# Stage160: Complete-Episode Monte Carlo Upper PPO

Stage159 reduced credit variance but did not establish useful learned planning.
Test the existing shared PPO upper update on complete paired-reference episodes:
gamma=lambda=1, MC return-to-go and a previously frozen state-value baseline.
No Q bootstrap, replay, uniform-action warmup or entropy bonus. One stored
latent action/log probability per executed macro action; tanh bounds the same
two 20-second residual coefficients. Native lower and 34 actor inputs unchanged.

120 factual + 120 reference training episodes/root, same Stage157/159 scenes.
Update after five complete episodes (one per regime), eight PPO epochs,
64-transition minibatches: 24 batches, 768 actor and 768 value optimizer steps.
The existing GaussianActor has two 64-unit tanh layers and state-independent
log std initialized to -1, unlike SAC's ReLU actor/state-dependent std.
This is a trainer replacement, not a one-factor test of bootstrapping alone.

Last actor only; training-state constant fixed before frozen evaluation.
80 evaluation episodes/root: learned, forecast, nominal and constant residual
on twenty registered scenes. All forecast/nominal controls must reproduce
Stage159 exactly. Compare true physical cost, waiting and individual components
against BOTH forecast and constant, and report the registered paired SAC delta.
Lower intrinsic reward remains diagnostic, not the upper optimization target.

Two roots397/401, 39,283,200 main native ticks. Worker qualification/root:
one full forecast reproduction plus two factual/two reference short episodes
with two PPO updates; 165,960 qualification ticks in total. Reset model/RNG
before full training. Scheduler node001-006 unpinned, 1 CPU/3 GB per task.
Only small result JSONs sync locally; checkpoints and native CSVs stay remote.
103 focused tests pass, including complete-episode return/value cancellation,
macro action likelihoods, the 768-step budget, paired physical controls and
code-only scheduler output placement. Local learning tests use a fake simulator.

Run `native_transit_mc_upper_stage160_development_20261011_r1`:
t141677/root397 DONE on node006; t141678/root401 DONE on node001.
Both passed full forecast reproduction and complete-episode short training
with two PPO updates and nonzero actor change, then reset for formal training.
All 240 factual training, 240 reference and 160 frozen evaluation episodes
qualify. Both roots completed 768 actor/value optimizer steps and all 80
forecast/nominal controls reproduce Stage159. Main budget: 39,283,200 ticks.

| Learned minus control | Cost delta (397 / 401) | Wait delta, min (397 / 401) |
| --- | --- | --- |
| forecast | +0.020059 / -0.018224 | -0.00470 / +0.02200 |
| constant residual | -0.000039 / +0.000152 | -0.00070 / +0.00160 |
| registered paired SAC | -0.001050 / -0.023926 | +0.00810 / +0.00070 |

The forecast cost contrast is dominated by one OOD scene/root with an extra
peak bus (397) or one fewer (401): fleet component +/-0.020833. Root401's
wait/headway/unserved components all worsen. Learned versus constant has no
fleet differences and almost zero cost differences. Final actor state std is
[0.113, 0.048]/[0.033, 0.077], compared with much larger SAC variation.
PPO is less disruptive, but useful state adaptation is not established.

The representation fixes both endpoints of every six-departure window and
centers both residual bases. It can reshape within-window intervals, not shift
service time budgets between windows. This is a concrete authority restriction,
not proof that it caused the learning result. Next: frozen rolling-horizon
authority with unchanged daily trip count/endpoints and the same native lower,
before another optimizer change or seed expansion.

## Limitations

Development scene/root reuse, not independent confirmation or joint HRL.
Partial observation and stochastic-training/deterministic-deployment differences
remain. Reduced target variance or MC regression is not itself a performance
claim. No seed expansion unless useful state adaptation beats both controls.
