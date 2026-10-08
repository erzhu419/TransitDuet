# Stage138: Native Single-Option Credit

Stage137 reduces ordinary PPO damage but bounded/projection forward updates
remain mixed and negative on average. Change actor credit alone, not the noise,
radius, optimizer, critic target or strong lower.

Roots410037/410049, periods50/100, horizon1200. Eight new scenarios with two
independent policy-noise folds each. For every sampled upper option, rerun its
identical prefix, replace only that action by the current mean, and continue
the stochastic policy with identical future upper/lower innovations. Credit is
sampled episode return minus this counterfactual episode return. Prefix rewards
cancel. The baseline rollout excludes the current action draw from execution;
it is a simulator-query baseline, not a learned value or test-data target.

Only the original sampled paths enter PPO. Counterfactual trajectories produce
labels and are never on-policy samples. Compare original MC-minus-critic and
native option credit on the same batch, optimizer seed and unchanged MC critic
targets; their critic updates must match exactly. Upper std0.15, authority0.05,
fixed empirical KL0.000555556, frozen deployed lower/value policies. Retain both
update signs on32 separate paired native scenes, with reward and tracking error.
No evaluation winner, confirmation CI or automatic seed extension.

Count all full-prefix reruns:per root32sampled+576counterfactual+384evaluation
episodes =992episodes/1,190,400steps; total1,984episodes/2,380,800steps. Sixteen
workers+parent/12GiB, scheduler dynamic node001-node006. Scalar JSON only, no
checkpoint or native-trace writes. Inherited costs remain separate.

## Run Receipt

Implementation `57d56a9e35`, registration `f4f7301997`, both pushed before submit.
Sixteen focused tests passed: single-option prefix/noise pairing, identical critic
updates, exact native budget, unchanged default runner and bounded-PPO regression.
Selected action means use the same scalar inference as native execution.
Stage137 cells and four selected Stage135 upper files are available on node004.
Run `pointmaze_option_credit_stage138_probe_20261008_r1`: `t136931` root410037,
`t136932` root410049 are DONE on node004/node006. No waiting
aggregator, hard node binding, checkpoint pull or local native training.

## Result And Next Step

Both cells pass exact accounting:1,984episodes/2,380,800steps;119,704bytes fetched.
Conditional credit removes much phase variance but does not yield stable gains:

| Period | Critic plus - warm | Option plus - warm | Option plus - critic plus |
| --- | ---: | ---: | ---: |
| 50 | -0.063693 | -0.028595 | 0.035098 |
| 100 | 0.004111 | -0.018286 | -0.022397 |

Phase variance fraction falls from92.5%-94.7% to4.5%-18.6%. Option-credit noise
fold cosine is0.531/-0.164 at50 and0.045/-0.295 at100; reduced temporal variance
is not consistent action-gradient agreement. Forward gains are positive in two
of four root-periods, with mixed tracking effects. Warm-minus-forecast remains
positive at both periods(+1.683892/+0.961554 equal-root means).

Stage139 replays these same sampled states and cached credits, counting inherited
label costs separately. Compare Adam, direct score and the existing damped Fisher
mean geometry at equal KL, both signs. Evaluate stochastic and deployed mean
upper policies separately on new scenes to distinguish optimizer geometry from
training/deployment mismatch. No new labels, evaluation winner or confirmation CI.

## Limitations

Conditional action credit need not reduce variance or improve the deployed mean
policy. Future stochastic-upper versus mean deployment remains different. This
requires native simulator queries and does not prove a sample-efficiency win,
joint HRL, frequency responsibility or promotion. Earlier evidence stays.
