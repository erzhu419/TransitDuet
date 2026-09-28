# Stage-29 Action-Conditioned Predictive State

Freeze roots 209011/209061 (preflight 208001), remote Stage-26 controller
weights, Stage-12 factual replay tolerance 1e-8. New path bases
3289000/3290000/3291000: fit offsets 1-16, evaluation 101-108, two each
for preflight. Exclude inherited, Stage-26/28 and derived training paths.

Fit trajectories retain balanced-jitter upper planning and continuous lower
feedback, adding independent uniform requested-action noise [-0.25,0.25]
before Box clipping. Evaluation trajectories have no excitation. At ten
checks 100/200/.../1000 plus offsets 0/5/10/15/20 twice, replay matched
prefixes then one requested-action intervention +/-0.25 on each axis.
Preflight has one check per evaluation path. External futures are labels,
never inputs. Planner schedules, prior actions and prefixes remain matched.

Generic predictor: existing causal GRU width64, latent16, Gaussian delta
heads for physical4 and external6. Current action enters only the physical
head. Observed history64 contains physical4, external6, previous requested
action2. Fit/query transitions start at step64, stride5. Factorial controls:
history_action, current_action, history_blind, current_blind; current views
repeat the last observed frame, blind views zero both previous/new actions.
All share capacity, initialization and training-only scales. Optimize full
Gaussian NLL, log variance [-8,4], Adam1e-3, minibatch128, gradient norm1,
64 fixed epochs (preflight4), no checkpoint selection.

Action gate requires both history/current action models to beat their blind
partners on held-out normalized physical MSE and paired physical-effect MSE,
also beating zero effect. History gate requires history_action target rate
MSE below current_action. Both roots must pass; ties fail. Report Gaussian
NLL, coordinate coverage/innovation rates and known-action 1/5/10-step mean
rollouts separately, without using them to select a model or horizon.

Each root: 19200 fit + 9600 evaluation + 179520 intervention + 1200 factual
steps = 209520 new steps, 3648 fit/1824 query transitions, 160 effect pairs,
four fits/7424 optimizer steps. Preflight: 2368 new steps, 96/96 transitions,
four effect pairs/fits, 16 optimizer steps. Raw arrays and models stay on the
server; JSON carries metrics, effect rows and one audit transition per path.
Scheduler node001-node006 dynamically, four workers (5 CPU/8 GB), preflight
one (2 CPU/3 GB); model fits run in parallel after trajectory collection.
Twenty-eight focused tests passed, covering transition/action alignment,
closed-loop excitation, matched interventions, external-head separation,
label isolation, Gaussian likelihood, forward replay and exact accounting.

Implementation frozen at `d5f4bfe3d9`. Preflight `t101733` completed on
node006: zero factual errors, 96/96 transitions, four intervention pairs,
four fits/16 updates and 2368 new steps. Server-only independent raw-array
recomputation matched all metrics, effect labels, forward tapes and 94/96
overlapping next-state/action alignments. Retrieved 32935 bytes of JSON.
Both scientific gates failed. Some external training changes were zero,
giving minimum target scales and extreme held-out NLL; preserve the result.
Full tasks `t101734/101735` completed on node006/node005 at the same revision,
419040 total new steps and four parallel model fits per root. The
[full result](freq_hrl_stage29_state_response_result_2026-09-28.md) passes
the action-response gate on both roots but fails the history and joint gates.
History target-rate MSE is 0.74%/1.43% worse than current/action. Independent
server-only recomputation matched all metrics and intervention labels, plus
1816/1824 overlapping next-state/action rows per root. No policy is promoted.

## Limitations

This qualifies one-step state response, not learned plan-value control or
deployment. Known-action multistep replay supplies future action tapes as a
diagnostic. Gaussian coverage is predictive calibration, not a confidence
bound or a calibrated regime-change probability. Two reused controller roots
provide development evidence.
