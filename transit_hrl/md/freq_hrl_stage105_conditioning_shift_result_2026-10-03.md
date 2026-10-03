# Stage105 Faster-Regime Conditioning: Full Result

t130218-t130226 all finished with exit0. All eight roots passed read-only server qualification; source-cell reaggregation and an independent equal-root bootstrap exactly reproduce the official summary. Hidden regime dwell was0.4-0.8s, with the original0.8-1.6s-distribution teachers and decoder held fixed.

| Registered primary reward contrast | Mean difference | Bonferroni12 corrected CI | Positive roots |
| --- | ---: | --- | ---: |
| period50 conditioned minus independent | +0.526550 | [0.159400,0.872114] | 7/8 |
| period50 conditioned minus base | +5.720601 | [5.271630,6.085222] | 8/8 |
| period100 conditioned minus independent | +1.941252 | [0.917317,3.203313] | 8/8 |
| period100 conditioned minus base | +5.542214 | [4.508531,6.617484] | 8/8 |

All four primary lower CI bounds exceed zero:the registered shifted-conditioning claim is **supported**. All12 contrasts contain10 positive and2 inconclusive results. Both learners improve over base and zero-plan at both periods. The negative period50 conditioning root410073 (-0.018065) remains archived.

Independent joint mean rewards are974.308577 at50 and855.891218 at100; the conditioning gains are0.054% and0.227% of those means. These give the reported reward scale, not an additional tested endpoint.

Original base-minus-zero is inconclusive:period50 -0.099602 CI[-0.342141,0.222228]; period100 -0.293053 CI[-0.924906,0.801439]. This is the source actor's residual contribution:the registered `zero` variant sets the residual blend alpha to0 but retains the ridge forecast reference and its velocity context, so it is not a flat/no-planning baseline. Next:matched-training-budget forecast-only lower learning and a genuinely flat full-feedback controller are needed to isolate learned upper-residual value and hierarchical value separately. No new task was dispatched in this result turn.

Exact cost:34,816 native episodes /41,779,200 steps,512 mean updates,256 update operations,73,728 extra upper replay forwards and32 final server-only checkpoints. Native wall1008.72-1052.67s/root. The61,643-byte compact pull retains all contrasts, per-root means/effects, actual task options, freeze/KL summaries and counters; raw evaluations, trajectories and checkpoints remain server-only. Source preparation is a separate inherited cost.

## Scope
This is adaptation across one regime-timescale shift using the original eight Stage96 initializers and Stage97 decoder/forecaster. Std, values and Adam remain fixed:it is MC actor-mean learning, not new-initialization confirmation, zero-shot transfer, cross-domain generality, full actor-critic or frequency superiority. Common-noise training adds replay work; realized trajectory KL is not matched. Stage104 remains a separate cohort with no pooling; its not-supported joint-superiority result and Stage67 critic HOLD are unchanged.
