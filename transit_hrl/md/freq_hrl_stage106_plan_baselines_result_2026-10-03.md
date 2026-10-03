# Stage106 Learned-Plan Value: Full Result

t131845-t131853 all finished with exit0. All eight roots passed native qualification; read-only source-cell reaggregation and a separately computed equal-root bootstrap exactly reproduce the official all26 Bonferroni-corrected intervals. Source cohort, task, training/evaluation rosters, freezes and sample/planning counters match the frozen r2 protocol.

| Registered primary reward contrast | Mean | Bonferroni26 CI | Positive roots |
| --- | ---: | --- | ---: |
| 50 conditioned joint minus trained forecast | -0.101978 | [-0.433496,+0.285426] | 3/8 |
| 50 conditioned joint minus trained flat | -179.391231 | [-184.594977,-173.745961] | 0/8 |
| 100 conditioned joint minus trained forecast | -0.849606 | [-1.636164,+0.095903] | 2/8 |
| 100 conditioned joint minus trained flat | -292.116895 | [-302.552225,-281.180518] | 0/8 |

The registered learned-plan increment claim is **not_supported**. Joint-versus-forecast is inconclusive, not equivalence; joint-versus-flat is supported harm at both periods, negative in every root. The full26-contrast family contains10 positive,12 negative and4 inconclusive results. The strong baseline result is not a narrowly missed positive interval.

Common-upper-noise conditioning still beats independent joint learning:50 +0.716531 CI[+0.128282,+1.177090],6/8 positive;100 +1.576400 CI[+0.586880,+2.755097],8/8 positive. Period50 negative roots410073 (-0.007593) and410101 (-0.123302) remain archived. This supports the named training comparison, not hierarchical superiority.

Mean final rewards are joint/forecast/flat 971.375/971.477/1150.766 at50 and847.327/848.176/1139.443 at100. Untrained flat already exceeds joint base by182.294/293.768. Own-initializer training gains are joint +5.918/+5.738, forecast +5.913/+6.130 and flat +3.015/+4.087, all supported under their registered contrasts. The large interface-level gap predates these updates; more seeds are not the next corrective action.

Exact cost:69,120 native episodes/82,944,000 steps,768 mean updates,512 update operations,73,728 extra upper replay forwards and64 final server-only checkpoints. Native wall2094.228-2145.878s/root. The129,997-byte compact pull retains all effects, per-root means, tracking diagnostics, source records, actual task options, freeze/KL summaries and counters; no raw trajectories or checkpoints were pulled. Shared source provisioning remains a separate inherited cost.

## Scope
This is teacher-assisted fixed-std MC mean learning on the faster-regime PointMaze task, not full actor-critic, from-scratch flat PPO/SAC or a universal statement about HRL. Flat also changes the derived input semantics to current target error and causal velocity; this comparison alone does not isolate which individual feature causes the gap. Period indexes the flat teacher initializer, not its control rate. Nominal call-weighted KL and native samples match, not realized trajectory KL or all computation. Stage104/105 remain separate; Stage67 critic HOLD is unchanged.

Next:preserve full lower instantaneous feedback and diagnose the value of an optional plan input against the strong flat interface before investing in more upper modules or training. Keep current reward/task parameters and baselines; neither weaken lower feedback nor tune a new stress setting to erase this result. The mandatory forecast-reference route is closed as the current positive hierarchy claim. No new task was dispatched in this result turn.
