# Stage-26 Temporal Plan Supervision Result

**Both roots fail both frozen development gates.** Tasks `t101585/101586`
completed on node004/node006 at implementation `b7bc0c587c`. Each root has
320 training pairs, 160 path-disjoint evaluation pairs and three matched
5,571-parameter models, each trained for 128 epochs/640 optimizer updates.
The returned path roster, check grid, coverage, fit budgets, selected
controller iteration and primitive-step accounting match the frozen protocol.
Independent row-level recomputation matches every aggregate and path metric.

Mean squared error of the 10/25/50-step ISE-rate curve:

| Root | Ordered history | Current frame repeated | Shuffled history | Zero |
|---|---:|---:|---:|---:|
| 209011 | 0.001304891 | 0.000937779 | 0.001602732 | 0.002167089 |
| 209061 | 0.000826556 | 0.000541823 | 0.001255866 | 0.001857882 |

Mean local ISE benefit of history decisions over each control; positive is better:

| Root | Current frame | Shuffled history | Always wait | Always now |
|---|---:|---:|---:|---:|
| 209011 | -0.001443079 | +0.000974031 | +0.006274864 | +0.001217162 |
| 209061 | -0.001866035 | +0.001405349 | +0.005650733 | +0.001314815 |

History reduces MSE against zero by 39.79%/55.51%, but has 39.15%/52.55%
higher MSE than the current-frame control. That control has lower MSE at
every horizon and on all eight evaluation paths per root. History changes
47/48 decisions relative to it; 28/30 changes have negative contributions.
Path-level history benefit is negative on six/five of the eight paths.

Charged steps are 4,860,700/4,851,700: 9,712,400 total, including 8,486,400
controller-reconstruction steps. Retrieved only 261,704 bytes of result JSON;
controller weights, temporal sequences and fitted model weights stayed remote.

Decision: retain this negative screen without changing epochs, thresholds or
seeds. Local timing response is learnable, but ordered history does not earn
its added role under this supervision. Next, resolve observation sufficiency
and the temporal-credit target before specifying another candidate. A
current-only control is not automatically promoted into a qualified policy.
Stage-9 remains the performance reference.

## Limitations

These are reused controller roots and local one-check response curves, not
independent-root confirmation or deployed episode gains. This result does not
prove history is unnecessary under other observations or longer credit windows.
