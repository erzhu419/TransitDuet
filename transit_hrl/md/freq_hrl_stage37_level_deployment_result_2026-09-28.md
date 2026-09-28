# Stage-37 Level Update And Gate Deployment Result

Tasks `t102233`-`t102280`: all 48 cells and 6400 evaluation episodes complete.
Forty training audits, eight cached-gate audits, 128 native checkpoint/mode
replays and the independent nine-endpoint root-count bootstrap pass.
Source `21f4ee2dcc`; full registration records `de8b457376`.

| Primary final-weight contrast | Mean | Nine-endpoint adjusted CI |
|---|---:|---:|
| Upper-only return vs frozen | -8.4031 | [-21.1024, 5.5260] |
| Lower-only return vs frozen | -5.0156 | [-11.5030, -0.2770] |
| Upper+lower return vs frozen | -15.8792 | [-28.0726, -4.6440] |
| Gate-only sampled minus threshold | -0.3636 | [-2.4831, 1.7524] |
| Joint sampled minus threshold | -0.6041 | [-2.8132, 0.8436] |
| Gate-only vs frozen under sampled gates | -18.6798 | [-27.0031, -12.0547] |

Lower-only and combined updates harm return; upper-only is inconclusive.
Sampling the original gate probabilities does not rescue the trained gate.
Selected checkpoints remain secondary: upper/lower/both choose iteration0
in 4/5/5 of eight roots. The diagnosis does not identify the cause of harm.

Registered method cost: 58800000 steps/1267672 upper/1740843 gate calls;
verification153600 steps/4200 upper/4892 gate calls. Eight completed jobs
were incorrectly retried because the old terminal marker was unrecognized.
All zero-exit originals are now recorded done; duplicate runs add10838400
steps (charged method total69638400), not new statistical replicates.
Current artifacts match the audited summary. Related submitters now print the
recognized `Training complete` marker; its scheduler integration check passes.

## Limitations And Next

Conditional development on eight reused controller roots, not independent
confirmation. Keep earlier negative results. Next compare lower intrinsic
credit against task-aligned credit under a fixed upper/gate, and inspect gate
SMDP advantage allocation before another joint-training trial.
Raw arrays and weights stay remote; compact summary/accounting JSON is local.
