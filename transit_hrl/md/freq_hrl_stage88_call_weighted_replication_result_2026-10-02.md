# Stage88: frozen call-weighted learning replication confirmed

t121154-t121162 all done/exit0; eight roots, 27,136 native episodes /32,563,200 steps; mechanical gate passed. Wall854-880s/root, peak4621-4842MiB. Only78,878-byte compact JSON pulled; final weights remain server-side.
Same eight frozen Stage78 teachers as Stage87, with fresh training/evaluation rosters and no Stage87 weight reuse. Allocation, eight updates, 512 training episodes per method/period, decoder and statistics unchanged; both joint actors use all64 episodes per round.
Source/std/critics/Adam/decoder/forecaster stay frozen; 640 actor-mean updates, 48 final checkpoints, no critic fits or raw traces. All20 endpoints use equal-root bootstrap65536 / Bonferroni20.

| Contrast | period50: mean [CI] | period100: mean [CI] |
| --- | --- | --- |
| joint-call minus lower-only | +0.2215 [0.1406, 0.2834] | +0.4801 [0.3190, 0.7087] |
| joint-call minus joint-level | +1.0201 [0.8050, 1.2513] | +1.2503 [0.9260, 1.6242] |
| joint-level minus lower-only | -0.7985 [-1.0626, -0.5490] | -0.7702 [-1.2813, -0.2422] |
| joint-call minus zero | +4.5467 [3.6008, 5.6606] | +5.4827 [4.3409, 6.7143] |
| source minus zero | -0.0666 [-0.2652, 0.1454] | -0.5034 [-1.2303, 0.0552] |

The frozen confirmation gate passes: all four primary CI lower bounds exceed zero. Joint-call beats lower-only and joint-level at every root in both periods.
Actual per-update call-weighted KL is .00099917-.00100088 for joint-call; the maximum paired relative difference versus lower-only is <.005%. Nominal cumulative budgets remain .008 versus .00408/.00404 for joint-level. Max old-logp replay error2.67e-5.
Preserve Stage83/85/86 negative results under their level-sum budget definitions. Next: fixed Stage88 final-checkpoint actor swaps, with fresh evaluation samples and the lower held exact, to isolate direct upper contribution and transfer; no retraining or budget tuning.

## Limitations
This confirms fresh-sample learning on the same eight teachers, not a new teacher population; Stage87/88 are not pooled as16 independent teachers. Teacher-initialized fixed-std/decoder MC mean learning is not full actor-critic or frequency superiority. Lower-only retains an active frozen upper and is not flat RL. Source-minus-zero CIs are inconclusive, not equivalence or a strong-source-baseline result.
Equal environment samples and call-weighted KL do not imply equal gradient compute; empirical update KL is not final trajectory KL. Joint performance alone does not isolate direct upper contribution from coadaptation. Stage67 critic-credit HOLD and the closed frequency-superiority claim remain unchanged.
