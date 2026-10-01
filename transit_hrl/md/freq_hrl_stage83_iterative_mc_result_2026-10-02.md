# Stage83 Result and Next Step

Preflight t120825/826 and full t120827-835 all exited0; code3f0bac6702,
25 focused tests passed.8 roots,8 registered updates,18,432 native episodes,
22,118,400 steps;16,384 training and2048 held-out evaluation episodes.
256 policy updates /384 actor-mean updates; only32 final weights saved server-side.
Compute620-656s/root,4.5-4.8GB peak; no intermediate checkpoint or raw trace writes.

Final reward contrasts, Bonferroni12 equal-root bootstrap CI:

| Period | Joint minus source | Joint minus zero | Joint minus lower-trained |
| --- | --- | --- | --- |
| 50 | +3.68034 [3.14722,4.12658] | +3.42083 [2.85920,3.90456] | -.84525 [-1.00365,-.67806] |
| 100 | +5.66636 [4.14162,7.24793] | +4.73130 [3.26739,6.17720] | -1.05227 [-1.45233,-.68595] |

Lower-trained also improves source and zero in both periods: source gains
+4.52559 /+6.71863. Iterative native MC learning works, but joint still loses
to same-budget lower training. Both std vectors, sources/Adam/values and decoder
stay frozen. Per-round actual KL [.000999718,.001000961]; max logp difference2.86e-5.

## Limitations
Lower-trained retains an active frozen upper; it is not a flat RL baseline.
Teacher-initialized fixed-std MC learning is not full actor-critic or frequency
superiority. Source itself trails zero in both periods. Stage67 critic-route HOLD
is unchanged; no source policy adoption or selected intermediate evaluation.

Next: native actor-swap interventions using saved final weights and fresh seeds.
Hold each trained lower fixed and exchange learned versus original upper; test
whether upper adds value or merely coadapts with the weaker joint-trained lower.
No extra training, radius/allocation search or best-checkpoint selection.
[Compact evidence](../results/pointmaze_iterative_mc_stage83_full_20261002_r1/qualification_compact.json).
