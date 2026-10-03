# Stage98 Fresh Joint Cohort Result

All nine tasks t128756-t128764 finished with exit code 0. Official qualification matches independent server-side reaggregation. The four preregistered primary endpoints are confirmed on all eight new Stage96 teachers; every root has a positive paired difference for each primary.

| Period | Joint-call Versus | Mean Reward Gain | Corrected CI |
| --- | --- | --- | --- |
| 50 | Lower-only | +0.34768 | [0.25571, 0.45224] |
| 50 | Joint-level | +1.12009 | [0.81965, 1.40236] |
| 100 | Lower-only | +0.63952 | [0.56272, 0.78178] |
| 100 | Joint-level | +1.49801 | [1.14382, 1.84389] |

- One equal-root 65,536-draw Bonferroni20 family, no old-cohort pooling. All 20 contrasts remain archived: 16 positive, 2 negative (joint-level versus lower-only), 2 inconclusive (untrained bounded versus zero plan).
- Joint-call gains versus untrained bounded base: +4.80726 [3.63309, 5.80091] at50; +6.71224 [5.65860, 7.86804] at100. The incremental advantage over lower-only is smaller than the total learning gain.
- Exact budget: 27,136 episodes / 32,563,200 steps, 640 mean-parameter updates, 384 policy updates; 48 registered fixed-final checkpoints exist server-side. Source, std/value and conditional-KL checks passed. Runtime 855.57-888.20s/root; RAM sampling was unmeasured.
- Retrieved 70,479 bytes of reduced JSON, no checkpoints or native trajectories. Native preflight negatives remain archived separately.

Scope: fixed-std/decoder MC mean learning on new teacher initializations of the same native task, not unseen-task generalization, full actor-critic or frequency-superiority proof.

Next: Stage99 rebuilds matched independent/common lower policies under U0 and the new final joint upper. Then repeat the unchanged staged-upper test on this new cohort.
