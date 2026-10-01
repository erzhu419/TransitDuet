# Stage84 result: upper adds value, joint-trained lower is weaker

Preflight t120843/t120844 and full t120845-t120853 all exit0; checkpoint/freeze, composition, pairing and budget checks passed.
Eight roots, seven combinations, periods50/100, 32 fresh paired seeds/root/period; 3,584 episodes / 4,300,800 steps.
Zero training/critic fits/traces/checkpoint writes; wall101-105s/root, peak4,144-4,223MiB; server-only checkpoint reads.
All 22 reward endpoints use equal-root bootstrap65,536 / Bonferroni22. UJ/U0: learned/source upper; LJ/LL/L0: joint/lower-only/source lower.

| Fixed-checkpoint contrast | period50: mean [CI] | period100: mean [CI] |
| --- | --- | --- |
| UJ-LJ minus U0-LJ: direct upper effect | +0.2086 [0.1812, 0.2309] | +0.4861 [0.3791, 0.6002] |
| UJ-LL minus U0-LL: transferred upper effect | +0.1935 [0.1633, 0.2161] | +0.4459 [0.3451, 0.5512] |
| UJ-L0 minus U0-L0: source lower control | +0.2465 [0.2185, 0.2689] | +0.5830 [0.4576, 0.7391] |
| U0-LJ minus U0-LL: lower difference | -0.9905 [-1.2722, -0.6983] | -1.3605 [-1.9111, -0.9794] |
Upper contributes positively with every tested lower; gains are modest relative to total episode return.
Joint still loses to lower-only: -0.7819/-0.8744, both CI-negative; joint, lower-only and transferred policies beat zero-residual.

## Limitations

Swaps use two training runs, not an equal-training-budget win. Half lower KL allocation versus joint-training interference remains unresolved.
Teacher initialization, fixed decoder/std, Stage67 critic-route HOLD and the closed frequency-superiority claim remain unchanged.

Next: equal-sample, equal-total-KL staged/alternating training versus lower-only, to retain upper gains without sacrificing lower learning.
All22 endpoints and task rosters: results/pointmaze_actor_swap_stage84_{preflight,full}_20261002_r1/qualification_compact.json and task_roster.json.
