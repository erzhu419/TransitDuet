# Stage87: call-weighted joint learning outperforms lower-only

t120935-t120943 all done/exit0; eight roots, 27,136 native episodes /32,563,200 steps; mechanical gate passed. Wall897-926s/root, peak4570-4739MiB. Only77,528-byte compact JSON pulled; final weights remain server-side.
Same512 training episodes per method/period; both joint actors use all64 episodes per round. Source/std/critics/Adam/decoder/forecaster stay frozen; 640 mean updates, 48 final checkpoints, no critic fits or raw traces.
All20 reward endpoints use the frozen equal-root bootstrap65536 / Bonferroni20 family.

| Contrast | period50: mean [CI] | period100: mean [CI] |
| --- | --- | --- |
| joint-call minus lower-only | +0.2361 [0.1483, 0.3230] | +0.4812 [0.2929, 0.6792] |
| joint-call minus joint-level | +0.9736 [0.8219, 1.1325] | +1.2304 [0.9039, 1.5076] |
| joint-level minus lower-only | -0.7376 [-0.8560, -0.6108] | -0.7492 [-1.0411, -0.3818] |
| joint-call minus zero | +4.1760 [3.5458, 4.7809] | +5.3584 [3.7440, 6.7455] |
| source minus zero | -0.2351 [-0.4196, -0.0039] | -0.6610 [-1.1659, -0.0913] |

Joint-call beats lower-only and joint-level at every root in both periods. Its per-update empirical call-weighted KL is .00099960-.00100052; the maximum paired relative difference versus lower-only is <.006%. Nominal cumulative call-weighted KL is .008 for both, versus .00408/.00404 for joint-level. Max old-logp replay error2.89e-5.
The fixed call-weighted allocation closes the joint-versus-lower performance gap under this registered protocol. Preserve Stage83/85/86 negatives, scoped to their level-sum budget definitions; Stage84's direct upper effect remains supported.
Next: independently replicate with fresh training/evaluation rosters, identical allocation, updates, decoder and statistics; no parameter search. Then apply final-checkpoint actor swaps to attribute the Stage87 gain.

## Limitations
Teacher-initialized fixed-std/decoder MC mean learning, not full actor-critic or frequency superiority. Lower-only retains an active frozen upper and is not flat RL. Source loses to zero; no strong-source-baseline claim. Equal environment samples and call-weighted KL do not imply equal gradient compute. Empirical update KL is not final trajectory KL.
This is one fresh training replicate per frozen teacher/root; independent confirmation remains required. Allocation and its call-weighted total change together versus joint-level. Joint performance does not by itself isolate direct upper contribution from coadaptation. Stage67 critic-credit HOLD and the closed frequency-superiority claim remain unchanged.
