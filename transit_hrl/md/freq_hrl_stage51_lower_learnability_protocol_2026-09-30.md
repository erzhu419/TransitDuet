# Stage51: native lower headroom and independent credit

Stage50 did not establish an optimizer repair. This diagnostic separates native control headroom, same-network supervised learnability, and repeatability of the present reward-credit direction. Stage50 is retained unchanged.

## Frozen Experiment

- Source: Stage42 `task_clock`, warmup16 (preflight warmup2), before any actor update. Upper/gate actors and all critics remain fixed; their state-mediated outputs may differ.
- Seven policies: frozen lower, waypoint/task feedback teachers, waypoint/task lower-MLP clones, and two corresponding shuffled-label clones. Each teacher receives only the existing lower input. The task teacher uses the currently visible target, bypassing the waypoint.
- Fixed feedback: unit-mass double integrator, continuous LQR with Q=diag(25,25,1,1), R=I, gains Kp=5I and Kv=sqrt(11)I. Command is Kp*error-Kv*velocity-current_force, clipped to +/-0.95 and mapped through atanh to the existing Gaussian action mean. Source action standard deviation is preserved in teachers and clones. No model identification or gain search.
- Clones fit raw action-mean MSE on frozen batch A only, using the unchanged lower network, fresh supervised Adam lr=0.0003, 32 epochs and minibatch1024. Shams permute A labels once and use the identical shuffle sequence and step budget. Preflight uses four epochs. Final epoch only; no model selection.
- Independent frozen batches A/B each contain eight complete 1200-step episodes; evaluation uses 16 fresh paths, both deterministic and sampled lower execution. Upper/gate are deterministic throughout. Preflight uses two paths per role and 300-step episodes.
- Credit: A's normalized episode-MC or original option-GAE score, including original entropy, versus B's raw reward-only episode-MC score. B uses undiscounted full-task return-to-go with leave-one-episode-out time baseline, summed per episode. Negative BC-loss gradients against the same B score are secondary diagnostics.
- Full eight roots: 310011,310023,310037,310049,310061,310073,310089,310101. Fresh Stage51 A/B/evaluation seeds are disjoint. All raw trajectories, gradient arrays and final checkpoints stay on the servers.

## Endpoints And Accounting

Eight fixed primary endpoints: deterministic return differences for each teacher/clone versus frozen, each clone versus its shuffled-label control, and MC/GAE independent-gradient cosine. Equal-root paired percentile bootstrap, 65536 draws, Bonferroni eight-endpoint intervals. Sampled returns and BC gradient alignment are descriptive, not alternative primary selectors.

Per root: 288000 native steps, 240 trajectory audits, 1280 supervised Adam steps, five score backward calls and one Riccati solve. Full total: 2304000 steps and 1920 audits. No original RL actor/value optimizer steps and no extra verification simulation. Preflight: 9600 steps, 32 audits and 16 supervised steps. Eight persistent rollout workers plus one learner, 9 CPU/12 GiB per full root, dynamic scheduler placement on node001-node006. Unit/preflight/aggregation use 2 CPU/4 GiB.

Waypoint teacher improvement establishes plan-compatible headroom. Waypoint clone must improve over both frozen and sham for same-network supervised utility. Task controls identify headroom that may require changing the upper/lower responsibility assignment. Positive gradient cosine establishes local cross-batch agreement, not finite-update return improvement. Freeze before native outcomes; no seed extension or outcome-based retuning.

## Limitations

These reused roots provide conditional development evidence, not independent confirmation. The LQR design is an approximate control model; teacher failure does not establish a performance ceiling. Behavioral-cloning failure may reflect optimization or behavior-support shift and does not establish unlearnability. Teachers and successful clones are positive controls, not validation of learned Freq-HRL or its frequency-separation claim. Bootstrap intervals from eight roots remain exploratory.

CARE implementation: [SciPy solve_continuous_are](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_continuous_are.html).
