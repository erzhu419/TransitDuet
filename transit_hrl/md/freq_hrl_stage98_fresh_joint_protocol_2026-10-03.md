# Stage98: Fresh Teacher Joint/Lower Donors

Stage97 qualified all eight new-teacher decoders. Rebuild the native mean learners using the same Stage88 core, not old trained weights.

- Source: full Stage96 clones/forecasters and full Stage97 fixed decoders for roots 410011, 410023, 410037, 410049, 410061, 410073, 410089, 410101. Preflight uses root410011's full sources, never root410001 or short decoder artifacts.
- Keep joint-call, joint-level and lower-only learners; two periods 50/100; base bounded-U0/L0 and zero-plan controls. No change to MC return credit, leave-other-rollout-out baseline, model/history inputs, Gaussian std freeze, or first-feasible decoder.
- Full training: eight fresh updates, each with two disjoint batches of 16 scenarios x two action-noise replicas (64 episodes). All active actors use the same complete batch. No reward normalization, optimizer/critic fitting, intermediate evaluation, best checkpoint or radius search.
- Keep radius 0.001, joint-call upper allocation 0.5 and lower 1-0.5/period; joint-level half each, lower-only 1. Fresh Stage98 scenario/noise/evaluation rosters; unchanged Stage80 noise mapping.
- Preflight: two updates x eight episodes per method/period, four final eval paths per variant, H300; 136 episodes / 40,800 steps, no checkpoint writes. Full: 3,392 episodes / 4,070,400 steps and six fixed-final checkpoints per root; cohort 27,136 episodes / 32,563,200 steps and 48 server-only checkpoints.
- Keep all 20 reward contrasts in one equal-root 65,536-draw Bonferroni family, unchanged bootstrap seed/indices. Report four primary joint-call versus lower-only/joint-level contrasts at both periods. Preflight is mechanical only; no root/period selection or cross-stage pooling.
- Retain every fixed-final joint-call donor for the next matched lower rebuild even if this intermediate comparison is negative. The staged-upper confirmation remains a later separate test.
- Scheduler: dynamic node001-006, 3 CPU/3GB preflight and 9 CPU/8GB full. Only compact JSON/completion markers return locally; full launch waits for native preflight qualification.

Scope: new-teacher, fixed-std/decoder MC mean learning, not full actor-critic or frequency-superiority evidence. Stage67 critic-credit HOLD and Stage93 negative findings remain unchanged.
