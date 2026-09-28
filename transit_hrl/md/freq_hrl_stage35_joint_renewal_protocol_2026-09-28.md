# Stage-35 Joint Renewal Training

Stage-34's 150-step deployment blocks saved calls but lost 74.64 return and
increased ISE by 85.23% versus fixed50. Stage-10's fixed-budget termination
also failed. This development experiment changes training, not their outcomes.

Freeze four treatments: learned-history renewal, equal-capacity current-only
renewal, fixed50, fixed100. All start from the same Stage-33 controller per
root with fresh optimizers. Upper/lower train jointly; learned treatments
also update the existing SMDP Bernoulli promotion actor/critic. Gate checks
every 25 steps, maximum plan age 100; lower runs every primitive step.
Gate observes only causal state/history, waypoint error and clocks. There is
no candidate preview: the planner is called only when a new plan is executed.

Upper and gate receive native dense reward minus 1 reward unit per actual
upper call; lower retains the existing intrinsic objective. Gamma 0.995,
GAE lambda 0.95, learning rate 3e-4, hidden width 128 and four PPO epochs use
the inherited trainer. Duration-aware discounts and actual option boundaries
apply. Do not clamp deployment to a post-hoc call quota.

Eight roots: 310011, 310023, 310037, 310049, 310061, 310073, 310089, 310101.
Each cell: 128 iterations x 8 fresh training paths x 1200 steps; selection
on 8 separate paths at iterations 0/32/64/96/128, maximizing return minus
actual call cost, then minimizing ISE. Evaluate 32 untouched paths. Initial
checkpoint may win and must be reported. All roles are frozen in the spec;
methods share environment paths and initial controller weights, not subsequent
policy trajectories. Total new method cost: 42,124,800 primitive steps,
including the controller factual replay per cell. Inherited Stage-33 training
is reused, not represented as fresh from-scratch evidence.

Primary endpoints: learned-history return increase, ISE reduction and actual
call savings versus independently fine-tuned fixed50; cost-weighted utility
increase versus fine-tuned fixed100 and learned-current. Equal-weight root
bootstrap, 65,536 draws, seed (35,35039); two-sided percentile Bonferroni
intervals over these five endpoints. Joint gate requires every lower bound
strictly positive. Cost-weighted utility alone is not no-tradeoff evidence.
No root deletion, seed extension, threshold/cost retuning after outcomes.

Preflight: separate root310001, four treatments, two iterations, 1 training
path/iteration, 2 selection and 2 evaluation paths. Only execution, causal
features, accounting and nonzero optimizer updates determine readiness.
No performance-based preflight selection. Full runs use scheduler's dynamic
node001-node006 pool, 8 rollout workers + 1 learner per task, 12 GiB RAM.
Only source directories are staged. Existing results/weights are read in
place on the shared server; pull only compact JSON, not raw NPZ/checkpoints.

## Scope

This is conditional algorithm-development evidence. Earlier failed gates
remain failed; a positive result requires independent confirmation next.
Planning-call savings do not establish total compute savings. Gate inference
counts and per-episode timings are recorded alongside upper/lower inference;
this protocol does not make a wall-clock speedup claim.
