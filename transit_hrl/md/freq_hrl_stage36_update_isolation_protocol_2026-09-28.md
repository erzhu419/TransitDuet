# Stage-36 Component Update Isolation

Stage-35 trained policies lost selection return and all learned cells selected
iteration0. This experiment isolates update responsibility before changing
credit, reward, architecture or thresholds. It is not a new success gate.

Freeze a 2x2 factorial: no updates, gate-only, upper/lower-only, joint. Add
equally fine-tuned fixed50. Each enabled level updates both actor and critic;
disabled levels receive no optimizer calls and retain exact initial weights.
All arms start from the same Stage-33 controller and Stage-35 gate initialization
per root, with fresh optimizers. Training samples all actions even when their
parameters are frozen; deployment is deterministic as in Stage-35.

Unchanged native loop: 25-step gate checks, maximum age100, no previews,
primitive-step lower feedback, gamma0.995, lambda0.95, four PPO epochs,
learning rate3e-4, hidden128. Upper/gate reward is native dense reward minus
1 per actual upper call; lower keeps the original intrinsic reward.
NumPy minibatch shuffle uses SeedSequence(36, optimizer-root, iteration).
Torch initialization and rollout sampling retain their Stage-35 seeds.

Eight inherited roots: 310011/310023/310037/310049/310061/310073/310089/310101.
Each cell runs 128 iterations x8 training episodes, horizon1200. Fresh paths
use base7400000 + root-index x10000: training+1..1024, selection+2001..2008,
evaluation+3001..3032. Selection evaluates iterations0/32/64/96/128, maximizing
charged utility then minimizing ISE. Both final128 and selected weights are
saved and separately evaluated on the same untouched 32 paths. **Final128 is
primary**; selected results cannot replace an unfavorable final result.
All 40 cells use 54,192,000 method steps including factual source replays.

Seven primary contrasts on final weights: return change for gate-only,
controller-only and joint versus frozen; return interaction
joint - controller-only - gate-only + frozen; joint return increase, ISE
reduction and planning-call savings versus trained fixed50. Equal-weight
optimizer-root bootstrap, 65,536 draws, seed(36,36039), two-sided Bonferroni
intervals across seven contrasts. An interval above/below zero identifies
a positive/negative effect; crossing zero is inconclusive. Utility, call
counts, selected cohort and selection curves are secondary diagnostics.

Separate preflight root310001: five arms, two iterations, one training path
per iteration, two selection and two evaluation paths per cohort. Readiness
depends only on freeze/update execution, trajectory/accounting and native
checkpoint replay, never on performance. Full jobs use scheduler's dynamic
node001-node006 pool, eight rollout workers plus one learner, 12GiB RAM.
Stage source directories only; raw trajectories and weights remain server-side.
Pull compact summaries only. Verification replays add 96,000 steps full
or 3,000 preflight, reported separately from method cost.

## Limitations

Conditional development on reused controllers, not independent confirmation.
No root exclusion, budget extension or outcome-driven retuning. A factorial
effect diagnoses this training recipe; it does not establish domain-general
Freq-HRL superiority, total compute savings or the cause within upper/lower.
Earlier failed experiments remain failed.
