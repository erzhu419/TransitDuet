# V60: separate learning from deployment sampling

Freeze V59 split-projection post-64-update actors and the B200 upper/deployment
background. No optimizer steps, new training, or main-policy switch.

## Fixed Diagnostic

Reuse the four V59 development contexts. For each, evaluate initial/updated
lower actor by deterministic/stochastic deployment. Stochastic pairs reset the
same isolated generator using `train_seed + 59000 + 1009 * replicate` for three
replicates. This gives 32 unique full-day rollouts. First run unit tests and an
exact initial/deterministic replay, then batch the matrix through scheduler on
node001--node006. The smoke replay is reused, not run twice.

Preserve seven holding bins, physical fleet, upper weights, frequency inputs,
pending-command state and the fixed censor horizon. Initial actor probabilities
must be bit-exact; its deterministic path and CSV metrics must match V56.
All modes retain once-only system accounting. Checkpoint/trajectory data stay
on the server; retrieve only small completion markers and aggregate status.

## Analysis / Next Step

Report paired actor effects within each deployment mode, sampling effects within
each actor, and their interaction. Average sampling replicates inside each
train/demand context before averaging the four contexts. Report journey,
wait/onboard, unrounded CV and mean squared CV, holding, readiness, passenger
exposure and backlog. Average squared CV directly for the constraint, rather
than squaring the averaged CV. Calculate each rollout's squared CV with V59's
completed-arrival moment accumulator, including its original reference budget.
Keep CV feasibility relative to the original deterministic reference separate
from improvement over the same-mode initial actor. Use the result to decide
whether to repair the learning update, constraint value/dual, or deployment mode.

## Limitations

These are development contexts, not independent held-out confirmation. Three
sampling replicates are not three training seeds. Post-64 actors are not V59's
post-32 actor used for its stochastic collection day. No significance or policy
promotion is inferred from this diagnostic; no constraint budget is relaxed.
