# MuJoCo v14.26 robust paired finite-difference preflight

## Decision context

v14.25 matched the actor update to the output-bias intervention and supported
Hopper and Walker, but HalfCheetah still rejected every candidate. Its action
critics were identifiable, while their actor derivatives were unstable: the
lower ensemble median cosine was positive but one critic pair strongly
disagreed, and an RMS step of only `1e-7` sharply increased frequency merit.
The remaining failure is derivative estimation, not action relevance or update
scope.

## Frozen mechanism

Collection retains the five deterministic paths per environment path: control,
upper output-bias plus/minus, and lower output-bias plus/minus at RMS `0.25`.
Upper targets use eight-decision duration-discounted native cost returns and
lower targets use 32 decisions.

The actor direction no longer differentiates a fitted MLP critic. For every
root and disturbance mode, the plus/minus cost difference supplies an
antithetic finite-difference estimate for the intervened level. The coordinate
median across paths defines each train direction. Upper and lower blocks are
normalized to equal RMS before the joint negative-cost update.

The critic ensemble remains a data-quality diagnostic. Both levels must retain
positive holdout R2 and positive fixed action-permutation MSE gain. Independently,
the paired direction must have positive train-versus-holdout cosine overall and
inside every disturbance mode. No design candidate is evaluated if either gate
fails.

Exact design evaluates output-bias RMS steps `1e-7`, `1e-6`, `1e-5`, `3e-5`,
and `1e-4`. Eligibility still requires zero reward violations, frequency-merit
reduction of at least `1e-4`, and worst frequency violation no greater than
three times baseline. Selection uses design paths only; one selected candidate
is then evaluated on untouched validation paths.

## Independent roles

Eight critic-train roots, eight critic-holdout roots, 16 design roots, and 16
validation roots are fresh and mutually disjoint, and do not occur in
v14.20-v14.25. Crossing each critic root with four disturbance modes yields 32
independent path directions in both train and holdout. The enlarged holdout is
needed because the six-dimensional actor-bias blocks cannot be diagnosed
reliably from only four directions per mode.

## Execution contract

The three cells are HalfCheetah, Hopper, and Walker at optimizer seed
`4196455150`. Each requests 24 CPU cores and 16 GB RAM. Scheduler placement is
dynamic across `node001-node006`, with `require_node=None`; no Slurm or login-node
execution is permitted.

## Evidence boundary

All three independent validation cells must pass before this direction estimator
can enter the shared actor-critic. This is adaptive development after v14.25,
not confirmatory evidence. After outcome access, roots, directions, steps,
thresholds, and eligibility rules remain frozen; a failed environment is
reported as a method boundary rather than tuned on its validation paths.
