# Freq-HRL MuJoCo Validation Scope

Date: 2026-08-08

## Purpose

The Trading and Transit experiments are application-facing but do not by
themselves establish that Freq-HRL is a general continuous-control algorithm.
The MuJoCo path adds standard Gymnasium dynamics without replacing or
reimplementing the physics engine.

## Controlled Comparison

The initial method registry contains four capacity-controlled policies:

- `flat_ppo`: one primitive-rate actor and one task-return critic;
- `generic_hrl`: upper and lower SMDP actors with raw observation and causal
  first-difference features at both levels;
- `freq_hrl_no_leakage`: the same SMDP architecture, optimizer, and action
  composition with slow/mid features routed upward and mid/high features
  routed downward;
- `freq_hrl`: the routed model plus a primal-dual cost on the lower action's
  causal low-frequency component.

The hierarchical methods have identical trainable architectures. The flat PPO
hidden width is selected analytically to match the full hierarchical trainable
parameter count within 3%. All methods use the same environment seeds,
primitive-transition training budget, standard 1000-transition episode
horizon, action bounds, reward, checkpoint-selection paths, and held-out
paths. Training collection continues through deterministic seeded resets after
early termination, so Hopper or Walker failures cannot silently give one
method fewer optimizer samples. Evaluation remains one standard episode and
never continues after termination.

Checkpoint validation is evaluated every four optimizer iterations rather
than after every update. The robust selector averages eight validation
observations, so its default smoothing horizon spans approximately 32 training
iterations. The initial observation, every evaluation event, the final
evaluation, and every non-evaluation training iteration are all retained in
the training-history artifact.

Upper actions are slow action anchors held for a fixed macro interval. Lower
actions are bounded residual corrections. The shared
`FrequencySeparatedActorCriticPPO` records one discounted SMDP transition per
upper interval and one transition per primitive lower action; upper log
probabilities are never repeated as primitive transitions.

## Evaluation Conditions

Primary evaluation must include the unmodified `standard` task. Mechanism
stress tests add deterministic normalized-actuation disturbances:

- low-frequency sinusoid;
- high-frequency sinusoid;
- mixed low/high signal plus a persistent midpoint bias;
- held-out frequency chirp.

Custom disturbances are secondary mechanism diagnostics and must not be
presented as standard MuJoCo scores.

The frozen checkpoint selected on disjoint validation seeds is hashed before
held-out evaluation. That same checkpoint is then evaluated on every registered
disturbance mode, and its parameter hash is checked again afterward. This
prevents condition-specific checkpoint selection and detects accidental
test-time mutation.

## Development Pilot

The scheduler pilot is deliberately separate from confirmatory evidence:

- preflight: one environment, four methods, one development optimizer seed,
  two updates, and a 64-transition horizon;
- pilot: three environments, four methods, and three development optimizer
  seeds, using 64 updates and 512 training transitions per path;
- placement: one core and 2 GiB per cell, dynamically assigned across
  `node001` through `node006` with no required-node pin;
- evaluation: the standard task is primary and all four deterministic
  frequency disturbances are secondary diagnostics.

Pilot optimizer seeds are development-only and cannot be reused in a formal
comparison. Pilot results may choose a training budget or reveal a broken
method, but they cannot support a paper performance claim.

Protocol v4 separates a trajectory trace boundary from an actual MDP
termination. Early environment termination receives zero bootstrap; Gymnasium
TimeLimit truncation and a fixed collection-budget boundary use an explicit
next-state critic value while still stopping GAE propagation across the reset.
The 512-transition training budget is divisible by the 16-step upper period,
so a collection boundary cannot silently cut one method at a different point
inside its macro action. Cost GAE uses its own next-cost-value bootstrap, and a
zero-violation rollout cannot inject a policy gradient from an uncalibrated
cost critic. Lower-frequency costs are dimensionless squared excess over a
registered causal RMS budget rather than an unscaled power difference.
Protocol v2 pilot results remain development evidence only and cannot be mixed
with v4 confirmation.

Every v4 cell records two distinct integrity hashes: a model-parameter hash
checked before and after evaluation, and a SHA-256 of the serialized checkpoint
file. The legacy `frozen_checkpoint_sha256` field is retained as an alias for
the parameter hash only; it must not be mistaken for a file hash.

## Current Evidence Boundary

The source-bound v2 pilot completed all 36 development cells. Its integrity
audit is valid, but its performance result is not sufficient: Freq-HRL beat
flat PPO consistently only on Walker2d and lost consistently on HalfCheetah and
Hopper. The pilot also exposed the truncated-trajectory and leakage-cost
contracts repaired in v4. These data select the next protocol; they are not a
paper performance result. Paper evidence requires a source-bound training-
budget plan, fresh independent optimizer seeds, at least three standard MuJoCo
tasks, untouched held-out paths, and multiplicity-controlled paired
comparisons.

The legacy `jtl110cpu` scheduler records are excluded from this evidence path.
Their remote process termination could not be confirmed after SSH handshake
failure, so long-lived scheduler labels are not accepted as proof of active or
completed computation.
