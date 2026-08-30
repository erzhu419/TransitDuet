# MuJoCo v18.1 Causal Actor-State Dataset Design

## Purpose

V17.14 closed the complete frozen linear FIR actor grid at 6/7 actor-floor
recovery. V18.1 adds the missing causal state input needed to test a nonlinear,
state-conditioned residual actor. It is a data-export phase, not a model screen.

## Frozen Source

The actor trace interface is frozen at Git revision
`f94f1f4a6a35d70f6b6d144bd886644e7efb2393` with Freq-HRL source manifest
`b06a97fc8f18129a2e1a9c23a52acb01ea683d1445c672074b9a6901ece23af6`.
Each path must exactly replay the registered v17.4 reward, executed-action, and
latent-policy traces before its state arrays are accepted.

## Data Contract

The panel remains the same 3 environments, 5 disturbance modes, and 8 reused
selection seeds, for 120 paths. At every step, the exporter records the
pre-transition observation, lower-policy state, disturbance bands' source
signal, upper and lower latent proposals, macro-decision indicator, episode
step, effective responsibilities, total action, and executed action. The lower
policy state is already causal and contains observation, current/past exogenous
bands, upper context, and router contexts used by the frozen policy.

V18.1 does not read actor-floor labels or v17.12 targets. Those remain a
separate server-only input for the later grouped model-selection phase. This
separation prevents target availability from becoming an online feature.

## Storage And Execution

The 120 NPZ paths remain under `.server_artifacts` on node003 beside the source
checkpoints and targets. Scheduleurm uses one CPU core and 1536 MB per path;
only compact JSON markers are synchronized locally. Slurm is not used.

## Gate

Every path must have the expected environment-specific dimensions, finite and
aligned arrays, exact upper-plus-lower reconstruction, an initial macro
boundary, an unchanged checkpoint parameter digest, and exact v17.4 replay.
Any failed path invalidates the dataset and blocks state-conditioned selection.

## Claim Boundary

This phase establishes only a causal reused-path state dataset. It makes no
model-selection, reward, online-learning, fresh-seed, or manuscript claim.
