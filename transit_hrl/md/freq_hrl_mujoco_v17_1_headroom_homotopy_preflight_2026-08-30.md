# MuJoCo v17.1 Headroom-Homotopy Development Preflight

## Why v17.1 exists

The frozen v17 screen established the raw-action mechanism but rejected the
architecture as a whole. The exact lower zero-DC projector closed every complete
upper macro interval and reduced raw lower LF power in all nine environment-seed
cells, while reward failed badly in Hopper and in two HalfCheetah cells. The v17
candidate also reached nominal additive clipping rates as high as 27.3% in an
individual cell. Therefore v17 supports pre-saturation command separation only;
it does not support plant-level separation or reward preservation.

v17.1 changes the control architecture rather than retuning the old gate:

1. The frozen smooth upper-plan suffix defines the feasible lower-action
   headroom at every primitive step.
2. The lower action is a causal homotopy between the direct feasible action and
   the exact zero-DC projection.
3. At router strength zero, actor state, promotion gain, and executed control
   are exactly the smooth-direct path. At strength one, each complete macro has
   exact lower zero sum.
4. The mean latent lower demand from the preceding macro is exposed causally to
   the next upper decision and can be promoted into the replanned upper target.

## Frozen development matrix

- Environments: HalfCheetah-v5, Hopper-v5, Walker2d-v5.
- Training disturbances: standard, low-frequency, high-frequency, mixed.
- Held-out disturbances: the four training families plus OOD chirp.
- Arms: smooth direct, headroom exact, headroom homotopy, homotopy with 0.5
  promotion gain, and homotopy with 1.0 promotion gain.
- Capacity: every arm observes one router-strength scalar and uses the same
  network dimensions.
- Optimization: 128 iterations, 512 sampled transitions per training rollout,
  reward-only checkpoint selection on crossed selection roots.
- Evidence role: development preflight only; one optimizer root cannot support
  uncertainty or publication claims.

All seed roots, source identity, thresholds, arm definitions, and scheduler
contracts are frozen in
`scripts/mujoco_v17_1_headroom_homotopy_preflight_spec.py` before dispatch.

## Advancement rule

One arm is selected globally, never separately per environment. An arm can
advance only when:

- trained-checkpoint, exact macro completion, active projection, exact
  responsibility reconstruction, nominal headroom, and promotion-mechanism
  checks hold in all three environments;
- reward is within the 5% noninferiority floor in all three environments; and
- upper HF nonworsening, raw lower LF reduction, raw-to-latent lower LF
  reduction, and joint frequency-merit reduction hold in at least two of three
  environments.

Among eligible arms, selection first maximizes the worst-environment reward
margin above the floor, then the median joint frequency-merit reduction. A
supported preflight only authorizes a new multiseed development run on fresh
roots. It does not revise manuscript claims by itself.

## Artifact and execution contract

- Scheduler: scheduleurm only.
- Placement: dynamic across node001-node006; no task binds to a specific node.
- Resources: one CPU and 1024 MB RAM per independent cell.
- Local sync: `cell_summary.json`, `evaluation_rows.csv`, and the server artifact
  locator only.
- Server-only: checkpoint and full training history.
- Slurm and jtl110cpu are not used.
