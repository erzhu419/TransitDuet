# MuJoCo v17 Zero-DC Plan Outcome

## Decision

`zero_dc_plan_screen_not_supported`

This is valid development evidence, not confirmatory evidence. The frozen
analysis used nine environment-by-optimizer-seed units, three capacity-matched
arms, and 40 held-out paths per arm. The 27 unique cell signatures each have a
successful scheduleurm attempt across node001-node005.

The first worker omitted scheduleurm's success token. Consequently, 36 earlier
attempts exited with code zero but were marked failed. The repaired worker reused
the complete server artifacts, exported only summary/CSV files, and produced one
`done` attempt per signature. These false-negative attempts remain in scheduler
history; they are not hidden or counted as independent runs.

## Frozen Result

- Trained checkpoint: 9/9 cells.
- Exact zero sum on every complete 16-step lower macro: 9/9 cells.
- Active zero-DC projection: 9/9 cells.
- Exact registered responsibility reconstruction: 9/9 cells.
- At least 10% raw lower-LPF32 reduction versus smooth direct: 9/9 cells.
- At least 10% raw lower-LPF32 reduction versus the latent proposal: 9/9 cells.
- At least 10% raw normalized joint-merit reduction versus hold direct: 9/9 cells.
- Smooth-upper HPF8 ablation improvement: 7/9 cells.
- Candidate upper-HPF8 relative reduction: 5/9 cells.
- Candidate absolute upper-HPF8 budget: 6/9 cells.
- Held-out reward noninferiority versus hold direct: 3/9 cells.
- All cell gates: 1/9 cells.
- Per-environment two-of-three support gate: 0/3 environments.

The raw frequency effect is large rather than marginal. Median raw lower-LF
reduction versus the smooth direct arm was 85.2% in HalfCheetah, 73.7% in
Hopper, and 92.9% in Walker2d. Median raw joint-merit reduction was 87.4%,
80.8%, and 92.6%, respectively.

The reward tradeoff is also large. Only one HalfCheetah seed, no Hopper seed,
and two Walker2d seeds met the five-percent reward floor. Two HalfCheetah seeds
lost more than 40% of direct-control return, while Hopper candidate return was
roughly 18-33% of its matched direct control.

## Diagnosis

The v17 architecture fixes the accounting failure exposed by v16.2: it changes
the command sent toward the plant, makes every completed lower macro exactly
zero mean, and sharply reduces raw lower-frequency power. The failure is now a
control and optimization failure, not a gauge-identifiability failure.

Full-strength projection from the first update removes persistent lower control
before the upper policy has learned to absorb it. The upper actor then
compensates. This explains why the smooth-upper ablation improved upper HPF8 in
seven cells, while the full candidate improved it in only five and substantially
worsened it in Hopper and two Walker2d seeds.

A fixed hard 16-step zero-mean constraint is therefore too abrupt as a training
geometry. More seeds would estimate the same tradeoff more precisely but would
not remove it. The next version needs a training-time homotopy that gradually
increases projection strength, makes that strength and the exact repayment debt
observable, and uses persistent lower demand to trigger causal upper replanning.
Evaluation can still require the exact full-strength zero-DC invariant.

## Saturation Boundary

The projector constrains the lower command before clipping the additive
`upper + lower` command. Candidate median additive saturation was 5.6% in
HalfCheetah, 7.9% in Hopper, and 15.5% in Walker2d; one candidate cell reached
27.3%. Thus this screen supports pre-saturation command separation only.

A later plant-level claim requires a joint feasible projector that reserves
upper-dependent actuator headroom. Exogenous disturbance saturation remains a
separate plant nonlinearity and must be reported rather than assigned to either
policy level.

## Claim Boundary

Allowed: on frozen development roots, exact causal zero-DC projection enforced
the complete-macro lower invariant and reduced registered pre-saturation raw
lower-LF and joint frequency merit in all nine cells.

Forbidden: v17 validates reward no-tradeoff, cross-environment upper/lower
frequency separation, plant-level separation after saturation, fresh-seed
confirmation, or a final Freq-HRL algorithm.
