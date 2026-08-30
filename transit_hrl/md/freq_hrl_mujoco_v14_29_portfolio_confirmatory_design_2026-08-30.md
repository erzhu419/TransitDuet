# MuJoCo v14.29 fresh-seed portfolio confirmation

## Confirmatory target

v14.28 established a development result on one optimizer seed per environment:
the domain-neutral portfolio rejected unstable actor transactions and selected
function-preserving responsibility routing in all three validation cells. The
result was not a multi-seed claim. v14.29 freezes the portfolio before training
or inspecting any new anchor outcome and tests optimizer-seed replication.

## Frozen algorithm

- Algorithm revision: `fc7fa8d8c1e55325af9cb32efece3e0cfc2bbd3c`
- Freq-HRL source manifest: `02f3ba95376021dff0aa11f30d46dd6159e63b55a1d2678d6011ea350745af39`
- Environments: HalfCheetah-v5, Hopper-v5, and Walker2d-v5
- Replication units: 16 fresh optimizer seeds in each environment
- Total anchors and portfolio cells: 48 each

The anchor optimizer, train, selection, and development-evaluation seeds are
absent from the earlier mechanism campaign. Anchor training retains the frozen
capacity-matched v14.17 training profile and router strength `0.5`. Portfolio
critic, holdout, design, and validation roots are separately fresh and mutually
disjoint from every anchor role and v14.20-v14.28.

## Frozen selector

Every cell receives the same 15 transactions: five orthogonal paired-FD actor
steps and ten function-preserving router strengths. Thirty-two design roots are
crossed with four disturbance modes and split into two fixed 16-root folds. A
candidate must pass both folds and pooled design before one candidate enters 32
fresh-root validation.

Router transactions have an additional hard gate. Executed-action, reward, and
latent-policy trace digests must match the paired baseline on every design and
validation path, with zero reward-mean and episode-return delta. A router
candidate that lowers frequency merit but changes behavior is rejected.

## Statistical gate

The unit is the optimizer seed, not a rollout path or environment. Abstention,
no eligible design candidate, and validation failure all count as failures. For
each environment independently, the two-sided 95% Wilson lower confidence bound
for validation success probability must be strictly above `0.5`. With 16 seeds,
this requires at least 12 supported cells in each environment. Every selected
router transaction must also pass exact trace invariance.

The interval therefore measures replication across fresh optimizer seeds
conditional on the one preregistered validation-path panel. Rollout roots are
paired stress conditions, not additional independent statistical units.

Only when all three environment gates pass is the portfolio reported as
confirmed. The claim is responsibility-restoration reliability under the
frozen MuJoCo stress protocol. It is not a claim of reward improvement, since
function-preserving routing intentionally leaves physical behavior unchanged.

## Scheduler contract

All work uses `scheduleurm` with dynamic placement across `node001-node006` and
`require_node=None`; Slurm and login-node training are prohibited. Anchor cells
request one CPU core. Portfolio cells request 24 cores and 16 GB RAM. The
launcher explicitly stages `scripts/` and `freq_hrl/` so cached remote cwd state
cannot execute stale source.
