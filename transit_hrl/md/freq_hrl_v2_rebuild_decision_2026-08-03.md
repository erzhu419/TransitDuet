# Freq-HRL v2 Rebuild Decision

Date: 2026-08-03

## Decision

Freq-HRL will not be closed as a paper on top of the v1 evidence package. The
project will first replace the training and evidence foundations. Existing v1
artifacts remain reproducibility records, but they are not eligible as
confirmatory evidence for v2 claims.

## Non-negotiable algorithm contracts

1. The upper planner never receives raw high-frequency observations. It may
   receive causal low-frequency features, forecasts, uncertainty, compressed
   high-frequency energy/persistence summaries, promotion state, and leakage
   feedback.
2. The lower controller receives the current upper plan and local causal
   high/mid-frequency context. It does not receive future low-frequency plans
   or upper value estimates.
3. Upper and lower policies have separate transition streams, PPO ratios,
   advantages, critics, and optimizers.
4. One upper transition represents one macro action. Its reward is accumulated
   inside the interval and its bootstrap discount is `gamma ** duration`.
5. Promotion terminates a macro interval early and requests a new upper action.
   It never duplicates the old upper log probability on lower transitions.
6. Credit follows responsibility. Upper rewards represent strategic outcomes;
   lower rewards represent execution, tracking, and local correction outcomes.
7. Leakage is measured from realized action effects. A lower cost critic and
   primal-dual update enforce the constraint without replacing observed
   environment outcomes.

## Non-negotiable evidence contracts

1. Headline tables, claim matrices, figures, and abstracts use observed raw
   outcomes only.
2. Hand-set outcome adjustments are projections. They may appear only in a
   separately labelled sensitivity appendix and can never change claim status.
3. A confirmatory experiment has one frozen configuration, one immutable code
   revision, declared primary outcomes, declared stress strata, and untouched
   evaluation units.
4. Tuning seeds and confirmatory seeds are disjoint. Simulator random seeds are
   not presented as independent real-world routes, days, agencies, or markets.
5. Baselines use matched observations, action constraints, training steps,
   parameter budgets, tuning budgets, and evaluation units.
6. Statistical support requires an effect direction, confidence interval,
   practical effect threshold, multiplicity policy, and hierarchical unit of
   analysis. Best-of-version artifact selection is prohibited.
7. A clean checkout must reproduce every paper table and figure from a tracked
   manifest or an immutable external release.

## Migration order

1. Build and test the asynchronous SMDP core.
2. Migrate learned trading Freq-HRL and establish protocol compliance.
3. Migrate Transit surrogate and native episode loops.
4. Quarantine v1 outcome-adjusted evidence and replace the claim gate.
5. Correct return, Sharpe, execution PnL, and hierarchical statistics.
6. Implement true flat and generic-HRL learned baselines.
7. Register and run synthetic mechanism, Transit, market, and order-book
   confirmatory matrices.
8. Write a new manuscript only from the resulting frozen evidence ledger.

## Paper claim boundary during rebuild

Until all migration gates pass, the valid description is "Freq-HRL v2 research
implementation under confirmatory validation." It must not be described as a
fully validated domain-general algorithm or as a deployment-ready controller.

## Implementation status

As of commit stage 3, Quant/trading and Transit surrogate both call
`train_frequency_separated_ppo` and emit separate upper/lower SMDP
trajectories. Transit surrogate uses one upper transition per timetable macro
interval, primitive lower transitions, responsibility-specific credits, and a
continuous learned promotion blend between the active and candidate plans.

Actor, reward-critic, and lower cost-critic optimizers are separate. This is a
required numerical invariant: macro reward scale may change critic loss but
must not suppress the PPO actor step through shared global gradient clipping.
The corresponding scale-invariance regression test is mandatory.

Native Transit now instantiates `FrequencySeparatedActorCriticPPO` and calls
`apply_smdp_updates` with native upper/lower streams. Upper transitions use the
runner's backfilled dispatch/timetable reward and a duration equal to the
number of causally assigned lower holding events. Trip-ID matching is used
first; a unique behavior decision ID handles native fallback trip keys. A
trajectory is valid only if every lower transition is assigned, every upper
transition has positive duration, behavior metadata is present, and no
zero-policy-blend heuristic override enters an on-policy claim. Lower events
are grouped by native trip before GAE construction, and every trip boundary is
terminal for bootstrap purposes; interleaved buses therefore cannot leak value
targets into one another.

The post-migration real copied-runner smoke (seed 9) produced 94 upper
transitions and 4,971 lower transitions across 262 independent trip
trajectories. All lower events were causally assigned (1,779 exact trip-ID
matches and 3,192 decision-ID fallback matches), every upper transition had a
positive duration, and separate upper/lower PPO updates were finite. The
source-level shared-core migration gate is therefore supported; this is an
implementation result, not performance evidence. Legacy
joint-optimizer v2 checkpoints are not optimizer-state compatible with this
stage and must not be mixed into confirmatory runs.

## Statistical contract

All v2 treatment effects are paired before uncertainty estimation. Duplicate
rows for the same variant and pair key are fatal rather than silently
overwritten. When a seed is reused across sources, scenarios, or stress
conditions, that seed is the default independent cluster: within-seed effects
are averaged first and the cluster means are bootstrapped. Reports expose both
the number of paired rows and the number of independent clusters. Improvement
intervals are exact sign transforms of the treatment-minus-control interval.

Headline multiplicity is controlled within each registered claim family using
Holm-Bonferroni correction of the paired sign-test p-values. A large nominal
sample count cannot bypass this gate. Existing v1 reports that counted repeated
source/scenario rows as independent or used an unadjusted p-value are
exploratory artifacts and must be regenerated before they can support a v2
paper claim.

## Trading metric contract

Every v2 trading row carries `metric_contract_version=trading_metrics_v2`.
Portfolio environments consume simple returns, never log returns masquerading
as simple returns. Net bar return is gross portfolio return minus one observed
transaction-cost charge, and the reported equity path must reconstruct from
that series within numerical tolerance. Order-book replay uses fixed-notional
normalized PnL increments and additive reconstruction.

`sharpe` is now an alias for annualized Sharpe with an explicit
`periods_per_year` and sample standard deviation (`ddof=1`). The old
`sqrt(episode_length)` ratio is retained only as
`episode_information_ratio`. Sortino uses downside semideviation. Drawdown
includes the initial equity point. CAGR and Calmar are undefined for horizons
shorter than 0.25 observation-years; short synthetic episodes report
`episode_return_to_drawdown` instead. Legacy trading rows without this contract
are ineligible for baseline headline claims and must be rerun.

## Capacity-matched learned baselines

The PPO-family comparison no longer mixes the v2 SMDP trainer with the legacy
joint-timestep PPO. `freq_hrl`, `generic_hrl_ppo`, and `flat_ppo` use identical
model dimensions, actor/critic parameter counts, optimizer settings, epoch and
minibatch budgets, train/evaluation seeds, and financial metric contracts.
Freq-HRL alone receives frequency-responsibility observations and promotion.
Generic HRL uses raw-signal transforms at both levels with the same macro
clock. The flat baseline removes temporal abstraction by replanning both
factorized action heads at every primitive step and gives both heads the task
reward. It is therefore reported precisely as a capacity-matched factorized
joint-action flat PPO baseline, not as a single-actor implementation. The
default comparison disables the handcrafted Freq-HRL actor prior; prior-enabled
results belong in a separate ablation.

## Complete off-policy learned baselines

The strong learned-baseline matrix now runs local, auditable implementations
of SAC and TD3 rather than registering them as missing external dependencies.
Both are single-level flat policies with a joint target-weight/execution-speed
action, persistent replay, twin Q critics, soft target updates, and deterministic
held-out evaluation. SAC includes a squashed Gaussian actor and learned entropy
temperature; TD3 includes target-policy smoothing and delayed actor updates.

All five policies use the same market paths, transaction costs, held-out seeds,
primitive environment-step budget, and `trading_metrics_v2`. The raw baselines
receive the same underlying raw signal transforms, position, active target,
target gap, and episode progress. Flat PPO, SAC, and TD3 also share the same
bounded target and execution-speed action semantics. Exact parameter equality
is claimed only for the PPO family because standard SAC/TD3 necessarily use
twin critics and different actor parameterizations. Their trainable parameter
counts and actor, critic, temperature, and total optimizer-step counts are
reported explicitly.

## Confirmatory strong-baseline gate

A one-seed bootstrap produces a degenerate zero-width interval and must never
be treated as confirmatory evidence. Strong learned-baseline checks now require
all of the following:

1. At least 10 independently initialized policy-training replicates by default.
   Held-out environment seeds are repeated measurements within a training
   replicate; they increase evaluation precision but do not increase the
   independent sample size.
2. A positive independently clustered bootstrap confidence interval.
3. Rejection by a two-sided paired sign test after Holm correction across the
   complete learned-baseline endpoint family.
4. A valid `trading_metrics_v2` contract for financial endpoints.

`total_return` is the primary task-performance endpoint.
`episode_information_ratio`, rather than an annualized Sharpe ratio over a
short synthetic episode, is the risk-adjusted endpoint. `FocusScore` and
`LowerLFDrift` are responsibility-separation endpoints. These three evidence
classes are aggregated separately: a mechanism improvement cannot rescue a
failed task-performance claim.

Failing the significance gate with a positive point estimate is reported as
`positive_mixed`, not `supported`. All earlier strong learned-baseline artifacts
must be regenerated under this gate before citation in a manuscript main table.
SAC/TD3 implementation coverage is now complete; their comparative performance
claim remains pending a sufficiently powered confirmatory run.

The formal matrix is sharded by `(scenario, policy, training_replicate_seed)`.
Pairing occurs on scenario, training replicate, and held-out environment seed;
bootstrap intervals and sign tests operate on the per-replicate mean paired
delta. Merge-time Cartesian coverage checks block incomplete or duplicated
cells. Artifacts that vary only evaluation seeds under one trained policy are
classified as pseudoreplicated and are ineligible for confirmatory claims.

## Confirmatory training protocol

The earlier learned-policy runner was not suitable for a final experiment: its
PPO actors were linear, the default six iterations provided only 4,320
primitive training steps, every iteration replayed the same three exogenous
market paths, and best-checkpoint selection reused those training paths. Those
artifacts are now ineligible even if they carry a training-replicate label.

The v2 confirmatory runner uses the following role-separated protocol:

1. PPO-family policies use identical two-layer MLP capacity and environment-step
   budgets. Hidden width is frozen at 64. Each policy may select optimizer and
   exploration settings from the same-size preregistered search space; this is
   a stronger baseline protocol than forcing one potentially unfavorable
   setting on every algorithm. Parameter equality and HPO-budget equality are
   audited separately.
2. Every `(training replicate, rollout root, iteration)` deterministically
   derives a fresh exogenous training path. Treatment and baselines receive the
   same paths within a replicate, while different replicates vary both model
   initialization and training data.
3. Fixed validation paths select the checkpoint. Held-out test paths are
   disjoint and are evaluated once after selection. Test outcomes never select
   a model or hyperparameter.
4. The default full budget is 64 iterations, five rollout roots, 240 primitive
   steps per path, or 76,800 training environment steps per policy cell. PPO,
   SAC, and TD3 receive the same primitive training-step budget.
5. Every cell saves its selected checkpoint plus seed roles, optimizer seed,
   parameter count, optimizer-step counts, training/validation/test environment
   steps, and metric-contract version.
6. Merge gates require both
   `fresh_deterministic_path_per_root_and_iteration_v2` and
   `disjoint_validation_paths`, plus the versioned selection utility
   `log_growth_drawdown_utility_v3`. Missing or mixed protocols invalidate the
   headline evidence even when metric rows are otherwise complete.

Checkpoint and hyperparameter selection now use a four-role partition:

1. fresh training paths update model parameters;
2. checkpoint-validation paths select the iteration within one training run;
3. disjoint tuning-validation paths rank hyperparameter candidates; and
4. held-out test paths are unavailable to the HPO executable and are loaded
   only by the final confirmatory runner.

The selection utility is log terminal net wealth minus 0.25 times maximum
drawdown. Net wealth already contains transaction costs, so turnover is not
penalized a second time. Short-episode information ratio remains a reported
risk endpoint but no longer selects checkpoints or hyperparameters. The HPO
artifact records an empty `heldout_test_seeds` field and
`heldout_test_access_status=not_loaded`; merge rejects any cell that violates
this rule. Candidate ranking averages tuning paths within each scenario, then
clusters uncertainty by independent training replicate and ranks by the lower
95% bootstrap bound.

The full stress matrix contains stationary low-noise, stationary high-noise,
localized burst, persistent shift, and OOD-period regimes. With five learned
policies and ten independent training replicates this is 250 training cells.
Scheduler placement is dynamic across the allowed CPU-node pool; no experiment
cell is hard-bound to a host.
