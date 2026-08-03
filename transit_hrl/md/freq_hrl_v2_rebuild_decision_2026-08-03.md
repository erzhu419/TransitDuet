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
