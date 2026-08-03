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

The PPO-family comparison uses two explicit algorithm contracts. `freq_hrl`,
`generic_hrl_ppo`, and `generic_hrl_gru_ppo` use the same asynchronous SMDP
trainer, while `flat_ppo` and `flat_gru_ppo` are canonical primitive-rate PPO
policies with one joint
target/execution action, one joint probability ratio, and one value critic. Its
hidden width is chosen to keep its active parameter count within 5% of the
Freq-HRL reference core; every generic or recurrent control is matched by the
same rule after its encoder is fixed, and inactive padding parameters are
forbidden. Epoch,
environment-step, HPO-search, train/validation/test seed, and metric budgets are
matched. Freq-HRL alone receives frequency-responsibility observations and
promotion. The MLP controls receive every observation in a contiguous 120-bar
causal raw window. The GRU controls receive every observation from episode start
through the current decision, with no bidirectional or future context. The
default comparison disables the handcrafted Freq-HRL actor prior; prior-enabled
results belong in a separate ablation.

## Complete off-policy learned baselines

The strong learned-baseline matrix now runs local, auditable implementations
of SAC and TD3 rather than registering them as missing external dependencies.
Both are single-level flat policies with a joint target-weight/execution-speed
action, persistent replay, twin Q critics, soft target updates, and deterministic
held-out evaluation. SAC includes a squashed Gaussian actor and learned entropy
temperature; TD3 includes target-policy smoothing and delayed actor updates.

All seven policies use the same market paths, transaction costs, held-out seeds,
primitive environment-step budget, and `trading_metrics_v2`. Generic HRL, flat
PPO, SAC, and TD3 receive a complete contiguous causal raw history plus position,
active or previous target, target gap, and episode progress. The MLP/off-policy
controls use the registered 120-bar horizon; the two GRU controls use the full
causal episode history. Flat PPO, flat GRU PPO, SAC, and TD3 share the bounded
target and
execution-speed action semantics. Near-equal active capacity is claimed only
inside the PPO family because standard SAC/TD3 necessarily use twin critics and
different actor parameterizations. All trainable parameter counts and actor,
critic, temperature, and total optimizer-step counts are reported explicitly.

The recurrent controls use independent single-layer unidirectional GRU encoders
inside each actor and critic. Raw samples are reshaped oldest-to-newest; static
position, plan, gap, coverage, and progress context is projected separately and
fused only after the GRU. They are trained by the same PPO ratio, GAE, optimizer,
checkpoint-selection, and seed protocols as their MLP counterparts. Closed-form
GRU parameter counts are regression-tested against the instantiated trainable
parameters, and encoder width is selected without dummy parameters to remain
within 5% of the Freq-HRL reference capacity.

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

1. PPO-family policies use capacity-matched MLP or causal-GRU encoders and
   identical environment-step budgets. The Freq-HRL reference hidden width is
   frozen at 64; raw-input baselines use the analytically closest active width
   within 5% of that reference budget. Each policy may select optimizer and
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

Two additional learning-validity controls are mandatory. First, every trainer
records the initial validation score, selected checkpoint iteration, and
validation gain. A random initialization selected at iteration `-1` is not
silently counted as a learned policy. A policy passes the matrix-level learning
gate only when at least 80% of its independent cells select a post-update
checkpoint and its mean validation gain is positive. Second, trading rewards
are multiplied by 100 only inside gradient updates to condition critic targets;
all environment returns, costs, drawdowns, and reported endpoints remain in
their original units. PPO actor/value networks use zero-bias orthogonal
initialization with small output gains so initial critic values do not dominate
the approximately milliscale financial rewards.

The learned-baseline contract was also rebuilt after a representation audit.
The old `flat_ppo` path used two independently updated SMDP actors and therefore
was not a canonical flat PPO. It has been replaced by one primitive-rate policy
over a joint target/execution action, one joint Gaussian probability ratio, one
task-return GAE, and one state-value critic. Its hidden width is selected
analytically to match the active Freq-HRL parameter count within 5%; no unused
padding parameters are counted. The MLP controls receive all 120 direct causal
observations in oldest-to-newest order, so they no longer subsample the
preregistered comparison horizon. This closes the v3 sparse-lag defect. The new
flat and hierarchical GRU controls additionally consume the complete causal
episode, closing the longer-memory objection against comparison with a stateful
EMA. Any representation-efficiency claim must beat both MLP and GRU controls.

The full stress matrix contains stationary low-noise, stationary high-noise,
localized burst, persistent shift, and OOD-period regimes. With seven learned
policies and ten independent training replicates this is 350 training cells.
Scheduler placement is dynamic across the allowed CPU-node pool; no experiment
cell is hard-bound to a host.

Confirmatory execution is mechanically coupled to the nested-validation
freeze. A final freeze is valid only if all seven policies cover the five
preregistered stress regimes, use at least five independent training
replicates, compare at least two candidates per policy, pass the learning gate,
and never load held-out test seeds. The freeze also records
`learned_baselines_v5_causal_gru_controls_2026_08_03`; a stale implementation
version or changed candidate parameters invalidates it. The scheduleurm formal
submitter requires this JSON, injects selected parameters separately for each
policy, and records a canonical JSON SHA-256 in every row and checkpoint.
The freeze now also contains the full 40-character Git revision and a
deterministic SHA-256 over every registered Python/configuration path and byte
under `freq_hrl`. Before submission, the local package must match both the
recorded Git tree and the frozen manifest. Every remote cell recomputes the
manifest from its staged package and fails before training on any mismatch;
merging rejects mixed or missing source identities. This makes code staging,
not only hyperparameter selection, part of the confirmatory evidence contract.
Exploratory smoke runs remain available, but their headline evidence status is
forced to `exploratory_unfrozen_hyperparameters`.

## Confirmatory analysis hardening

The pre-final audit found that the original paired-check default omitted
`flat_gru_ppo` and `generic_hrl_gru_ppo`, even though both policies were part of
the registered seven-policy matrix. This made a complete matrix incapable of
closing its all-baseline claim. The v3 analysis contract now includes all six
controls by construction and regression-tests the control family.

The final analysis reports both an equal-weight pooled effect over the five
registered stress regimes and separate stress-stratum effects. Held-out paths
remain repeated measurements within an independently trained policy replicate.
Multiplicity is controlled by Holm correction within the registered pooled or
stress-stratum endpoint family.

Statistical significance alone is insufficient. Before held-out evaluation,
the following smallest effects of practical interest were fixed from the
nested-validation scale: `0.005` absolute episode return, `0.25` episode
information ratio, `0.02` FocusScore, and `0.05` LowerLFDrift reduction. A
positive confidence interval that does not clear the corresponding threshold
is reported as statistically supported but practically too small, not as a
supported headline result.

Merge-time coverage now checks the explicitly registered
`scenario x policy x training-replicate x held-out-seed` Cartesian product.
Inferring the expected grid from observed rows is not permitted for a formal
merge because it cannot detect a seed or policy missing everywhere. The merge
also reloads the validated frozen configuration and verifies every row's
candidate ID, canonical parameter JSON hash, code revision, and source manifest.

The 504-cell pilot launched from revision
`8d228c2fac012acb41cad7aa9b3a8ec035b1e687` remains a tuning-scale diagnostic.
Because this audit changes the registered source manifest, it will not be used
as the final freeze. The final HPO will rerun the complete equal-budget candidate
space on the hardened revision across all five stress regimes before any
held-out test seed is loaded.

## Bottom-up algorithm audit after the hardened HPO launch

The 1,400-cell HPO launched from revision
`11512773e315757cddcb408373ec32b0832da579` is a valid, immutable evaluation of
the routing and asynchronous-SMDP core. It is not a validation of the complete
method described in `freq_hrl_dev_manual.md`. The HPO entry point deliberately
uses direct upper targets and disables the plan curve, leakage penalty,
primal-dual leakage update, raw-drift recentering, and handcrafted actor prior.
Its results therefore remain useful as a strong learned core comparison and as
an ablation, but they cannot be labelled "full Freq-HRL" in a main claim.

The static audit identified three algorithmic gaps that must be closed before a
new full-method freeze.

1. The trading lower reward currently contains transaction cost, target-tracking
   error, and leakage cost, but no marginal task term whose value depends on the
   routed high-frequency observation. With constant transaction-cost mechanics,
   the lower policy can learn a generic speed-versus-tracking compromise without
   using HF information. This is insufficient evidence for an HF lower
   controller. The rebuilt credit contract must expose an additive, auditable
   lower contribution such as the realized return of the execution deviation
   from the active plan, plus volume-dependent impact, short-horizon inventory
   risk, and leakage. Upper and lower credits must reconstruct the observed task
   reward within numerical tolerance.
2. `LearnedPlanActionMapper` currently evaluates Bernstein coefficients at one
   fixed offset and returns a single target. The rollout then holds that target
   until the next upper decision. This is a coefficient-parameterized target,
   not an executed plan curve. The full method must retain the active
   coefficients and causal origin, evaluate the curve at every primitive step,
   rebase continuously on promotion, and charge smoothness in the upper reward.
3. A projected lower effect may be used for diagnostics, but it must not replace
   the raw physical action effect in the constraint. Subtracting a rolling
   baseline before computing lower LF drift can suppress the measured violation
   without changing the actual inventory or timetable drift. The full method
   must train its cost critic and dual variable on raw action-effect leakage;
   projected metrics remain explicitly labelled sensitivity diagnostics.

The audit also found a one-step timing ambiguity in trading credit assignment.
The environment marks return on the pre-trade position while a newly selected
upper target begins accumulating reward in the same macro interval. This mixes
the previous plan's first-bar return into the new upper transition and is most
material for short promotion intervals. The rebuilt environment contract must
declare the observation, execution, and mark-to-market order and test macro
credit against an exact hand-computed trajectory.

### Required v3 gates

1. A lower-action intervention test must show that changing only causal HF input
   changes the learned lower action while LF state and plan are fixed.
2. A reward-conservation test must verify that upper credit, lower credit, and
   separately reported penalties reconstruct every primitive task reward.
3. A plan-execution test must verify multiple distinct values along one learned
   curve and continuity across scheduled and promotion-triggered replans.
4. A raw-effect leakage test must fail when physical LF drift is large even if a
   projected diagnostic is small.
5. A timeline test must verify that no return or demand outcome observed after an
   action enters that action's state, and that every outcome is credited to the
   action that could causally affect it.
6. Full-method HPO and held-out confirmation must be separate from the current
   routing-core HPO. Both artifacts remain reportable, with immutable source and
   protocol identities.

Until these gates pass, the project stays in algorithm-rebuild mode. Manuscript
closeout would preserve an attractive implementation but leave the central
frequency-responsibility claim underidentified.
