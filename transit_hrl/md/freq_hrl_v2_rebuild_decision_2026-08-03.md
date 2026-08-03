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
