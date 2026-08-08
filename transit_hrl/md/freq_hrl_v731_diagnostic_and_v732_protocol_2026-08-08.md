# Freq-HRL v7.3.1 Diagnostic and v7.3.2 Protocol

## Decision

The v7.3.1 matrix is development-only diagnostic evidence. It must not create
a frozen configuration and must not enter a confirmatory table. The matrix
repaired the off-policy execution contract and completed 158 valid cells, but
it exposed two blockers: a 32-iteration boundary-limited training budget and
an excessive stationary-low-noise promotion false-positive rate.

Freq-HRL v7.3.2 therefore adds three source-bound gates before held-out access:

1. support-only null-rate calibration for the learned promotion gate;
2. a pre-registered 64/96/128 iteration budget ladder;
3. a fixed confirmatory plan with independent training replicates, held-out
   path seeds, aggregation units, endpoints, and multiplicity correction.

## v7.3.1 execution audit

- run: `full_method_hpo_v731_final_20260808_r1`
- formal task range: `t73399` through `t73608`
- terminal scheduler accounting: 158 done, 52 cancelled, 0 failed
- valid-cell whitelist:
  - full Freq-HRL: 30
  - flat PPO: 30
  - flat GRU PPO: 7
  - generic HRL PPO: 30
  - generic HRL GRU PPO: 1
  - flat SAC: 30
  - flat TD3: 30
- cancelled ranges: `t73459-t73488` and `t73519-t73548`
- cancelled-cell outputs are excluded even when a partially written
  `cell_summary.json` was synchronized.

The six node preflights were submitted as one scheduler wave. Four completed
before formal cells launched; the node003 and node004 preflights overlapped the
first formal cells by seconds. This is a procedural defect in v7.3.1 and is not
described as strict preflight ordering. v7.3.2 requires all six preflight
artifacts to be synchronized and validated before any experiment submission.

## Why 32 iterations cannot be frozen

The selected checkpoint was in the final two iterations for 18/30 full-method
cells and 30/30 generic-HRL cells. Across complete variant families, selected
checkpoint medians were 30 for Freq-HRL, 27 for flat PPO, 31 for generic HRL,
26.5 for SAC, and 17.5 for TD3. The recurrent matrices were cancelled after
the boundary diagnosis and are incomplete.

These observations invalidate 32 iterations as a convergence budget. A clean
cell and positive utility do not override this gate. The v7.3.2 budget plan
fixes five fresh optimizer seeds and nine representative settings across all
seven independently tuned model families. Budgets 64 and 96 are mandatory.
The protocol selects 96 only if every representative has at least 80% trained
checkpoints, positive mean validation learning gain, and no more than 40% of
replicates selecting within the last 12.5% of iterations. Otherwise 128 is
required and is selected only under the same all-family gate. Failure at 128
blocks final HPO and triggers algorithm redesign.

## Promotion diagnosis and repair

At v7.3.1, every full-method candidate failed the 0.8 selective-replicate
gate. Stress-versus-low-noise advantage and rate lifts were generally
positive, but stationary-low-noise action rates remained roughly 0.38 to 0.60
for failed replicates. The best observed selective counts were 2/5 for
`v73_balanced_strict`, 2/5 for `v73_forecast_margin`, and 1/5 for
`v73_balanced_margin`.

v7.3.2 keeps median prediction-bias correction and then raises the decision
threshold to the support-only stationary-low-noise empirical quantile needed
to enforce the registered action-rate cap. Calibration uses only the dedicated
promotion-calibration seeds. It cannot inspect tuning utility, OOD,
`promotion_recovery`, or held-out confirmatory paths, and it cannot mutate the
checkpoint.

## Source-bound confirmatory contract

The confirmatory plan is committed in source before final HPO. It fixes:

- 24 independent training replicates;
- 8 held-out path seeds used as repeated measures within each replicate;
- 6 scenarios, including persistent shift, recovery, stationary noise,
  localized burst, and OOD;
- 6 capacity-matched learned baselines and 5 one-factor mechanism ablations;
- pooled primary endpoints `total_return` and `LowerLFDriftAbs`;
- equal scenario weighting, paired training-replicate inference, bootstrap
  intervals, randomization tests, and Holm correction.

Final HPO must consume a valid budget-decision JSON whose SHA-256, plan hash,
selected iteration count, commit, and Freq-HRL source manifest match every HPO
cell. The frozen HPO output is rejected if any selected family remains
boundary-limited, fails to learn, or if the full method fails its mechanism
gate. Confirmatory tasks are not launchable without that frozen output.

## jtl110cpu accounting boundary

The legacy v6 queue currently retains 81 `running` records assigned to
`jtl110cpu`. Their wall time is not credible algorithm runtime. Direct SSH
fails during key exchange, log-tail probes fail, no progress marker is
available, and an earlier force-cancel could not confirm remote process-tree
termination. Scheduler therefore preserves the records rather than falsely
marking them done or cancelled.

These records are excluded from all v7.3.2 runtime and scientific evidence.
No new Freq-HRL task may target `jtl110cpu` or `jtl110cpu2`. New tasks use
dynamic placement over `node001` through `node006`, one declared physical core
per independent cell, with no hard node binding and no login-node environment
installation.
