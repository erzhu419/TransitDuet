# Freq-HRL v7.3 Pilot Audit and v7.3.1 Final Repair

Date: 2026-08-08

## Decision

The v7.3 calibrated-advantage pilot passes the pre-registered selective
promotion advancement gate. Two candidates advance to independent final HPO.
The first attempted final matrix is invalid because a baseline-only metadata
initialization bug caused non-frequency policies to read Freq-HRL-specific
promotion fields. No result from that matrix is eligible for model selection,
freezing, confirmation, or manuscript reporting.

## Audited v7.3 pilot

Run: `full_method_hpo_v73_calibrated_pilot_20260808_r1`

- Source revision: `506d40fc82742b569877036c6bdecaffe7605ed0`.
- Design: six Freq-HRL candidates by three independent optimizer replicates.
- Support scenarios: stationary low noise, stationary high noise, localized
  burst, and persistent shift.
- Result: 18/18 cells completed and the merge status was
  `selective_candidate_found`.
- All six candidates passed the learning gate. Two candidates passed the
  mechanism gate in all three independent replicates.

| Candidate | Selective replicates | Utility mean | Utility 95% CI | Low-noise rate | Stress rate | Stress-low rate lift | Advantage lift | Calibration accuracy before/after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v73_forecast_margin` | 3/3 | 0.01419 | [0.01245, 0.01596] | 0.1753 | 0.5196 | 0.3443 | 0.02553 | 0.4850 / 0.5804 |
| `v73_balanced_strict` | 3/3 | 0.01325 | [0.01071, 0.01603] | 0.0686 | 0.3368 | 0.2682 | 0.02462 | 0.4579 / 0.6510 |

This is advancement evidence, not confirmatory performance evidence. The
pilot remains diagnostic and does not create a frozen configuration.

## Invalid first final attempt

Run: `full_method_hpo_v73_final_20260808_r1`

The six-node environment preflight passed. The 210-cell matrix was then
submitted with the registered final optimizer seeds. The off-policy path
failed after training at `run_hpo_cell` while constructing not-applicable
promotion metadata:

```text
KeyError: 'promotion_advantage_target_threshold'
```

The key exists only for frequency-policy candidates. It was read before the
variant guard, so SAC/TD3 and later non-frequency baselines could not complete
the common cell contract. The run was stopped after structural diagnosis. The
original 210 cells ended as 86 done, 54 failed, and 70 cancelled. Including six
successful preflights and 54 invalid automatic retries, the scheduler family
ended as 92 done, 54 failed, and 124 cancelled. Remote cancellation reported
zero kill failures.

No cell summary from this run was synchronized or inspected before the repair.
The failure was diagnosed only from scheduler state and tracebacks. Therefore
the registered final optimizer seeds remain frozen; no outcome-adaptive seed or
hyperparameter change was made. Nevertheless, the entire run is excluded
because a final matrix may not mix source revisions or incomplete policy
families.

## v7.3.1 repair

The repair changes only common result metadata for policies to which promotion
calibration does not apply:

- baseline target and decision thresholds default to `0.0`;
- Freq-HRL candidates retain their explicit target and calibrated decision
  thresholds;
- baseline calibration status remains `not_applicable` with zero samples;
- the tuning protocol, HPO implementation, scheduler signature, and project
  labels are versioned as v7.3.1.

The model architecture, optimization budget, environments, candidate grid,
registered seed roles, selection objective, and advancement thresholds are
unchanged.

## Verification and restart rule

- Targeted verification: 15 tests and 7 subtests passed, including complete
  one-cell SAC and TD3 paths.
- Full verification: 385 tests and 65 subtests passed.
- A new final run name and one committed source revision are required.
- All 210 cells must be regenerated under that revision.
- Merge must reject incomplete policy families, mixed source identities, pilot
  optimizer reuse, or any access to OOD and confirmatory seeds.
- Only a `frozen_from_support_validation_only` merge may advance to held-out
  confirmation. Pilot and invalid-final outputs remain non-reportable.
