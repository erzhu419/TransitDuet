# MuJoCo v18.1 Causal Actor-State Dataset Outcome

## Decision

Status: `causal_state_dataset_validated_on_reused_paths`.

V18.1 replayed the frozen v17.4 checkpoint on the unchanged 120 development
paths and exported each observation and lower-policy state immediately before
the corresponding action and transition. It did not read v17.12 actor-floor
labels or target actions during export.

## Validation

All 120 scheduled exports completed on node003. The downstream v18.2 loader
validated 40 paths per environment, including all 113 reference-feasible and
seven actor-floor paths. The resulting trajectory totals were 40,000 steps for
HalfCheetah-v5, 3,322 for Hopper-v5, and 6,549 for Walker2d-v5. State, action,
and reference traces aligned within the frozen tolerance.

The 120 NPZ traces remain server-only. This repository retains only the compact
validation facts required to reproduce the development decision.

## Claim Boundary

This record establishes a valid causal state dataset for reused-path model
development. It does not establish model quality, reward improvement, online
learning, fresh-seed generalization, or manuscript performance support.
