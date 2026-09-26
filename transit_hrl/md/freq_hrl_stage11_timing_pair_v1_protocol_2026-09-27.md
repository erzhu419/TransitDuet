# Stage-11 Same-Budget Timing-Pair Development Protocol

Date: 2026-09-27

Protocol: `pointmaze_timing_pair_stage11_v1_development`

Algorithm revision: `5c2ab7af7201b8b7f98f77c72bd50dd88472ce01`

Question: Does action-aligned counterfactual supervision improve the
fixed-budget causal trigger? The old Stage-9 branch label compares an extra
renewal call with no call at mostly off-grid event times. This protocol
compares two schedules that each make exactly one upper call per 50-step bin:
at a registered check offset 0, 5, 10, 15, or 20 versus the forced offset 25.
Both arms have the same prefix and exogenous path. All other bins use offset
25, except the first bin at step 0. The target is paired full-episode
`ISE(wait-to-25) - ISE(plan-now)`.

Controller training, causal feature construction, Ridge predictor family,
grouped path cross-validation, 0.75 out-of-fold threshold quantile, and
closed-loop inference match Stage-9. Each of eight branch-fit paths contributes
12 uniformly selected noninitial bins, with offsets cycled across the five
legal checks. The 96 paired full-episode rollouts require 230,400 primitive
steps per root, versus 230,400 for Stage-10 trigger training/selection and
244,646-245,468 for the old Stage-9 branch fitting on these roots. A separate
single fixed-schedule replay validates the new evaluator and is not training.

Preflight uses root `208001`, 300-step episodes, and two pairs per each of two
branch-fit paths. Development uses already-revealed roots `209011` and
`209061`; 16 held-out trigger-evaluation paths remain disjoint from fitting.
Read out paired-prefix agreement, call budgets, label distribution, candidate
ISE/return, fixed and old Stage-9 candidate outcomes. Advance to fresh-root
confirmation only if the aligned candidate beats both fixed and the old
Stage-9 candidate in ISE on **both** development roots. Otherwise keep the
negative result and investigate state-coverage or sequential credit. No
paper claim follows from these reused roots.

Scheduler placement is dynamic across node001-node006 at one CPU and 1536 MB
per cell. Only compact `result.json` is synchronized.
