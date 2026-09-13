# MuJoCo v24 Policy-Mean Upper-Target Result

## Decision

V24 stops. All 48 cells completed, yielding 1,920 evaluation episodes. The
candidate won only 7/12 environment-by-optimizer-root comparisons and failed
the frozen reward, upper-MSE, and correction gates. The original algorithm
revision is `8e3b571185a84d0adb00307421a89c0f38a81412`; results remain attached
to that revision after the numerical repair below.

## Performance

Returns average four optimizer roots, each evaluated on eight shared roots
crossed with five disturbance conditions. Causal denotes the matched v23
first-sample control. Correction changes below are relative to that control.

| Environment | Zero return | Causal return | Hindsight return | V24 return | V24 vs causal | Wins | Component / total correction change |
|---|---:|---:|---:|---:|---:|---:|---:|
| HalfCheetah | 1995.83 | 2199.15 | 2102.63 | 2005.82 | -8.79% | 2/4 | -10.17% / -11.48% |
| Hopper | 176.53 | 161.36 | 188.54 | 200.45 | +24.22% | 4/4 | -10.53% / -8.47% |
| Walker2d | 164.48 | 249.83 | 256.89 | 199.74 | -20.05% | 1/4 | +38.50% / +34.25% |

Hopper passes its total-correction cap at 0.24064 (cap 0.25). HalfCheetah
trades return for lower correction. Walker2d loses on both endpoints.

| V24 return change vs causal | Standard | Low frequency | High frequency | Mixed | OOD chirp |
|---|---:|---:|---:|---:|---:|
| HalfCheetah | -7.16% | -12.81% | -12.31% | -15.00% | +3.97% |
| Hopper | +21.33% | +25.38% | +23.30% | +25.68% | +25.93% |
| Walker2d | -24.65% | -18.42% | -20.88% | -18.98% | -17.37% |

Hopper's largest root contributes 83.50% of its absolute gain; removing it
leaves +5.38%, and the median paired-root gain is +6.70%. HalfCheetah's
-34.52% root drives its aggregate loss; omitting that root changes the mean
contrast to +1.94%. Walker2d remains negative when any one root is omitted
(-28.77% to -13.12%), and both correction metrics worsen in every condition.

## Mechanism Diagnosis

Upper consistency MSE changes from 0.16695 to 0.15173 in HalfCheetah (-9.11%),
0.64997 to 0.69935 in Hopper (+7.60%), and 0.53607 to 0.73596 in Walker2d
(+37.29%). These are prediction errors averaged over different training
trajectories, not direct measurements of conditional target variance.

The code still projects a sampled upper action while replacing only the lower
sample by its Gaussian mean. Projection and tanh are nonlinear, so this plug-in
target need not equal the expected projected target. The remaining upper
sampling noise and inverse-tanh amplification are hypotheses for a targeted
mechanism study, not established causes of the observed regression.

Walker2d's forward reward falls from +5.33 to -40.06 while mean episode length
changes only from 246.36 to 241.66. The return loss therefore primarily reflects
lost forward motion. All 160 candidate Hopper episodes and all 160 candidate
Walker2d episodes terminate before the 1,000-step horizon; candidate mean
lengths are 93.71 and 241.66, respectively.

## Numerical And Selection Audit

All arms pass certificate, realized prefix-budget, and frozen fallback gates.
The maximum cell-mean fallback rate is 0.005115. Minimum candidate Dykstra
convergence is 0.94671, recorded as the preregistered numerical diagnostic.

All 480 candidate evaluation rows fail the frozen `1e-12` target equality
check, with maximum discrepancy 2.9021e-6. Preview used float64 projected
components; the executed target used float32 components before inverse tanh.
A nonzero-action regression reproduces this discrepancy. Matching component
precision fixes the test and preserves its deterministic rollout return and
correction. The analyzer now records the failed audit and finishes the report;
the original threshold and rejection remain intact.

Checkpoint selection uses mean per-step reward; the gate uses episode return.
A server-side reduction of existing histories finds different maximizing
checkpoints in only 4/48 cells. All four Walker2d candidates have identical
rankings; the only affected candidate is Hopper root 258618141, with a 0.103%
selection-return gap. This mismatch merits prospective correction but does not
explain v24's main failure. Only a 30,045-byte summary was retrieved.

## Limitations And Next Step

Four optimizer roots support development diagnosis, not confirmatory CI or
superiority. Evaluation paths and conditions are repeated measurements. The
matrix compares projected HRL variants and does not establish an advantage
over unconstrained or non-frequency baselines. Retire v24 roots and this
parameterization; the precision repair does not rescue its reward gates.

Next, measure same-state target bias and variance separately for upper and
lower sampling, and compare raw-space error with executed-action error. A new
mechanism should address Walker2d forward motion and Hopper early termination
before another large campaign. Any subsequent selection rule and experiment
must be frozen on fresh roots.

## Reproduction

Run `scripts/analyze_mujoco_v24_policy_mean_upper_projection_target_development.py`
and `scripts/diagnose_mujoco_v24_policy_mean_upper_projection_target_development.py`
with `--run-name mujoco_v24_policy_mean_upper_projection_target_development_20260912_r1`.
They write `analysis.json`, `README.md`, and `diagnostics.json` under that run's
`analysis/` directory. Run `scripts/audit_mujoco_checkpoint_objective.py
--run-dir <server-full-run-directory>` on the server to reproduce the recorded
selection-rank diagnostic. Checkpoints and full histories remain server-only.
