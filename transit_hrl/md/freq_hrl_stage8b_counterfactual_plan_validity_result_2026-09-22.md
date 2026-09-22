# Stage-8B Counterfactual Plan-Validity Result

Date: 2026-09-22

Run: `pointmaze_plan_validity_stage8b_v1_development_20260922_r1`

Tasks: `t100431`-`t100438`

Decision: **`stage9_not_authorized`**

## Integrity

All eight registered optimizer roots completed. Two tasks ran on each of
`node003`, `node004`, `node005`, and `node006`; the scheduler was free to place
them on any of `node001`-`node006`. Only eight compact `result.json` files were
synchronized. No trajectory CSV or checkpoint was pulled.

The audit found 320 unique seeds. Each root used 8 controller-training, 8
checkpoint-selection, 8 branch-fit, and 16 held-out branch-evaluation paths,
with no reuse within or across roots. Each fit path supplied four opportunities
in each of six classes (192 rows per root); each evaluation path supplied the
same balanced coverage (384 rows per root). The 37-feature schema and masks
were identical across roots.

Every paired branch had an exact common prefix: both the maximum prefix-state
difference and candidate-feature difference were `0.0`. `keep` made zero upper
calls at the opportunity, `renew` made exactly one, neither branch made a later
upper call in the common 0.50-second window, and the lower controller remained
closed loop. The extra branch replay budget was recorded separately as
1,964,456 fit and 3,902,056 evaluation primitive steps.

The first formal analysis exposed an analyzer defect: it required different
optimizer roots to reuse identical path seeds, although the frozen protocol
assigned independent seed blocks to roots. The check was corrected to require
equal role sizes, mutually disjoint roles, and no seed reuse across roots.
Regression coverage was added. This changed neither experimental data nor any
registered endpoint.

## Registered Result

The primary endpoint is paired local ISE advantage,
`ISE(keep) - ISE(renew)`. Positive values favor renewing the waypoint now.
Optimizer root is the statistical unit; intervals are two-sided 95% Student-t
intervals over the eight frozen roots.

| Registered check | Root mean [95% CI] | Outcome |
|---|---:|---|
| controller learned, ISE gain vs untrained | 21.075555 [15.576245, 26.574865] | supported |
| renew value at regime +250 ms | 0.087615 [0.075357, 0.099874] | supported |
| regime +250 ms minus regime +10 ms | 0.095215 [0.076190, 0.114241] | supported |
| regime +250 ms minus force pulse +10 ms | 0.046065 [0.030954, 0.061177] | supported |
| regime +250 ms minus distractor change +10 ms | 0.042576 [0.030904, 0.054247] | supported |
| causal-history Spearman rank correlation | 0.224558 [0.143126, 0.305991] | supported |
| causal-history selected local value | 0.084823 [0.069473, 0.100173] | supported |
| causal-history selection minus plan/state selection | 0.004908 [-0.001556, 0.011371] | inconclusive |

Seven of eight registered conditions passed. The conjunction fails because the
causal-history predictor did not establish better selected utility than the
current plan/state predictor. No roots may be appended, and the result does not
authorize Stage 9.

## Interpretation

The experiment establishes a narrower positive result than Stage 8: renewing a
plan has positive local causal value after the consequence of a persistent
regime change develops, and this value can be ranked from causal observations.
It also rejects the shortcut from that fact to a new learned trigger. The full
37-feature history model did not beat a simpler current plan/state model under
the registered utility endpoint.

Post-hoc diagnostics are hypothesis-generating only. History selection beat
plan age alone by 0.023437 [0.007995, 0.038880]. The four-feature change model
had higher rank correlation than the full history model by 0.064128 [0.007163,
0.121093], but neither its utility advantage over plan/state nor over history
excluded zero. These diagnostics cannot replace the failed registered model.

The admissible successor is a fresh predictor-development protocol, not extra
Stage-8B roots and not deployed-trigger evaluation. It should develop a compact
causal plan-validity representation and regularization without inspecting new
qualification paths, then freeze a new independent-root gate before any
closed-loop trigger is trained.

## Claim Boundary

Supported: delayed plan renewal has measurable paired local value, that value
exceeds matched force-pulse and distractor opportunities, and a causal-history
predictor has positive rank and selected utility.

Not supported: causal history improves selected utility over current
plan/state; a budgeted trigger improves closed-loop control; belief is needed;
or the result establishes a frequency-specific or domain-general Freq-HRL
advantage.
