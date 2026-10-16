# V73: causal fleet-state sufficiency, consumed-day development

Freeze V59 actors, upper/B200 deployment, dual, production values and main.
Recapture all 48 V70 native references / 144 pre-action prefixes. Require exact
original prefix (including RNG), full-day summary, lower holding and upper
paths before reusing unchanged seven-action / 32-future labels. Smoke references
0 and 36 cover both original training seeds and all three windows; reuse them.

Add every current physical bus slot: AVL position, direction, speed, operating
state, APC load count, current trip/actual launch, completed return and remaining
executed holding/service. Add every published logical trip: planned eligibility,
release/headway/min-gap, mode, query status, completed eligibility/actual launch.
Use fixed full arrays, no truncation or pooling. Exclude OD, future arrivals,
latent day profiles, RNG, future actual launches/returns and future actor actions.
All timestamps are historical/current or published commitments, not forecasts.

Keep V72's original state, quadratic action contrast, day-balanced ridge 1,
five outputs and native-anchored constrained selection. Registered families:
original+fleet, original+plan, original+fleet+plan (sole primary). Reuse cached
V72 zero/action-only/original-state controls after exact first-fold coefficient
and scoring reproduction. Use the algebraically identical dual ridge when
columns exceed fitting rows; test against the primal objective. Report input,
coefficient and effective dimensions; this is not a capacity-isolated comparison.

Exclude each complete demand day from fitting and scaling, all four actors and
three windows together. Primary prediction gate: MAE and MSE paired day-CI uppers
<0 for passenger exposure, CV-squared and readiness versus all three controls.
Joint gate: passenger upper <0 and all four constraint uppers <=0. Report all
families, effects, choices, day700006 and its evaluation-only removal. No posthoc
ablation winner or relaxed gates. Both gates precede any fresh-day confirmation.

All tests, 48 native replays, 12-fold fits and aggregation run via scheduler on
node001-node006. Arrays/raw snapshots/weights remain server-side; pull only small
markers/summary JSON. No actor update or policy switch. This diagnoses missing
current resource information, not fixed-headway superiority or joint deployment.

## Outcome / Next

Source `6be3d39249`; scheduler `t165760` smoke, `t165766` 48-worker references
on node005, `t165767` 12-worker folds and `t165768` aggregate on node004.
All completed without retries: 19 tests, exact V72 first-fold reproduction,
48 exact native days / 144 prefixes. Only 46,252 bytes of JSON were collected.
Passenger MAE: zero 1796.49, original 1997.77, fleet 2187.64, plan 2116.60,
fleet+plan 2262.90 person-seconds. Primary-minus-original +265.12
[145.96,383.25]; CV-squared and readiness MAE also worsen. Primary changes
26/144 actions: eight passenger improvements, 16 worsening, two ties, three
jointly nonworse points. Exposure +12.66 [-66.60,98.77] person-seconds;
actual holding +4.32 [2.23,6.30] seconds/day. Both registered gates fail.
No fresh confirmation, production update or policy switch.

V74 completed the fixed-fit error/support diagnosis; see
`protocol_v6_v74_fit_gap.md` for the next structural-feature/data-budget step.
Do not tune ridge/gates on V70.

Limitations: consumed 12-day development data, two original training seeds and
no capacity isolation. The primary has 4127 inputs / 41280 coefficients and
effective degrees of freedom 40.1-40.9 per output, versus V72's 11.7-11.9.
This failure does not establish that current fleet/plan information is irrelevant.
