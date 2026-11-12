# V410: State-Dependent Holding Beats the Matched Within-Day Rule

All 54 paths completed without retries: six calibration paths and 48
confirmation paths on 16 new heavy-demand days. All three arms passed
physical accounting, no-early-departure, and the paired 5,200-event and
generated-population checks. The old development days selected the
unmodified transition table (tilt zero): +0.076 s/trip holding difference
and at most 0.54 percentage points of action-share difference. No outcome
was used for selection. Server aggregate t214916 completed on node004.

Capped V406 minus the frozen within-day rule; 95% paired day-bootstrap
intervals. Negative values are lower.

| Endpoint | Difference [95% interval] |
| --- | ---: |
| Holding, s/trip | -0.254 [-0.743, +0.221] |
| 0 s bin, percentage points | +0.067 [-0.154, +0.320] |
| 5 s bin, percentage points | +0.121 [-0.147, +0.389] |
| 10 s bin, percentage points | -0.189 [-0.499, +0.131] |
| Journey, s/person | -4.568 [-5.096, -3.989] |
| Waiting, s/person | -3.409 [-4.014, -2.754] |
| Onboard, s/person | -1.159 [-1.396, -0.945] |
| Actual CV2 | -0.009567 [-0.011264, -0.008084] |
| Readiness, s/trip | -3.224 [-3.680, -2.750] |
| Public delay, s/day | -844.6 [-964.1, -720.6] |

The holding and all three bin-share intervals are inside the frozen margins
(+/-3 s/trip and +/-2 percentage points); all four outcome gates pass.
Journey, actual CV2 and readiness improve on all 16 days. Readiness and
public delay measure the same delay burden at different scales here.
Restricted journey P95 improves by 15.86 s [-30.17, -2.71], while the
unboarded-count interval crosses zero.

Against pure fixed10, capped V406 saves 5.795 s/person and 5.125 s/trip of
readiness delay, with mean CV2 -0.001421 (10/16 days improve); it also uses
4.865 s/trip less holding, so this is not a matched-dose contrast. The
random transition rule itself saves 1.227 s/person versus fixed10, but
CV2 worsens on every day. The capped controller preserves the service gain
without this loss of regularity.

**Interpretation.** Under the shared 0/5/10 menu, matching mean commanded
holding and the marginal bin shares does not reproduce the model
controller's outcomes. This supports the value of the state-dependent
selection policy as a package in the tested heavy-demand setting, not an
isolated effect of any feature or an exact match of holding across trips,
stops or times. The reference still uses the native bilateral request;
it lacks the model controller's additional state. The study does not
establish a frequency, timetable or learned-supervisor benefit.

**Decision.** Retain as supplemental confirmation, no main-policy switch.
The next useful question is transfer beyond the single corridor and 1.2x
regime, preserving the frozen comparison rather than tuning on these days.
The model ran 16--34 min per day versus about 1--1.5 min for the simple
controls; its daily callback p95 of 0.234--0.838 s is not a deadline guarantee.

[Frozen protocol](PROTOCOL.md) | [Calibration](review/calibration/results.json) |
[Small aggregate](review/aggregate/results.json)
