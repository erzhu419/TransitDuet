# V407 Development Result: Do Not Promote

All sixteen traffic paths completed once. Server readout t214771 verified the
physical accounts and paired 5,200-event streams; no failed/repeated traffic,
training, confirmation expansion or main-policy change. Only compact JSON was
retrieved. Both regimes fail the original development screen.

Candidate minus comparator, two paired demand days per regime; negative is lower.

| Comparator | Regime | Journey s/person | Actual CV2 | Readiness s/trip | Holding s/trip |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed10 | Nominal | +1.889 | -0.012873 | +4.191 | +20.601 |
| Fixed10 | Heavy | -3.678 | -0.014955 | +1.074 | +7.662 |
| Frozen V406 single | Nominal | +2.007 | -0.001283 | -0.731 | +2.004 |
| Frozen V406 single | Heavy | +4.348 | -0.005774 | +2.061 | +0.038 |

Against fixed10, actual CV2 improves on all four days, but readiness and public
delay worsen on all four. Journey improves on only two. Public delay increases
1,098 s/day nominal and 281.5 heavy. This is not solely a confidence-bound failure.
Against frozen V406, journey worsens on all four days. Heavy-day holding differences
are only +0.630/-0.553 s/trip, while waiting rises 2.009/6.698 s/person and onboard
time barely changes. Aggregate holding alone does not describe the changed service.

Fixed15 is a stronger-regularity operating point, not a rescue comparator: V407
has lower mean journey/readiness than fixed15 but worse CV2 on every day. One
nominal day also has worse journey. Neither holding-dose equivalence nor a better
tradeoff frontier is established by these two simple operating points.

The corrected model is active: 800/836 nominal and 686/583 heavy extra requests
relative to fixed10. Reference-incomplete queries fall from V406's 341--354/day
to zero. Pending and unfinished obligations are recorded, with maximum passenger
balance error below 1.1e-10. More complete finite-window accounting did not produce
better all-day control. It does not by itself establish better forecast accuracy.

Retain the frozen V406 positive signal: both new heavy days improve journey,
CV2 and readiness versus fixed10 (means -8.026 s/person, -0.009182, -0.987 s/trip).
Its nominal journey mean is only -0.118 s/person, with one worsening day and
readiness worse on both. Do not generalize the heavy result to the whole method.

V407 wall time is 42--85 min/day versus 28--56 for V406 and 64--77 s for the simple
rules. Extra computation did not buy a journey advantage. Keep the corrected
kernel as an engineering component; close this control candidate without tuning,
confirmation or promotion. Original code, historical outcomes and paper unchanged.

## Limitations

Four development days, not sixteen independent samples or a confirmation study.
The repair combines continuous service, known clocks and boundary-state guards;
its components' individual causal effects are not isolated. No learned/frequency
or APC-cost attribution test. Unboarded counts range from 0 to 1,506 across arms;
their waiting remains charged to the fixed evaluation endpoint. There is no
one-second runtime guarantee. Evidence: review/readout/results.json.
