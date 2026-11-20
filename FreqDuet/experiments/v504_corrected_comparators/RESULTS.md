# V504: Corrected Comparator Results

All nine formal tasks (t224816--t224824) finished once. The frozen
aggregator accepted 144 evaluation paths over 16 new matched MBTA-informed
simulation days. Each path used 14 physical buses, launched and completed
262 trips, and had no unserved passengers. All four fresh NoFreq training
runs completed the required 200 episodes and 6,336 upper/12,672 lower
updates. The [receipts](submission_formal.json),
[paired inputs](confirmation/), and [aggregate](confirmation/aggregate.json)
are archived here.

| Aligned minus comparator | Matched NoFreq, mean [95% interval] | Matched Manual, mean [95% interval] |
| --- | ---: | ---: |
| Journey (s/passenger) | -4.562 [-8.753, -0.181] | -4.259 [-7.741, -0.708] |
| Actual arrival CV2 | -0.00240 [-0.00967, +0.00450] | -0.04592 [-0.05792, -0.03429] |
| Readiness delay (s/trip) | +2.320 [-1.357, +12.864] | -96.026 [-122.268, -71.699] |
| Absolute original-timetable deviation (s/trip) | -583.701 [-1006.268, -207.326] | -245.634 [-406.907, -121.954] |
| Frozen joint criterion | Not met | Met |

**Against Manual.** The journey gain is about 0.38%; all four seed means
and 13 of 16 day means favor Aligned. The wait difference is -3.926
s/passenger, while the onboard difference is -0.333 s/passenger with an
interval crossing zero. Both arms had the same budget rule, but mean
commanded holding was 3.460 versus 11.733 s/trip; this was not a
matched-dose study. Manual readiness delay averaged 99.680 s/trip and
exceeded 27 s/trip on every day, versus 3.654 s/trip for Aligned. Less
lateness explains the lower absolute original-timetable deviation; the
earliness difference is +62.123 s/trip with an interval crossing zero.
This establishes a joint win over the specified dynamic rule, not over
all possible timetable-and-holding controllers.

**Against NoFreq.** Journey was lower on 15 of 16 day means and three of
four seed means, but neither CV2 nor readiness met non-harm. The Aligned
seed 50314 on day 53020003 had readiness delay of 150.500 s/trip versus
5.336 for NoFreq; this pair contributes about 98% of the total readiness
difference. It is included in the primary estimate. The absolute
timetable deviation is better than NoFreq, but still averages 515.419
s/trip. This is displacement from the original nominal schedule,
including deliberate retiming, not a field punctuality measure. Both
learned arms retain harmonic inputs in the shared budget ledger.

**Interpretation.** The exploratory NoFreq-minus-Manual comparison has
similar mean journey (+0.303 s/passenger) and a much smaller readiness
delay (-98.346 s/trip), but larger original-timetable deviation
(+338.067 s/trip). The large circulation benefit against Manual cannot
be attributed to harmonic actor inputs. V504 supports a bounded
fixed-fleet learned-control claim on this corridor, plus a modest
journey benefit from the tested harmonic actor inputs. It does not
establish a joint frequency advantage or generalize to other corridors.

Recompute the prespecified results without running traffic:

    python3 study.py aggregate --output confirmation

Evaluation median wall times were 27.3 s (Aligned), 27.9 s (NoFreq),
and 25.5 s (Manual) per complete day. The four training-plus-evaluation
tasks took 1.90--1.96 h each in parallel.
