# V505: Holding, rolling control and the circulation failure

The formal budget was 208 paths, run as 26 scheduler tasks t225021--t225046.
All used common exogenous demand, 14 vehicles and the complete 262-trip
roster; no passenger was unserved. All 64 passive aligned paths exactly
reproduced the V504 endpoints. The two selected anomaly replays also
reproduced their V504 endpoints. The two initial trace serialization
failures and the two explicitly permitted replacements remain in the
submission receipts. No controller was retrained.

| Aligned minus comparator | Frozen upper, zero holding | Frozen upper, rule holding | Rolling |
| --- | ---: | ---: | ---: |
| Journey (s/passenger) | +0.299 [-0.878,+1.751] | -0.529 [-1.809,+0.174] | -19.110 [-22.309,-15.842] |
| Waiting (s/passenger) | -0.309 [-1.246,+0.766] | -0.010 [-1.195,+0.543] | -13.222 [-15.906,-10.551] |
| Onboard (s/passenger) | +0.608 [+0.114,+1.137] | -0.519 [-0.968,-0.097] | -5.888 [-6.857,-4.860] |
| Actual arrival CV2 | -0.00256 [-0.00757,+0.00249] | +0.00060 [-0.00379,+0.00271] | -0.02348 [-0.03079,-0.01432] |
| Readiness (s/trip) | -1.256 [-16.202,+9.357] | -4.573 [-27.599,+0.320] | +1.842 [-1.697,+13.336] |
| Absolute dispatch offset (s/trip) | -7.619 [-51.715,+22.810] | -14.001 [-68.215,+22.274] | -1753.773 [-1915.697,-1634.895] |
| Prespecified joint criterion | Not met | Not met | Not met |

The corrected V504 whole-controller win over Manual stands. Holding alone
does not have an established journey benefit once upper weights are fixed.
The zero-hold arm is the parsimonious alternative on these days; this is
not a superiority or equivalence finding. Aligned holds less than the rule
with the same upper actor (3.460 vs. 7.167 s/trip) and reduces onboard time,
but not journey with a confidence bound below zero.

On the same states, the budget removed at least one bin at 10,540 of
382,720 learned decisions (2.75%) and blocked the highest-logit choice at
72 (0.019%). For the rule, the corresponding counts were 17,430 and 967
(4.55%, 0.253%). Zero holding has no blocked preferred action. These
diagnostics hold the observed state fixed; bypassing the mask in a new
closed loop was not tested.

Rolling had earlier actual dispatches: mean original-timetable early
displacement was 2268.04 s/trip (about 38 min), versus 319.70 for Aligned.
It was better on readiness measured against its own plan (1.81 vs. 3.65
s/trip), but worse on journey and actual CV2. The finite forecast with
unconstrained retiming is an informative comparison, not an operator-ready
implementation of a published-time-constrained optimizer.

On 50314/53020003, Aligned first lacked a bus at 07:33:16 at terminal B
(trip 39); vehicle 12 returned from trip 24 at 07:35:39 and departed one
second later. In total, 45 of 47 denied trips were at B; the last denied
trip departed at 18:49:57. NoFreq first lacked a bus at 14:21:15 at terminal
A, had 12 denied trips, and released its last denied trip at 17:12:15.
This is evidence of asymmetric physical supply on this case, not proof
that holding caused the failure.

The independent [MBTA stop-event summary](confirmation/field.json)
covers 4,688 complete trips across 23 weekdays in October 2025. Inbound
and outbound median observed times were 27.18 and 31.53 min; the selected
Aligned simulation day gives 28.14 and 26.38 min. The simulated outbound
is faster. Different endpoints and the selected simulation day make this
a plausibility check, not a matched calibration error. The initial HPC
download failed certificate validation; the gateway downloaded the
official archive once, and its ISO time strings were parsed from the
existing copy. The agency file has neither a vehicle ID nor a block ID.

The direct [snapshot check](confirmation/field_vehicle.json) found no
trips with stopped observations at both endpoint IDs. A supplemental
[time match](confirmation/field_crosswalk.json) with the Bus Observatory
archive linked 801 of 1,015 usable agency trips (Oct 6--10). For consecutive
opposite-direction trips of the same observed vehicle, the endpoint-to-start
gap medians were 8.22 min at Haymarket (167 pairs) and 7.78 min at Woodlawn
(193 pairs). On the selected simulated anomaly day the same-vehicle medians
after inbound and outbound trips were 3.84 and 18.04 min. These are selected
terminal gaps, not passenger-ready, driver-ready, or verified block records.

The [aggregate](confirmation/aggregate.json) can be rebuilt without
running traffic via python3 analyze.py. This is mechanism attribution on
the already-used V504 evaluation cohort, not another independent test.
