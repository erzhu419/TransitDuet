# Stage295: Frozen Screening Signal Transfer

Stage294 validates new-prefix waiting/service gains but not joint cost/wait or
fleet information. Both E16 halves improve; both cost uncertainty summaries
cross zero. More futures alone cannot remove scene heterogeneity.

One unpinned scheduler CPU/3 GiB task reuses server-side S8/E32 panels, all
240 prefixes, 1,440 states and 16 fixed profiles. No fits, new simulations,
threshold searches or policy choices. Stage294 remains closed confirmation.

Compare selected S8 means with E32 and both E16 halves. Partition retained
negative fleet means and signed cancellations; partition expected thinning
into zero-event rejects and nonzero retained decisions. Contributions use
the full denominator and frozen matching, with no unobserved nonzero weights.
Reproduce all published paired means and half means. Reuse Stage293 prefix
attribution and shared-future variance formulas; count physical net fleet
events after profile averaging, never repeated profile exposures as seeds.

The output decides what a new training experiment must explain: screening
optimism, sparse fleet events and/or persistent between-prefix differences.
It cannot replace Stage294's failed joint criterion or select a new controller.
Only scalar JSON is pulled; prefix records, panels and models stay remote.

Run: `native_transit_screen_transfer_stage295_20261018_r1`; scheduler task
`t183620`, worker revision `8976972d7b`. Twenty-nine focused/regression tests
pass, including category conservation, independent fleet reversal, profile
duplication, frozen-mean/half replay and remote worker input wiring. The small
parent completion receipt travels inside this task's registration input;
parent output directories are never staged.

## Result and Decision

Task t183620/node006 completed done/exit0, with a 24,300-byte scalar report. All frozen paired
means/E16 halves reproduce within 5.64e-18; no fits or native simulations.
Abstention's selected S8 fleet mean is -1,893.45 x1e6; independent E32 is
+62.84 x1e6. Service remains favorable (-146.11 to -164.43 x1e6). This is
large post-selection fleet optimism, not a lost service signal or replay bug.

The 731 negative-S8-mean decisions contribute E32 fleet +20.35 x1e6
(E16 halves -35.94/+76.63); 133 signed-cancellation decisions contribute
+42.50 x1e6 (both halves positive). Deleting cancellations alone is not a
validated remedy. These are contributions with the original full denominator.

The 864 profile-state changes cover 234 physical states/135 prefixes. Against
raw, 109 prefixes improve waiting, 26 worsen and 105 are unchanged; the largest
waiting-gain prefix supplies 8.4% of net gain. Thus the new waiting result is
not a single-prefix artifact. About 60% of analytic cost variance is attributed
descriptively to between-prefix variation after subtracting shared-future MC
variance, versus raw or thinning. More futures alone will not remove it.

Close this diagnostic and the Stage294 confirmation. Retain waiting/service
transfer; do not claim fleet safety, cost noninferiority or joint improvement.
A new method needs calibrated paired fleet-risk evidence on independent
training scenes, followed by untouched scene confirmation. Do not select a
replacement gate or cancellation rule using these E32 outcomes.

Limitations: category/event/variance decompositions are descriptive. Net fleet
event signs follow profile averaging, not individual vehicle-event counts;
S8 selected means are not independent performance estimates. No new policy
was learned or promoted.
