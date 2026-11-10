# V402: turnover-aware budget versus simple limits

Freeze manual joint, K4, native seven actions, cost/forecasts, 12 physical buses,
262 trips, and irreversible commands. The only change is the holding constraint.
Pressure uses the original rho=.50 law. Static issues ceil(c * remaining controls)
once per supported physical cycle, without supply pressure or replenishment.
Stop cap applies h<=cap to supported cycles. All policies begin at the existing
7200-second publication clock and use the same strictly completed history.
Limits enter plan supply forecasts, model continuation and actual execution;
already submitted commands are never edited. Future unpublished limits are not
assumed by any model. Timetable algorithms, not realized timetables, are fixed.

Reuse four V401 manual-rho=.50 development references. Run the sealed static/cap
grids on those days only. For each family choose the closest holding-exposure
brackets and freeze a daily ex-ante mixture matching development mean holding.
This is a simple state-blind randomized operating setting, not state-dependent
action mixing. Both constant components are evaluated to compute its expectation.
If no bracket exists, stop; do not extend the grid. All calibration results stay.

Use 16 new demand days, pressure plus at most four sealed simple components.
Report both components as well as fixed-mixture expectations. No test-day
matching, interpolation changes, checkpoint selection or new training.
Primary: total dispatch delay against the original public timetable, including
control waiting. Protection: actual CV squared, zero noninferiority margin.
Passenger journey/wait/onboard and readiness are reported, not silently folded
into the primary gate. Two paired comparisons use 97.5% one-sided bootstrap
upper bounds (Bonferroni across families), with demand day as the sampling unit.
Passing requires lower delay and non-harmful CV2 against both simple families;
it does not imply every passenger outcome improved or general optimality.

At most 150 charged starts: 3 engineering, 64 calibration, 80 confirmation,
3 reserved technical replacements with distinct receipts, no automatic retry.
Engineering compares pressure to V401 and cap45 to a native unrestricted plan
reference. Synthetic checks and metadata admission precede traffic. All work is
scheduler high/manual, node001--006, at most eight traffic paths. No promotion.
