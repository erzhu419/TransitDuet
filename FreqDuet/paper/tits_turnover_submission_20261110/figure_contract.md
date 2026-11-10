# Figure contract

Python/matplotlib continues the repository's existing analysis workflow.
All statistics and plot generation run through scheduler high on the server.
The local copy only inspects compact results and compiles the manuscript.

- Calibration tradeoff: show the finite four-day development grid, not an
  independently verified or optimal Pareto frontier. Parameter-order curves
  connect only observed grid settings; the pressure point is not selected anew.
- Budget retention: connect only the two frozen settings on the same 16 days;
  show journey, readiness and holding jointly with CV2, not a continuous frontier.
- Independent tradeoff: show the original frozen pressure/static/cap means on
  16 independent days; use the existing paired confidence intervals in the table.
- Reference tradeoff: distinguish the 16-day allocation study from the four-day
  common-reference development study. No cross-study paired comparison.
- Factorial coordination: separate timetable, budget and interaction under the
  same model fast layer. Four independent days per predetermined regime.
- Regime robustness: show journey, dispatch and CV2 contrasts together. No new
  weights or service-quality tolerance are chosen from these outcomes.
- Vehicle-time mechanism: use the predeclared first nominal day and first issued
  trip budget, not a selected success. Submission times, executed command amounts,
  physical returns and next dispatches are evaluation labels, not controller inputs.
- Passenger cohorts: show timestamp-based journey differences by generation hour
  and origin station with the original whole-day mixture probabilities. Cohort
  passenger weighting is separate from the primary equal-demand-day estimand.

Exports are editable PDF and SVG with a PNG preview on white, at approximately
182 mm double-column width and 7--8 pt labels. Source JSON accompanies the
scripts. Legends define units, paired demand-day sample size and intervals.
There are no generated photographs, fabricated trajectories or smoothed data.
