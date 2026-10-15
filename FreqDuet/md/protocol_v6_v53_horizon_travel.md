# V53: completed-prefix horizon-conditioned travel mean

V52 identifies travel as the main late-clock error source. V50's fixed-origin
leg mean is competitive in 0-900 seconds but loses to V48 by 4.32 seconds MAE
in 900-1800 seconds; replacing it with the observable movement mean does not help.

Fit two bounded least-squares coefficients from completed legs only, using
predictions issued strictly before their departures. Smoothly interpolate the
coefficients between lead times 450 and 1350 seconds to form a convex mixture
of V50's current conditioned mean and V48's historical mean. Rolling training
uses at most 4096 prediction-label pairs in the past three hours; each leg can
inform separate previously issued forecast horizons. No future labels enter a
fit. Zero coefficients reproduce V50. Keep its seven-bin holding, passenger
process, marginal travel ratios and independent/shared noise unchanged.

First run scheduler smoke and 12-day prefix screening. At the six V52 clocks
require cohort short-lead MAE no worse than V50 and long-lead MAE no worse than
V48. Do not weaken this screen after viewing results. Only a passing model
proceeds to closed-loop forecasts; the real controller remains frozen. All
six demand seeds are development data, not new independent confirmation.
