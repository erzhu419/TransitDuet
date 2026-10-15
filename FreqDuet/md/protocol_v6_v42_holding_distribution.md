# V42: categorical holding and cumulative forecast bias

Replace the V41 mean-to-nearest-bin continuation with probabilities over
0/5/10/15/20/30/45 s. Fit a 12-feature multinomial logistic model to at most
4096 strictly earlier observed decisions, using the existing Torch LBFGS
optimizer, ridge coefficient 10 on non-intercepts and 0.5 pseudo-count per bin
at the prefix mean feature vector.
Do not class-balance: the observed action frequencies and first moment matter.
Fewer than 32 labels use the smoothed empirical prefix distribution.

At 2/6/10 h, propagate 64 discrete trajectories for 15/30 min with private RNGs,
fixed seeds 20261015 + draw index. Preserve capacity, masks and physical reuse.
Historical-stop-mean and V41 rounded-response remain matched comparators.
Compare predicted cumulative holding with actual cumulative holding, and score
the same frozen distribution on future *observed* states separately to locate
action-fit versus state-propagation bias. Report passenger exposure/backlog,
per-day signed errors, Monte Carlo standard errors and nominal 90% bands.
Keep nonreturned current trips censored in return-window probabilities/bands.

Development: actors 12011/933013 x demand 45007/934011.
New-demand audit: same actors x demand 45013/934029 (unused by V41).
The latter demand seeds appeared in older diagnostics, so this is fresh to
this forecast model, not a wholly untouched policy/scenario confirmation.
No parameter selection between cohorts. Verify every instrumented full day
against its archived V35 U200/L200 trajectory before aggregation.

Freeze the engineering screen before runs: on both cohorts at 30 min,
holding MAE <= historical, absolute holding bias <=5% of actual mean,
waiting/onboard MAE <=105% of historical, holding-band coverage >=80%.
These are forecast diagnostics, not policy-superiority tests. The bands only
sample holding uncertainty; future upper replans and flow/travel uncertainty
are not covered. No main-policy switch in this step, regardless of the screen.

Run smoke/tests, eight full days and aggregation through scheduler on
node001-node006. Retain raw forecasts/labels remotely; pull small summaries only.
