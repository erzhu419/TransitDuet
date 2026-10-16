# Stage270: Shared Risk Heads

t165779/node006: done/0, 60.99s, no retries; 21 tests passed. 32 cells,
6,336 risk solves, no raw-reference fits or native episodes. Pulled 1.42 MB only.
Direct/signed MSE costs agree within 3.39e-13; event gradients below 4.37e-7.

Primary penalty .01, mixed domains: hurdle MSE beats direct MSE by 2.65% on
training states with opposite futures, but is 80.93% worse on excluded prefixes
and 20.52% worse on held prefixes. Raw-reference MSE: .000655806 / .000895036;
hurdle: .002162356 / .001272942; simplex: .000954142 / .001025003 (OOF / held).
Both bank directions fail. All declared penalties remain worse than raw in these
mixed-domain tests. Amplitude oracle MSE is only 4.39e-6 / 5.29e-6. Projection has
local benefits, including root397 held labels, but no mixed-domain win.

Decision: do not deploy standalone risk heads. Next Stage271: retain raw and
cached learned-risk forecasts as frozen baselines; compare matched cold-start
versus prior-offset residual fitting. Zero residual must reproduce its prior.
Keep service, features, bank assignment, actions and selectors fixed. Determine
correction bounds on training only; test prefix-label OOF and opposite-bank error
before any native rollout. No raw-reference retraining.

Limitations: reused development labels, not seed CI or policy gain. This test
does not separate low-rank representation loss from estimation noise or loss of
the historical prior. Protected Stage253/248/169 remain closed.
