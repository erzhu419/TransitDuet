# Stage268: Fleet Events and Future Influence

t165755/node006: exit 0, 80.06s, no retries; 22 tests passed.
32 cells / 1,280 records reproduce Stage267 risk responses within 2.08e-14.
Pulled 2.19 MB scalar report; labels, models and influence NPZ remain remote.

Mixed removed held queries, free_old / shrunk: event occurrence contributes
64.8% / 55.5% of sample risk scatter, within-event amplitude 35.2% / 44.5%.
Fleet action-pair events occupy 4.33% of model-future exposures; 71.9% of
state-bank-model exposures have zero observed events. Deduplicated counts:
442 / 7,680 future keys have an eligible event in at least one context.
Top 16 unique futures carry 38.5% / 41.1% of risk noise energy, not all of it.

Focal top16 shares are 65.0% / 69.8%. Deleting one cached future can reduce
its observed squared risk response 63.1% / 68.6%, but this is a frozen-mean
diagnostic. Root401 instead has 52.9% / 59.8% within-event amplitude scatter:
occurrence-only treatment is insufficient. Shared endpoint cross terms retained.

Limitations: local conditional scatter, not policy improvement or CI; repeated
model exposures are not new trials. Eligibility is context-specific; zero events
do not imply zero true probability. No fitting, simulation, selection or changes
to Stage253/248/169. Genuine tail events must remain in the estimand.

Next: Stage269 should audit signed action-level event probability and amplitude
targets across the two existing future banks. Verify their unregularized product
equals the direct mean; decomposition alone cannot reduce variance. Use this
audit to specify a controlled shared-risk-head comparison, without deleting tails.
