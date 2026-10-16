# Stage264: Head, Population and Future Halves

Reuse Stage249 labels and Stage260/263 models; no fitting, simulation or selection.
Freeze model, rank/key, basis, reliability, GLS precision, lambda, prefix KKT
multiplier/operator and cap. Split the target-swap response into excluded/retained
training labels crossed with service/risk heads; keep all six signed cross terms.

Each bank's assigned 16 futures split into first8/last8. Cross16 equals the mean
of the two matched cross-bank half contrasts; their difference equals the
within-anchor minus within-other contrast. No independent-policy/seed CI claim.

t165689: node006, exit 0, no retries, 46.49 seconds; seven tests passed.
32 cells / 1,280 records qualified. Stage263 target-swap RMS reproduced within
5.14e-15; target reconstruction exact; squared decomposition error <= 6.28e-15.
Pulled only the 2,764,337-byte scalar report; raw ledgers remain remote.

Mixed removed held queries: 12 bank-cells / 120 active prefix-models per family.
These are total-cost function responses to target perturbations, not outcomes.

| Model | Service RMS | Risk RMS | Joint RMS | Service/risk half cosine |
|---|---:|---:|---:|---:|
| free_old | .003846 | .001997 | .004620 | .875 / -.040 |
| shrunk | .002929 | .002520 | .004169 | .825 / -.017 |

The focal prefix's two selected contexts are service-dominated: held service RMS
.027355 / .020380 versus risk .002100 / .002315. Its excluded-label response is
62.7 / 48.0 times the retained-label RMS: the six removed states' labels dominate,
not variation elsewhere in the remaining 54 states. Risk-target variability alone
does not explain this response; mixed_a and single-root groups are more risk-dominated.

The focal prefix accounts for 96.9% / 92.8% of the service component's held squared
response. Its service half cosine is .938 / .938; the descriptive complement's
cosine is -.068 / .048. Pooled service stability is not broad prefix replication.

Limitations: fixed-operator total-cost prediction responses to head labels, not
observed service/risk outcome changes. Anchor quantities depend on training;
half contrasts reuse those quantities. Nonzero paired risk counts are
draw-state-feasible-action exposures, not independent fleet events.
397/913970203 was designated from Stage263; the complement is posthoc aggregate
subtraction, not a training-data deletion. Component RMS values do not add.
Stage253/248/169 stay closed; no default/deployment change.

Next: propagate cached per-state, per-future joint service/risk label covariance
through the same KKT operator. Compare observed bank differences with conditional
MC-noise magnitude and locate amplified directions before new fits or cap changes.
