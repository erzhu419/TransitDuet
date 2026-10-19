# V142: fixed finite-sample rank intervals

V140/V141 point forecasts are read unchanged, not regenerated. Use only saved
V137 paths and strict completed labels; no replay, training or action selection.
Tests/audit run through scheduler; full details stay remote.

Keep each tail probability0.1 and the original1h/all-past window rule. Replace
linear interpolation by ranks r=floor((n+1)/10), s=n+1-r. For n<9 no finite
upper order statistic exists: upper=null, explicit insufficient-support status;
physical lower support0 is not a zero duration estimate. No-survivor cases stay
missing, not zero-imputed. No pooling, window expansion or quantile search.

Report finite availability, unavailable counts, and paired old/new coverage and
width on exactly the same finite support, separately for service and unissued
ready residence. Never count unbounded intervals as coverage successes.

Rank reference: [Dunn (2021), Theorem4.2.1 and Appendix D](https://robinmdunn.github.io/files/RobinDunn-dissertation.pdf#page=78).

Limitations: exchangeability is not established for adaptive completed-window
bus data, so rank theory is not a field-calibration guarantee. Current-stage
duration intervals are not full-route ETA uncertainty. These consumed selected
paths cannot establish independent calibration or policy gains.
