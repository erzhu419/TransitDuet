# V165: fixed fresh 64-day registration

Sealed by high/manual scheduler task `t208335` on node002, source `4a6d783667`;
six unit tests passed. New days 2951001-2951064 use samplers 1650000000-1650000063:
12 frozen actors x 64 days x old/handoff = 1,536 calibration paths.
V162's 2851001-2851032 remain development only and are excluded from the new fit.
Confirmation 2852001-2852032 and samplers 1600000032-1600000063 remain unchanged.
All eight method sections exactly match V161: same joint estimator, keys,
19-day minimum, weighting, ranks, native window, tails, comparator and gates.
Freshness: 45 result roots, 29,313 metadata files, 72,587 artifact names and 332
generators; no day/sampler conflicts, confirmation-use evidence or read errors.
Historical query caches are covered by the prior registry and published metadata,
not rescanned. Only the 6,835-byte summary was fetched locally.
No collection, fitting, training, confirmation or frozen-controller change ran.
Next: admit a passive new-cohort collector against consumed original paths,
then collect exactly the registered 64 calibration days after authorization.
No interim-driven additions or replacements; support failure ends this attempt.
