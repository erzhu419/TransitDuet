# V406: Fixed-Comparator Holding Allocation

V405 preserves native bilateral regularity by retaining much more holding than
the strong stop-cap comparator. This independent candidate aligns its internal
reference with a predefined simple control policy instead of tightening a trip
quota and then protecting the tightened reference.

The reference is native bilateral clipped to 10 seconds at every query, with
fixed public releases. Candidate commands retain the full original seven bins,
including values above 10. Every future model step uses the same cap rule,
including fallback and pending-transfer intervals. The reference is a policy,
not a candidate-dependent allowance. No original trip budget is imposed.

Keep V405's passenger/CV2/public-delay/readiness guards, paired-stop fallback
checks, non-early release candidates, censored-demand scenarios and original
12 physical vehicles. Single-action, paired-transfer, and joint publication
arms use the same predictor and reference. A direct simple-cap baseline and
the unchanged legacy V404 cap_10 controller are both evaluated, not relabeled
as equivalent controls. The old V405 failed paths are not replaced or reused.

Twenty physical starts: four fresh development days x five arms, scheduler high
on CPU nodes 001--006. No training, automatic retry or main-policy promotion.
The original zero-nonharm criterion remains: every day must be nonworse on
actual CV2, public delay and readiness against both simple controls, with lower
mean journey within each regime. Decompose transfer vs single and joint vs
transfer before attributing benefits to either mechanism. These are four-day
development outcomes, not independent confirmation or a frequency-learning test.
