# V410: Within-Day Transition Comparator

Keep V409's capped V406, fixed10, heavy demand (1.2x), 12 vehicles, and
public timetable. The new rule samples a command from the empirical
0/5/10 transition row indexed by the current fixed10 projection, restricting
the row to the original legal mask. It uses no additional traffic state
and no event-graph estimate. Empty legal row mass falls back to fixed10.
Draw once per query from a separate day-seeded policy stream; all variants
on a day start with the same policy seed and share the original exogenous
event stream. This is not an assertion of identical query order across arms.

Estimate the 3x3 counts only from V409's two capped development traces,
2470201/2. Retain upward and downward changes. On those same old days,
run three simple variants weighting each legal row cell by
`exp(tilt * (seconds - reference_seconds) / 5)`, with tilt -0.5, 0, +0.5.
Select only among variants whose mean holding is within 3 s/trip and all
three mean action shares are within 0.02 of capped V406. Choose the smallest
absolute dose discrepancy, then smallest maximum share discrepancy,
then smallest absolute tilt and then manifest order. Use no outcome-based
selection. If none qualifies, do not release confirmation. Freeze the
selected table and tilt; no refit on V408/V409 confirmation days or V410.

Confirmation: 16 new days x fixed10/transition/capped V406. Joint gate:
the paired 95% day-bootstrap intervals for the mean holding and each bin
share difference must lie entirely inside +/-3 s/trip and +/-0.02,
respectively; only then interpret journey improvement with actual CV2,
readiness and public delay nonharm (upper interval bound <0 for journey,
<=0 for the other outcomes). Report fixed10 contrasts separately.
Matching these aggregate summaries is not matching exact commands or
establishing how much any individual feature contributes.

At most 54 new paths: six short rule calibration paths, then at most 48
complete-day confirmation paths. No replacements, auto-retries, training,
policy switch, or old result edits. Zero-traffic preflight, high/manual
scheduler on node001--006, small local summaries only.
