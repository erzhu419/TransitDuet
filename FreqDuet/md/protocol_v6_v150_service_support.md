# V150: read-only V149 support and fallback diagnosis

Consumed V149 paths only, no new prediction candidate, replay, fitting or training.
Freeze D7996d60b4 points, V146(3s,0s),3600s window,.1/.1 tails and main policy.

Reconstruct the exact saved V149 per-stop service calculation before attaching
labels. Fixed support strata: no same-phase sample,1--8,9+ completed samples;
nine is the established descriptive convention, not an accuracy guarantee.
Separate query minimum support, per-stop support and retained structured/opaque
boundaries. Episode-weighted contributions must recover the previous11.9s
downstream-service bias; report conditional bias/MAE separately. Empty crossed
groups stay inconclusive. Use existing5000-draw actor/day crossed bootstrap.

For retained fallbacks, distinguish strict-segment boundaries, observed-launch
evidence and completed-service survivor depletion. A same-time archived timestamp
does not establish availability or callback order. Current follower load captured
at a later ready event is not an input at an earlier query; own-bus load is not
follower load. Inspect archived schemas and prefix support, not hypothetical fields.

All tests/analysis through scheduler on node001--006; only summary JSON locally.
Keep the original V149 gate and do not tune on these confirmation days.

Supplement: read the saved ready-label insertion order to separate a follower's
already completed same-clock callback from a later callback in that tick. Require
the same physical follower and reconcile the full service-fallback population.
This establishes event observability only; no new points or future-label inputs.
