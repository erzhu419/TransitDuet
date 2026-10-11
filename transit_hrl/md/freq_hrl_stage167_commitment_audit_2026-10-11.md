# Stage167: Score Horizon Versus Executed Commitment

Stage166 makes the supply term responsive without beating window forecast on
both roots. Stop model/weight tuning and test one credit-horizon mismatch:
the score evaluates the entire remaining tentative plan, but only the next six
departures are committed before the same-direction upper replans.

Passive audit only. Execute the identical Stage166 selector with the same fitted
mean/residuals, native RNG, lower weights, scores, cap, clock and commitments.
For every candidate additionally clip the same occupancy ensemble at its next
same-direction callback: seventh proposed departure minus the native lookahead;
the final block ends at the evaluation clock. Record a six-interval waiting
proxy including that next departure anchor, without changing selection.

After execution use actual launch/completion times to reconstruct joint-direction
occupancy within each selected window. Its irreversible peak must match the next
same-direction historical peak (or final episode peak). Verify that the first
six commitments AND the next-block anchor retain their final planned times;
measure how much the remaining proposed tail was subsequently rewritten.

Two roots, five regimes, all four Stage166 scenes: 40 frozen episodes, 2,455,200
native ticks and 40 exact physical/schedule/original-selector reproductions.
Report tail-only fleet discrimination, full-versus-prefix proxy argmin mismatch,
selected-window fleet-cost prediction error and rewritten tentative tail size.
If mismatch is substantial, test commitment-aligned credit/scoring next; if not,
investigate the occupancy model or action authority instead of more seed runs.

Ten unpinned scheduler tasks on node001-006, 1 CPU/3 GB each. Only code and new
run model inputs are staged. Native episodes run remotely; only result JSONs
with compact trip times are pulled, never raw CSV/checkpoints.
44 focused tests pass, including scalar/window occupancy, unchanged original
scores/actions/dispatch, next-anchor persistence, trip accounting and scheduling.

Run `native_transit_commitment_audit_stage167_frozen_20261011_r1`, revision
`1f6864c035`: t142372-t142381 are running on node001/004/005/006 after scheduler
staging/dispatch. Full 40-trajectory and window accounting qualification remains
pending; no new performance claim is registered.

## Limitations

This audit measures the executed policy, not counterfactual outcomes for the
four unchosen actions. Prefix scores are diagnostic, not a deployed policy or
native passenger-wait objective. Frozen reused scenes do not establish learned
upper authority or new performance gains.
