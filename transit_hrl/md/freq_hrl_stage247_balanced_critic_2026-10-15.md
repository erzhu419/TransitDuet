# Stage247: state-balanced paired GLS

Run `native_transit_balanced_critic_stage247_development_20261015_r1`:33 tasks
done/0, no failures/retries (t155839, t155843-t155873, t155874);15 tests passed.
All184,800 Stage246 control fields reproduce exactly. Only the2,722,298-byte
aggregate was manually pulled. Zero raw-reference refits/native episodes.

## Design And Results

Freeze B targets/shrinkage,198 features,17 candidates, TSVD ranks1/3/8/16, head
contrast, A RMS cap and OOF gate. Primary `state_gls_pooled_safe` sets total mass
to2 per active state while preserving all pair/head ratios; `state_head_gls`
sets each active state/head's mass to1. Full-fit maximum mass error1.78e-15. Mixed Transit
held-prefix/fresh8 deltas versus raw, negative better:

| Pooled-Safe Objective | Cost Delta | Wait Delta (min) |
|---|---:|---:|
| Frozen All-MSE | -0.000129902 | -0.000213411 |
| Frozen Pair-GLS | -0.000097005 | -0.000427214 |
| State-GLS (primary) | -0.000227328 | -0.000584896 |
| State/Head-GLS | -0.000114673 | -0.000494271 |

Primary pooled cost/wait improve in both fresh halves. Within mixed views, root401
pooled means turn favorable (-0.000278409/-0.000516667), but its half1 still harms slightly.
Mixed_b also harms (+0.000005826/+0.000059896); dedicated single-root models have
adverse cost means. Primary changes18 states, with3 sampled fleet-event draws.
Versus raw, cost/wait are2.25/2.54 conditional MC SE from zero. Paired improvement
over old pair-GLS is only1.69/0.91 SE. Primary half-safe gains are near zero.
This improves the development candidate, not the default or a general superiority claim.

## Next And Limits

Freeze current actions; pre-register64 new independent futures (32+32) for
state-GLS, pair-GLS, all-MSE and raw. Replay their precomputed action-divergence
union with common future draws; keep new outcomes out of fitting and selection.
Do not continue tuning weights against these8 reused draws. Current limits are
development/conditional-MC evidence,3 fleet events and mixture/future-half
instability, not independent training-seed CI or end-to-end HRL. No consistent
root401 downweighting was established. Defaults/reserves unchanged; models remote.
