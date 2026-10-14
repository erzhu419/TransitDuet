# Stage226: joint cost/wait action increments

Stage225 does not justify contender-only wait weighting: rare high-magnitude mistakes, future sign flips and imperfect cost ranking all remain. Test joint physical increments instead.

## Protocol

Two matched 3410-parameter critics: relative to original action or conditioned raw action. Same210 causal inputs/width16, two residual outputs over ridge cost/wait, own-training RMS targets, AdamW600. Select lowest predicted cost increment with wait increment<=0; exact anchor-first ties. Frozen GLM/raw/c2v4 controls stay unchanged.

Each outer cell uses20 inner fits: leave one six-callback prefix out, train54 states on two futures and validate6 on the other two, both directions. Refit each inner raw reference without the held prefix/futures; reproduce Stage223 OOF predictions. Full final raw weights cannot be used for inner selection. Final two critics fit60 states/four futures for600 steps.

Compare zero/last600/MSE-selected/joint-selected snapshots of each identical training path. Joint selection minimizes worst source-root normalized cost/wait increment versus raw on internal validation; if none beats zero, retain raw. Earlier checkpoints win ties. All paths run600 even if an earlier snapshot is selected; outer labels never choose checkpoints.

32 unbound scheduler tasks plus one report,1 CPU/3 GB each.640 reference fits,1280 inner+64 final joint paths:1,190,400 updates total, zero native episodes/actor updates. Anchor arms have equal compute; frozen controls do not have equal total new training compute. Only the aggregate report is pulled; models stay remote.

## Results

Fits t152141-t152172 and report t152175 completed with exit0. Eight tests passed; inner raw predictions reproduce exactly, and240 external controls within1.39e-17. Only the471,964-byte report was pulled, no models/raw. Default unchanged.

MSE selection chooses zero steps in32/32 raw-anchor and31/32 original-anchor cells. Raw-relative last600 mean normalized training MSE is about0.036 versus held-other2.865, worse than zero-step held1.240. The new nonlinear joint objective overfits; this is not learned improvement over the ridge forecast.

Mixed cost/wait both negative: raw/c2v4 8/8, original-selected7/8, raw-selected6/8; last600 original1/8/raw4/8. Original-relative and raw-relative objectives both retain performance tradeoffs.

## Next Decision

Fix the actual checkpoint aggregation issue below using saved paths, without retraining. Then assess refit/held transfer and regression overfit; a zero-step or raw-hold winner is not learned improvement.

## Limitations

Same240 development states/eight draws, no new independent CI or native learned HRL. Inner54/two-draw versus final60/four-draw refit transfer remains testable, not guaranteed. A predicted joint improvement or internal raw-hold rule is not an actual held no-tradeoff guarantee.

Known issue: five selected anchor/cell cases have negative fold-normalized joint scores but a positive pooled physical cost or wait increment. Averaging after different fold-RMS divisions can reverse the physical mean sign. Stage227 will pool physical increments first and use one positive training-derived RMS per source, retaining these v1 results and all trained weights.
