# Stage85: matched-budget training order

Stage84 established positive direct upper effects but a weaker joint-trained lower.
This test compares paired joint, alternating lower/upper, staged lower-then-upper and lower-only training.

- Same source policies and frozen Stage78 decoder, std, values, optimizers and forecaster; fresh Stage85 seeds.
- Per method/period:16 credit chunks, 32 native episodes/chunk, total512 episodes; eight fixed roots, periods50/100, horizon1200.
- Paired joint: collect both chunks under one unchanged policy, then update lower from the first and upper from the second.
- Alternating: update lower/upper after each successive chunk. Staged: eight lower updates followed by eight upper updates.
- All three dual methods: eight updates/level, 32 episodes/gradient, nominal conditional KL0.0005/update, cumulative0.004/level.
- Lower-only: pool each pair of chunks; eight lower updates, 64 episodes/gradient, KL0.001/update, cumulative0.008 total.
- Every method has the same total environment samples and cumulative nominal KL0.008; no allocation or radius sweep.
- Only final policies evaluated, 32 fresh paired evaluation seeds/root/period. All22 contrasts use one equal-root bootstrap65536/Bonferroni22 family.
- Preflight: one root, horizon300, four chunks with eight episodes each, four evaluation seeds; mechanical pass only.
- Full budget:35,840 native episodes / 43,008,000 steps; scheduler node001-006 dynamic placement, nine CPU/8192MiB per root, eight workers.
- Final inference weights remain server-side; no intermediate checkpoints/traces or local checkpoint pulls.

Decision: use order contrasts among the matched dual methods; separately test each against lower-only at equal total budget.
If order changes do not close the lower-only gap, retain that result instead of selecting an allocation post hoc.

## Limitations
Paired joint assigns independent half-batches by level, unlike Stage83's shared full gradient batch. Lower-only pools both chunks.
Cumulative nominal update-state conditional KL does not match trajectory KL or parameter path length. Teacher initialization, fixed decoder/std, Stage67 critic-route HOLD and the closed frequency-superiority claim remain unchanged.
