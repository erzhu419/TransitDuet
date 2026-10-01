# Stage77: coherent residual calibration

Frozen on 2026-10-02, before new native rewards.

Stage75 found harm from the sampled upper curve; Stage76 traced a large part of
the lower input mismatch to planned velocity. Stage77 tests one historical-only
repair, without training or changing the Stage67 HOLD.

For each frozen Stage55 clone and period, reconstruct every actual BC label
state. Reuse Stage76 marginal upper proposals and reconstruct the original
clipped curve. Change both reference-error columns and planned velocity while
holding the physical state/history fixed. Let e be BC command MSE and d the
original curve's conditional command-change RMS. Freeze alpha=min(1,sqrt(e)/d)
(alpha=1 if d=0), once per root/period, before native evaluation.

Execute p_alpha=p_base+alpha*(p_original-p_base), rounded to float32. Reference
and velocity come from this same curve, preserving the anchor and world bounds.
No independent velocity clipping, reward-selected scale, or optimizer update.

Preflight: one root, four fresh seeds, periods 50/100, horizon 300; 24 probe
episodes plus 16 original-execution equivalence replays. Full: eight roots,
32 fresh seeds/root, horizon 1200; 1,536 episodes / 1,843,200 native steps.
Compare zero, original and calibrated curves under paired environment seeds,
stepwise lower noise and identical upper proposals. Report all 12 reward and
tracking contrasts with equal-root bootstrap (65,536 draws), Bonferroni12 CIs.
Velocity support and nonlinear command response are descriptive diagnostics.

Scheduler uses node001-006 dynamically: 3 CPU/3 GiB preflight, 9 CPU/8 GiB full.
Only completion markers and compact JSON are pulled; no new traces/checkpoints.

## Limitations / Next Step

The RMS ratio is a calibration rule, not a guaranteed nonlinear command bound.
These are teacher-initialized development roots, not joint-HRL training or a
frequency-superiority test. Retain every contrast and negative result. Judge
whether the repair restores native performance before revisiting joint credit;
do not adopt a policy or retune alpha on these evaluation rewards.
