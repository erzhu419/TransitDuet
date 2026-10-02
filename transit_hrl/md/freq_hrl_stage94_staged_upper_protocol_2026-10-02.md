# Stage94: lower first, then independent-credit upper

Stage93's global confirmation failed; retain its result without threshold, root, or period rescue.
The new question is whether sequential upper learning improves the U0-trained common-noise lower and beats the matched independent-lower sequence.
Reuse both Stage93 final U0-trained lowers for all eight roots; freeze each lower and initialize its upper from U0.
Upper training uses the original independent upper/lower noise pairs, never shared-upper replay.
Each branch: periods 50/100, horizon 1200, 8 upper updates, 64 episodes/update, conditional upper KL 0.0005/update.
All scenario, noise and evaluation roles shift Stage93 roles by 1,000,000; evaluate only the last update with 32 fresh episodes/root/period.
Eleven compositions include both staged branches, their U0 and fixed Stage88-upper references, actor swaps, joint, base and zero.
Four primary endpoints: staged-common minus source-common and staged-common minus staged-independent, each at both periods.
All 28 contrasts use equal-root bootstrap 65,536 draws, seed (94,94094), Bonferroni28; global success requires all four primary lower CI bounds positive.
Preflight is mechanical only: root 310011, horizon 300, 2 updates, 8 episodes/update, 4 evaluation episodes/composition/period.
Full additional cost: 22,016 episodes / 26,419,200 steps, 256 upper mean updates, 32 final checkpoints, 48 donor loads; no upper-replay forwards.
The lower-then-upper nominal call-weighted proxy is 0.008/branch; inherited lower compute is separate, not equal total compute or trajectory KL. Same teachers, mean-only learning, no new frequency-superiority or full actor-critic claim.
Dispatch preflight first through scheduler, dynamically across node001-node006; formal 8-root submission requires preflight pass. Pull only compact JSON/logs/markers; checkpoints remain server-only.

Preflight t126835/t126836 completed with exit 0: mechanical gate passed, 152 episodes / 45,600 steps, 8 upper updates, 6 donor loads, 22 compositions, zero checkpoint writes and replay forwards; wall time 44.48 s. No performance claim.
Formal frozen run submitted as t126885-t126892 (eight roots) and t126893 (all-root qualification), dynamic node001-node006 with no pin. Next: analyze all four primary CIs and all 28 registered contrasts after completion; no root/period selection or cross-stage pooling.
