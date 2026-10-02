# Stage95: independent staged-upper confirmation

Confirm Stage94 using fresh upper-training and native evaluation samples, with the same eight teachers and both fixed final Stage93 U0-trained lowers. Both uppers restart from U0; Stage94 upper weights are not reused.
Preserve periods 50/100, horizon 1200, 8 updates x 64 episodes, upper conditional KL 0.0005, and 32 evaluation episodes/root/period. All scenario, noise and evaluation roles shift Stage94 by 1,000,000.
Preserve all eleven compositions, 28 contrasts and four primary endpoints. Equal-root bootstrap 65,536, seed (95,95095), Bonferroni28; confirmation requires all four primary lower CI bounds positive, with Stage94 and Stage95 reported separately.
Full additional budget: 22,016 episodes / 26,419,200 steps, 256 upper updates, 32 final checkpoints, 48 donor loads, 176 compositions and zero replay forwards. Donor loads and freeze checks cover both Stage93 lowers and the unchanged Stage88 joint reference.
Native preflight: root 310011, horizon 300, 2 updates x 8 episodes, 4 evaluation episodes; 152 episodes / 45,600 steps, 8 upper updates, zero checkpoint writes. Preflight gates only mechanics, not performance.
Scheduler uses dynamic node001-node006 without pinning; full eight-root dispatch requires preflight pass. Only compact JSON/logs/completion markers are pulled, and checkpoints stay server-only.
Limitations: this confirms upper-sample robustness conditional on existing lowers, not fresh lower training or unseen teachers. New-teacher validation is next; Stage94 evidence and earlier failures remain unchanged, with no retuning, pooling or root/period selection.
