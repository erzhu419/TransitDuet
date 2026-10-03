# Stage97 Fresh Decoder Cohort Result

- Tasks t128664-t128672 finished with exit code 0. All eight full new-teacher roots and both periods passed exact server-side reaggregation.
- All 16 decoders reproduced saved BC MSE, retained the first-feasible contraction, paired upper noise and froze source/Adam state. Command-change / BC-RMSE ratios span 0.5353-0.7242.
- Period50 scales span 0.02140408-0.02460535 (three response passes each); period100 scales span 0.06210447-0.09705902 (two passes each). Root410011 reproduces its preflight scales without adjustment.
- Native budget: 128 episodes / 153,600 steps. Offline budget: 128 label archives / 153,600 reconstructed state rows / 691,200 actor rows / 1,152 batches / 40 response evaluations.
- Runtime: 14.29-14.78 seconds per root; no optimizer, checkpoint or native-trace writes. Scheduler RAM sampling was unmeasured.
- Equal-root probe reward changes, bounded minus zero: period50 -0.0989 (4/8 positive); period100 -0.6550 (2/8 positive). Keep these mixed diagnostics; neither is a performance CI or a scale/root admission rule.
- Only 38,198 bytes of server-reduced JSON were retrieved. Checkpoints, raw labels and native paths stay on the server.

Next: rebuild joint-call, joint-level and lower-only final mean learners from the new clones and frozen decoders using the existing Stage88 core and budget. Then rebuild matched lower donors before the unchanged staged-upper confirmation; do not pool this cohort with Stage94/95.
