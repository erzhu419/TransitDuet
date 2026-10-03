# Stage97 Native Fresh-Decoder Preflight

- Tasks t128471/t128472 completed on node004 with exit code 0. Exact server-side reaggregation reproduces the saved preflight qualification.
- Root410011 uses the full Stage96 BC clones and all eight 1200-step label paths per period; it does not use the short Stage96 preflight teacher.
- Period50: first-feasible alpha 0.02223362 after 3 response evaluations; command-change RMS 0.03797466 <= BC RMSE 0.06499178.
- Period100: first-feasible alpha 0.06479194 after 2 response evaluations; command-change RMS 0.04444746 <= BC RMSE 0.06244186.
- Both decoders froze before native probes. BC MSE reconstruction, shared upper-noise pairing, source/Adam freeze and exact realized-budget checks passed.
- Cost: 16 native episodes / 4,800 steps; 19,200 label-state rows; 86,400 offline actor rows / 144 batches / 5 response evaluations. No optimizer, checkpoint or native-trace writes. Runtime 10.25s; scheduler RAM sampling was unmeasured.
- Probe means, zero -> bounded: period50 219.4180 -> 219.5512; period100 194.8822 -> 194.7440. These four-path diagnostics do not select alpha or roots and are not performance confirmation.
- Only a 7,435-byte compact JSON was retrieved; checkpoints and raw labels remain server-only.

Next: launch the already frozen 8-root Stage97 full decoder prerequisite (128 native episodes / 153,600 steps), then rebuild fresh joint/lower donors. No protocol change, root selection or pooling with Stage94/95.
