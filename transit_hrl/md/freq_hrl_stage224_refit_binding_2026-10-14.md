# Stage224: bind calibration to the final controller

Stage223 c2v4/final4 restored mixed cost/wait benefit to 8/8, but its final training decisions still violated OOF-only floors. This stage tests final-controller boundary drift directly.

Reuse the exact final4 models and c2v4 OOF predictions. Compare unchanged GLM/raw/risk/OOF controls with final-only and joint OOF+final constraints. Each source root must meet its own raw-controller cost and wait means; alpha=1 is always feasible. All modes minimize the same OOF physical-mass loss.

## Protocol

32 scheduler cells on unbound node001-node006; 1 CPU/3 GB each. Six scalar calibrations/cell; zero network updates, native episodes or actor updates. Final constraints use only 60 training states and their four training futures. Outer prefixes/other futures are evaluation only. Download only the compact report.

## Status

Six new tests and five Stage223 tests passed: boundary transfer, source-specific floors, causal prediction/label isolation, complete coverage, control reproduction and scheduler placement. Formal results pending; default policy unchanged.

## Next Decision

Compare held cost/wait and proper loss against both raw and OOF-only controls. Retain a candidate only if final binding improves transfer rather than merely reverting to raw. Training floor satisfaction itself is not performance evidence.

## Limitations

Development data reuse the same 240 states/eight future draws, not fresh independent seeds. Source roots also differ in lower checkpoints and demand/prefixes. This is one-action supervised selection with the original greedy suffix, not end-to-end learned HRL.
