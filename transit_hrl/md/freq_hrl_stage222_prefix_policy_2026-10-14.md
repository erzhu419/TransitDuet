# Stage222: prefix/future and selected-action calibration

Stage221's prediction-only shrinkage reduced noise overfit but lost mixed_b wait gains. This stage reuses all final Stage221 models and compares prefix/future risk-only shrinkage against the same objective constrained by selected-action cost/wait floors, separately for each source root.

Each outer training cell uses ten leave-one-complete-prefix-out folds crossed with two complementary inner future splits: 54 training states/two draws, six validation states/other two draws. Only training-internal OOF outcomes select alpha. The constrained solver enumerates causal gate menu events; alpha=1 is the unchanged raw controller and meets its own relative floors.

32 unbound CPU1/RAM3072 fit tasks across node001-node006. New budget: 640 analytic cost fits, 640 GLM fits, 1,920 wait fits at 600 updates each (1,152,000 updates), 192 scalar calibrations, zero new native episodes/actor updates. Raw predictions and final models remain remote; only the compact report is retrieved.

## Status

Eight focused tests and six Stage221 control tests passed, including real-model endpoint reproduction and prefix/future label isolation. Scheduler dispatch pending. Default/native policy unchanged.

## Next Decision

Check whether prefix holdout closes the remaining prediction gap, and whether selected-action floors preserve mixed-view wait/cost benefit better than risk-only shrinkage. Compare all Stage221 controls before interpreting new effects.

## Limitations

Inner training uses nine prefixes/two draws versus ten/four in the final model. Floors are relative OOF means, not held-control guarantees. The same 240 states/eight futures are reused; roots bundle lower-policy and demand changes. This is still one-action supervised development, not independent-seed CI or end-to-end HRL.
