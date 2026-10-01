# Stage63 Episode Credit Result

Tests t116620/node004: six passed,54.120s, including the existing Stage46 entry regression. Production archive preflight t116623/node006 and qualification t116626/node001 completed successfully. Four option controls reproduce Stage60 exactly; upper networks/Adam are paired and calibration actors remain frozen.

Preflight uses24 existing archive episodes,7200 lower/108 upper reconstructions and7200 extra episode-value calls. It executes32 lower actor/96 lower value/8 upper actor/40 upper value optimizer steps; all40 actor proposals are retained, with no extra guard interpolation. No new native steps, evaluation paths or forecaster fitting. Checkpoints stay on the server; local compact JSON is11KB.

All preflight episode-MC EVs are positive (0.01041-0.01096), and episode-MC MSE is below the option critic by only0.0025%-0.0179%. This clears the frozen mechanical preflight, not a meaningful critic-fit or reward claim. Recalibrated advantage sign disagreement is14.5%-15.8% at period50 and4.8%-5.0% at period100. Both treatments move; conditional mean KL stays below0.02.

## Next

Run the frozen eight-root full archive comparison with16 critic-calibration iterations/eight paths per iteration. Retain every root/case. Native evaluation stays HOLD unless every active actor moves under the same mean-KL budget and every episode critic has positive probe episode-MC EV and smaller episode-MC MSE than its option control. Passing remains a prerequisite, not performance evidence; inspect actual EV, bias and effect size before adopting the change. Repair of the nearly constant upper critic is a separate intervention.

Full preregistration committed as c215033712 before full outcomes. Scheduler t116628-t116635 are running dynamically on node004/005/006 (allowed pool node001-node006,no pin). All eight roots have logged exact control/shared-upper checks for their first period50/zero_train cell; full four-case qualification is pending. Each task has9 CPU cores; declared RAM is12GB, while current measured task RAM is approximately2.5-2.8GB. No raw archives or candidate weights are pulled locally.
