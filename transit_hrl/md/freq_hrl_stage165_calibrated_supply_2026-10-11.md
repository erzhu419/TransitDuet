# Stage165: Cold-Start Prior Control Value

Stage164 confirms the 784 s freeflow prior substantially understates operational
trip times. Use its registered fit-only direction means, unchanged after seeing
check outcomes: root397 False/True 2046.2504/2000.8458 s; root401
2049.6275/1990.2427 s. Replace only the pre-first-completion estimate. Each
direction's first real completion replaces its prior; subsequent EWMA (.2) is
exactly the original algorithm. Do not initialize EWMA with a prior pseudocount.

Four conditions: window forecast, legacy supply, calibrated demand-only and
calibrated supply. Window and legacy supply must exactly reproduce Stage163;
calibrated demand must exactly reproduce Stage163 demand-only physical outcomes
and schedules, since its fleet term is absent. These three controls execute
before calibrated supply in every scene. All strengths, scores, physical fleet
cap, lower weights, daily endpoints and 44 six-trip commitments stay unchanged.

Two roots, five regimes, all four registered Stage163 scenes: 160 episodes,
9,820,800 native ticks, 120 source-control reproductions, zero native/actor updates.
Primary comparisons: calibrated supply minus legacy supply, calibrated demand
and window forecast in actual cost, wait, fleet/nonfleet cost and episode reward.
Record initial predicted peaks, fleet-term decision differences, prior use and
historical-peak plateaus. A better duration estimate must earn physical value.

Ten unpinned scheduler cells, node001-006, 1 CPU/3 GB each. Stage only code plus
the compact Stage164 calibration directory; raw CSV/checkpoints remain remote.
Results pending. Net value on both roots is the prerequisite for learning the
allocation action; otherwise diagnose remaining candidate/occupancy authority.
42 focused tests pass, including first-completion replacement, unchanged EWMA,
demand-only dispatch neutrality, native commitments/goals, prior/control/score
qualification and small-directory unpinned staging. Tests use stub environments;
all full native episodes run remotely.

## Limitations

Frozen two-root development on reused control scenes, not learned upper authority
or independent confirmation. A day-level direction mean misses startup variation,
headway-dependent holding, future shocks and capacity. Occupancy remains a point
estimate, and the historical episode peak is irreversible.
