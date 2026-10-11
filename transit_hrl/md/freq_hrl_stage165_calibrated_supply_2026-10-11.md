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
the new run's `inputs` directory containing the frozen calibration summary.
Historical result/raw directories are not sync targets; CSV/checkpoints remain remote.
All 160 episodes and 120 exact source controls qualify. Net value on both roots is the prerequisite for learning the
allocation action; otherwise diagnose remaining candidate/occupancy authority.
42 focused tests pass, including first-completion replacement, unchanged EWMA,
demand-only dispatch neutrality, native commitments/goals, prior/control/score
qualification and run-specific input staging. Tests use stub environments;
all full native episodes run remotely.

Run `native_transit_calibrated_supply_stage165_frozen_20261011_r2`, revision
`f8059d2ab6`: t142211-t142220 completed on node001/node005/node006.
Calibration changes neither the 20/20 schedules nor the 880/880 selected actions
on either root. All actual cost, wait and reward deltas versus legacy supply
are exactly zero. Versus window forecast, cost still increases by +0.1794995
/ +0.08356945, despite wait falling by 0.81465 / 0.70835 min.

Initial candidate peaks become 12 instead of 5-6, but every first-decision fleet
penalty remains zero. The prior is used at only 63/880 and 62/880 decisions;
753/880 and 730/880 have every candidate point peak at the historical floor.
Fitted individual-trip standard deviations remain 175-193 s. Next isolate the
expectation of the nonlinear fleet penalty under fit-only duration variability,
with the irreversible historical floor and physical objective unchanged.

The r1 batch t142199-t142208 was cancelled while all ten were queued, to replace
historical calibration-directory staging with a new-run input directory. Its
workers never launched; all ten remote Stage164 raw directories remain present.

## Limitations

Frozen two-root development on reused control scenes, not learned upper authority
or independent confirmation. A day-level direction mean misses startup variation,
headway-dependent holding, future shocks and capacity. Occupancy remains a point
estimate, and the historical episode peak is irreversible.
