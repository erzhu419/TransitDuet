# TransitDuet evidence and V211 boundary

Read-only parent sources:transit_duet/configs_ablation/H_timetable_v5_stable_300.yaml,
runner_v3.py,env/sim.py,and results_remote/revision3_heldout/summary.json.
Original project untouched. Seven methods,ten training seeds,50 held-out episodes
per seed,pre-specified episode299. Parent aggregate means match its paper:

| Method | Reported CV | Dispatch CV | Wait min | Total passenger min | Positive hold command s |
|---|---:|---:|---:|---:|---:|
| TransitDuet rolling timetable | .467981 | .082105 | 4.87291 | 16.149816 | 13.517652 |
| Fixed-timetable SAC300 | .494422 | .088514 | 5.130396 | 17.102648 | 21.887810 |
| Fixed-timetable SAC360 | .508580 | .054933 | 6.387666 | 19.043096 | 23.875504 |

Positive held-out means are real evidence under THAT protocol,not a FreqDuet
result or a new paired-significance test. The target-headway-only row has slightly
lower CV(.466851) but higher total time(16.804324):no universal metric dominance.

Key differences:parent rolling timetable actually moves next planned launches;
it uses narrowed categorical actions,explicit timetable context and upper holding/
passenger/lateness penalties. Its env.launch_bus creates a vehicle when the terminal
lacks one;_can_launch_trip caps concurrent service at fleet target+buffer,not a
hard12-vehicle reusable pool. Parent passenger/CV accounting differs from V6.
Specifically,env/sim.py:664 computes CV from each vehicle's final forward_headway,
not all station arrivals over the day. Waiting time includes boarded passengers;
total time includes completed passenger journeys. These are real limitations on
interpreting its reported stability/service gains,not V6-equivalent endpoints.
Actual dispatch CV is worse than fixed360,so the positive reported CV row does
not establish improvement in every regularity measure. avg_holding_sec averages
recorded positive commands,not cumulative executed holding per trip;zeros are
not appended in Bus._start_dwelling.
Its fixed-timetable SAC baseline is not V6's strong360s/no-holding fixed baseline.
The named fixed300 row realizes mean dispatch357.44s(planned357.24s),not300s;
its inherited900s baseline-shift cap and recursive actual-launch anchoring remain
active. Treat it as the original bounded rolling-policy baseline,not exact300s.

V211 borrows actual timetable/holding coordination,NOT dynamic vehicle creation,
300s service supply,new holding bins,reward weights or parent outcome numbers.
Retain V6 physical/action/evaluation contracts. V107 bilateral025 has verified
joint improvement versus B200(journey-1.4031min,CV2-.006404,readiness-462.18s),but
CV2 vs preserve is inconclusive;full frequency-separated long-training superiority
and durable strong-fixed improvement remain unestablished. V211 is opportunity
validation before any new learning or deployment decision.

Borrow the executable plan/lower-context coordination and test system costs,
but do not infer each mechanism's contribution from the combined held-out row:
the wider-action/no-context ablations are training-window diagnostics. Current
B200 upper authority is one shared-direction coefficient,not independent controls.
Parent lower rewards still optimize local target/similarity with a headway cost;
global passenger-time penalties enter upper backfill. It is not an existing
solution to the lower action-level system-credit problem diagnosed in V99--V105.
