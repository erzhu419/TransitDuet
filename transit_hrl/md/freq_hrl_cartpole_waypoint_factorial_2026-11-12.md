# Cart-Pole Waypoint Factorial

[Frozen protocol](../protocols/plan_validity_cartpole_waypoint_factorial_2026-11-12.json).
The preceding qualification changed LQR position weight, fixed offset and
path bank together. On one fresh common bank of 32 paths, cross position
weights 10 and 30 with offsets 5 and 20, keeping the first call at zero.
In every cell run the original current and one-call-old waypoint conditions
with identical disturbances and 24 planner calls. No clock search, feedback
tuning, event-policy rerun or learned policy fitting.

Let D = ISE(lagged) - ISE(current). Compare the gain effect averaged over
clocks, the clock effect averaged over gains, and their difference in
differences on each whole path. Report all four cells, both path halves and
nominal paired 95% t intervals for whole-episode and late-option ISE.
Paths are the paired units; the eight treatments are not eight independent
samples. This diagnostic cannot change the original qualification.

Budget: 256 episodes, 307,200 rollout steps, 6,144 calls and 20
linearization steps. Two scheduler workers, one per LQR configuration.
