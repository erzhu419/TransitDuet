# V106: full-day service tradeoff screen

V105 did not justify additional fitting: A is worse than zero action signal on
all three costs, the frozen teacher lacks joint benefit, passenger replication
is inconclusive and CV replication remains noise-dominated. Pause this
single-anchor selector route. Retain V96 as a development reference.

Compare native B200, frozen V96 preserve, zero holding, and forward/bilateral
headway feedback at gains 0.25/0.5/1. All arms retain learned B200 upper,
fleet, clocks, deployment state and the existing seven feasible bins up to 45 s.
Forward command = gain * (target - observed forward gap). Bilateral command =
gain * (observable follower departure gap - forward gap)/2; use forward feedback
when follower evidence is absent, zero when forward evidence is absent. Clip
and quantize to the nearest feasible bin; ties use the shorter command.

Two archived development actors (12037, 933083), four new days
2061001--2061004. Native/preserve retain three stochastic sampler paths;
deterministic rules/zero use one shared path per actor/day: 104 full-day replays.
Smoke reproduces cached V104 native and V96 preserve, and compares two sampler
seeds for zero and both rule families before admitting this reuse.

Report all cost-versus-actual-CV2 points and paired differences versus native,
preserve and zero, plus evidence coverage. Use existing crossed actor/day
bootstrap after sampler averaging. Joint diagnostic: lower passenger time,
nonworse CV2, readiness and completion versus preserve under the original
strict intervals. No fitted CV margin, interpolation-based victory, gain
selection, training or promotion. This is a small development screen, not
confirmation or proof of frequency separation. Zero retains learned upper;
it is not relabelled as the exact fixed-headway external baseline.

All tests/replays/aggregation via scheduler on node001--node006; only markers
and the final small JSON return locally. If a simple rule achieves comparable
regularity at lower service cost, freeze it for independent confirmation before
redesigning learned credit; otherwise report the measured tradeoff first.

## Run record

Frozen source `f16fb97e77`. Smoke `t183519` passed on node005: six unit
tests and eight replays, including both cached controls and sampler invariance.
Formal tasks `t183520--t183523` cover 0--21/21--44/44--67/67--104 with
7/8/8/13 workers, all gated by the real smoke marker. Aggregate `t183524`
waits for their four actual shard markers. All tasks are restricted to
node001--node006. No model, actor, dual or main-policy update.

## Outcome

All 104 replays and aggregate `t183524` completed with exit 0. Only the
235779-byte final JSON was pulled. None of the six rules passes the original
joint gate. Forward025 versus preserve: journey -1.7456 [-2.9822,-0.7546] min,
CV2 +0.000242 [-0.008725,0.007615], readiness -310.24 [-452.52,-178.89] s.
Bilateral025: journey -1.6914 [-2.8663,-0.7634] min, CV2 +0.001123
[-0.002448,0.005331], readiness -311.17 [-447.26,-179.44] s. Both reduce
passenger time and readiness in both actors and all four day averages, but
CV2 worsens on two day averages and improves on two; nonharm is not confirmed.
They still cost +0.807/+0.861 min and +72.6/+71.7 readiness seconds versus zero,
while substantially improving CV2. No completion failures in any arm.

Forward050 improves CV2 versus preserve (-0.012427 [-0.021420,-0.004717])
but not joint service cost; bilateral050 increases onboard exposure despite
roughly equal vehicle-holding time. Gain 1 improves CV2 but adds +2.95/+5.52
journey minutes and +631/+1116 readiness seconds versus preserve. Bilateral
evidence is actually used (~84% of decisions), not merely forward fallback.
Next V107 freezes BOTH quarter-gain rules for other archived actors/new days;
no "statistically equal CV", gain retuning, main switch or training expansion.
