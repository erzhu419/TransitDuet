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
