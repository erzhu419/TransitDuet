# Stage187: Fresh Development Scene Coverage

Stage186 failed the grouped cost/wait gate despite active encoder gradients.
Test scene coverage with eight new seeds per lower root/regime: 80 groups,
480 correlated states, 2,880 labels. Scene seed is
910000000 + root*10000 + regime_index*100 + replica. Exclude all four Stage169
scenes and Stage173 checks; no outcome-based scene selection.

Preserve frozen lower, original greedy continuation, 64 duration draws,
full preaction timetable inputs and physical cost/wait/fleet targets.
Planned budget: 480 tasks, 3,360 episodes, 206,236,800 ticks, zero updates.
One CPU/3GB per task, node001-006 unbound; raw rollouts/checkpoints stay remote.

Run: native_transit_value_expansion_stage187_development_20261012_r2,
code d21b0a715e. Preflight t143421-t143430 completed exit0: 70 episodes,
10 exact identities and60 labels qualified across both roots/five regimes.
Ten compact JSONs total898,030 bytes; eight manual compact-directory syncs.
Remaining470 tasks registered as t143440-t143909 and dispatched.
Eight focused tests pass, including native clock and scheduler retry semantics.

Full r2 completed480 tasks, all exit0. Node-side t144065 qualified all80
fresh groups,480 identities,2,880 labels and3,360 window-ledger episodes.
The merged100-group bank remains server-only for Stage188 matched fits.

The aborted r1 checked `ep_steps` as seconds, but it counts lower transitions.
Remove that incorrect check, retain actual clock/trips/schedule/windows;
archive20 failed attempts and10 stopped retries separately, use none as labels.

Next: finish the480-cell bank; combine with120 existing states for100 scene
groups/600 states/3,600 labels, then matched scalar/sequence grouped value fits.

## Limitations

Coverage is a hypothesis, not the established explanation for Stage186's
failure. Six probes within a scene are correlated; there are still only two
frozen lower training roots. Labels describe one forced action followed by
original greedy continuation, not repeated learned control or end-to-end HRL.
