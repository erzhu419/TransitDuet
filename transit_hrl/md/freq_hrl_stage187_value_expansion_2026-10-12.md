# Stage187: Fresh Development Scene Coverage

Stage186's active full-timetable encoder still failed the grouped cost/wait
gate. Test the coverage hypothesis with a fixed fresh development cohort,
not further tuning against the same twenty scenes.

Register eight scenes for each of two frozen lower roots and five regimes:
80 scene groups, six probes per scene, 480 states and 2,880 candidate labels.
Scene seed is 910000000 + root*10000 + regime_index*100 + replica.
Reserve all four Stage169 scenes and Stage173 check scenes outside this cohort.

Keep the original greedy selector, frozen lower, 64-draw duration priors,
full timestep inputs, seven paired episodes per probe and physical targets.
Budget: 480 scheduler tasks, 3,360 episodes, 206,236,800 ticks, zero policy
updates. Ten replica0/probe8 tasks cover both roots and all five regimes;
qualify those before submitting the other470. No outcome-based selection.

One CPU/3GB per task, node001-006 unbound. Sync compact result JSON only;
full observations, branch rollouts and native checkpoints remain on servers.
Six focused tests pass. Next: finish paired collection, then compare matched
value fits on expanded scene groups before opening confirmation action labels.

## Limitations

Coverage is a hypothesis, not the established explanation for Stage186's
failure. Six probes within a scene are correlated; there are still only two
frozen lower training roots. Labels describe one forced action followed by
original greedy continuation, not repeated learned control or end-to-end HRL.
