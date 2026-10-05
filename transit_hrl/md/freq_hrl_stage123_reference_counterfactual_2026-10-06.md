# Stage123: Native Reference Counterfactual Credit

Stage122 removed time-to-go variance but did not stabilize period100 upper
directions or consistently improve lower directions. Do not adopt LOO training
on that evidence. Instead test full-suffix action credit on the exact Stage121
reference channel, with the strong forecast teacher and both policies frozen.

First two registered roots, both periods50/100. Two fresh scenarios at each of
steps0/300/600/900, horizon1200. At each causal query state perturb one option's
eight Bernstein coordinates by+/-0.005, then return to the frozen zero-mean
upper. The lower teacher still receives forecast advice, not the perturbed
curve; donor response limit0.05 and lower noise remain unchanged. Include the
full recovery tail, not only the manipulated option's return.

Prefix noise is identical across both suffix-noise panels. Coordinate
contrasts share innovations within each panel; compare native gradient A/B
and its pullback through the causal upper-state mean. Test A's normalized
direction in B and B's in A, both signs, with the same0.005 action-vector norm.
Selection labels come from training counterfactuals; they are not deployable
policy information. All prefix states/actions/rewards and external signals
must match. Zero intervention is tested against the Stage121 forecast source.

Each query costs38full episodes:34coordinate paths plus4crossfit paths.
Per root:16queries,608episodes/729,600steps. Both roots:1,216episodes/
1,459,200steps. Four workers+parent/8GiB, dynamic scheduler node001-node006.
No optimizer steps, policy adoption, checkpoint or native trace files.

This diagnoses whether native upper credit is usable; it does not prove a
learned plan beats forecast, or close Stage121/promotion/domain-general claims.
If the paired native slopes and crossfit gains are credible at both periods,
the next step is a bounded causal actor update and fresh rollout validation,
not immediate full-scale joint training.

Five targeted tests passed: zero reproduces Stage121 forecast, common prefix
and innovations, full-suffix identities, causal-state pullback, and actual
episode/call budgets with zero policy adoption. No existing training code changed.
