# Stage148: Native Goal Execution and Physical Credit

Stage147 found negligible upper service authority and very weak lower band
sensitivity. Before any routing/encoder sweep, cross four configurations:
legacy, physical lower, service credit, and both. Routing stays correct; native
RE-SAC, action bounds, network sizes, lower reward, penalties, estimator and
fixed dispatch schedule remain unchanged. All work stays in transit_hrl.

The existing physical lower encoder receives target headway explicitly and
replaces bus ID/mixed units with target, ratios and physical progress at the
same 33-dimensional size. This is a representation change, not scale-only.
Service credit replaces repeated episode/gap rewards with existing additive
global-interval wait/fleet exposure, scale 100. Headway credit has zero weight
because its target is controlled by upper. The stream definition is unchanged.

Four short preflight tasks use root217, two training episodes and five frozen
low-noise interventions each. Only after matching demand, clock, network size,
updates, checkpoint reload and frozen weights qualify, run the two-root
development matrix (217/229; 300 episodes each, no checkpoint selection).
Evaluate five regimes under baseline, neutral upper, fixed upper +/-60 seconds,
and zero holding. This separates weak learned proposals from a lower that
cannot execute changed goals. Native service metrics, not shaped reward alone,
decide whether the upper path has gained useful authority.

Twenty-five focused authority/routing/diagnostic tests passed, including
unchanged legacy configuration, separated factors, explicit target encoding,
target-independent credit, intervention RNG, paired budgets and code-only staging.

Next: collect preflight; then 8 development tasks, 2,400 training episodes and
200 frozen episodes. Keep checkpoints server-only and synchronize small JSON.
Two-root results are descriptive mechanism evidence, not statistical
confirmation or proof that correct frequency routing outperforms controls.
Interval outcomes include other buses and delayed effects; they are physical
temporal credit, not exact counterfactual credit. Fleet exposure differs from
the primary peak-fleet cost, so inspect both wait and peak-fleet tradeoffs.
