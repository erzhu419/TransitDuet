# Stage178: Occupancy-Conditioned Value Learning

Stage176 fails even apparent training action selection. Use Stage177's 60
states/360 labels and preaction occupancy, retaining original native costs.
Fixed 2x2: base/occupancy representation x paired MSE/MSE+soft expected regret.
Primary is occupancy_regret; no post-hoc promotion of a successful control.

All four methods have the same 96-16-2 tanh network, seed178, 1,500 full-batch
AdamW updates per fit, learning rate .003 and weight decay .01. Base controls
zero the 24 added floor, live-trip and committed-departure overlap inputs.
The two heads remain fleet/nonfleet in original physical units. Regret uses
their summed native cost, not episode reward or forecast/proxy-score targets.

Ten scene, five regime and two lower-root held-out fits plus one full fit per
method. Feature centering/scaling and cost RMS are fitted on training only.
Fixed primary must improve mean held-out cost for both roots in all three
splits before a separately registered native screen. Four unpinned scheduler
CPU tasks on node001-006: 72 fits/108,000 critic updates, zero native episodes,
zero actor/lower updates. Pull logs and compact JSON only.
12 focused tests pass: causal feature/mask inputs, decision-gradient direction,
identity, unchanged ridge controls, grouped fitting, serialized inference,
matched parameter counts, primary gate and unique unbound scheduler tasks.

## Limitations

Development screen: ten correlated scenes, two frozen lower roots and one
optimizer initialization. Labels follow one action then the original greedy
continuation. Occupancy overlap is a causal proxy, not a fleet guarantee.
Neither fitting success nor cross-fit gains establish learned closed-loop HRL.
