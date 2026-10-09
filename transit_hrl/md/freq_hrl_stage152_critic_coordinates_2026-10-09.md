# Stage152: Upper Critic Action Coordinates

Stage151 did not establish useful learned dispatch upper control. A server-side
root241 checkpoint probe found 60-second first-layer action contributions of
7.44-10.67 versus 0.034-0.208 for representative absolute-mean state vectors
(about 50-220 times larger). These are diagnostic vectors, not replay states or
proof that input scale caused poor physical control. The new run records actual
causal states, learning statistics and same-state Q curves.

Change only the upper critic action coordinates from physical seconds to
`(action - midpoint) / half_range`. Apply the same transform to replay, actor
and Bellman-target queries. Actor/replay/actuator units remain seconds; physical
bounds, integer release timing, native RE-SAC, regularization, reward and lower
controller are unchanged. Old seconds-coordinate controls remain available.

Cross seconds/unit coordinates with legacy/service-interval credit, all using
signed dispatch. Fresh roots293/307, four methods, 300 train episodes/cell,
30 upper warmup, 2,700 upper and 9,000 lower updates. Each worker qualifies
separately before fresh full training. Last checkpoint only; frozen learned,
zero-upper and zero-holding controls share four scenes in each of five regimes.
Budget: 2,400 train and 480 frozen episodes, 176,774,400 native ticks plus
216,000 qualification ticks. Scheduler uses one CPU/3 GB per cell, all six
compute nodes eligible, no pinning. Only code is staged and small JSON pulled.
The 47 focused tests pass. Four preserved seconds-coordinate optimizer updates
match Stage151 bit-exactly; its complete archived summary is unchanged.
Run `native_transit_critic_units_stage152_development_20261009_r1` is submitted
as scheduler tasks `t140274-t140281`; performance results are pending.
All eight are running on node004/005/006. Sampled root293 workers pass the short
qualification; the seconds-coordinate worker has started fresh full training.

Judge the coordinate contrast and the same-checkpoint upper intervention
separately. A larger action or a better whole-training cost without useful
learned-upper control is not sufficient to expand seeds or claim full HRL.

## Limitations

This is two-root mechanism development, not confirmation. No reward weights,
quantization, critics' regularization or frequency routing are tuned in this run.
Plan curves and learned promotion remain open until useful upper authority.
