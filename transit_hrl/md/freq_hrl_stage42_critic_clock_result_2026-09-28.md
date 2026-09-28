# Stage-42 Critic-Only Control Clock Result

Implementation `d90a461511`; pre-outcome matrix freeze `34c7feb8f1`.
All109 shared-core/focused regression tests pass on scheduler `t103608`.
Tasks `t103621`-`t103660`: all40 original cells/6400 evaluation episodes
complete on node001-node006, without node pinning or duplicate children.
Two launch-only SSH retries add no training. Native audit `t103686` exits zero:
40 trajectory audits,400 snapshot/mode replays,120 probe/warmup/learning
credit replays and16 paired comparisons pass. Independent eight-endpoint
bootstrap passes. The pre-outcome controls, seeds and budgets are unchanged.

| Deterministic return contrast | Mean | Eight-endpoint adjusted CI |
|---|---:|---:|
| Intrinsic-sham vs frozen, final | -4.8028 | [-10.0412, -1.0695] |
| Intrinsic-clock vs frozen, final | -4.5309 | [-8.6497, 0.5181] |
| Task-sham vs frozen, final | -3.9735 | [-7.1095, -0.9667] |
| Task-clock vs frozen, final | -2.0332 | [-5.9263, 3.0198] |
| Intrinsic clock-minus-sham, final | 0.2719 | [-3.4446, 4.8891] |
| Task clock-minus-sham, final | 1.9403 | [-2.0183, 7.0067] |
| Intrinsic clock-minus-sham, first update | -0.8813 | [-3.1621, 0.6385] |
| Task clock-minus-sham, first update | 0.1956 | [-1.3529, 1.7869] |

Both sham arms harm final return; the other six effects are inconclusive.
Clock-only repair is not established. All four learned final means are below
frozen in deterministic and lower-sampled deployment. Frozen returns are
901.4020/900.5171; task-clock returns899.3688/898.0146, respectively.
Initial/warmup deployed actor outcomes equal frozen. First-learning native
states/actions/rewards/durations/done/logp match within both reward pairs.
Warmup critic distances average0.7755 intrinsic/2.8086 task; old-value
RMSE0.007061/0.840489. Mean clock-column norms at warmup/final are
0.6849/1.3213 intrinsic and1.3858/3.4653 task; sham/frozen remain zero.
Every learned arm charges5120 actor/10240 critic optimizer steps; frozen0.

Method cost:20064000 primitive steps/486219 upper/699749 gate calls.
Verification:624000 steps/15234 upper/21953 gate calls, charged separately.
Only [the compact summary](../results/pointmaze_critic_clock_stage42_v1_full_20260928_r1/qualification_summary.json)
is local; raw trajectories and weights remain remote. Deferred zero-exit
scheduler records are closed through the existing terminal handler, not reruns.

## Limitations And Next

Eight reused roots support conditional development, not independent confirmation.
Lower-sampled means are descriptive; no noninferiority endpoint was registered.
Both clocks change together; task reward also removes the intrinsic action cost.
The fixed all-level stochastic probe is not aligned-policy value truth.
Native audit success and nonzero clock weights establish execution, not utility.
Next diagnose whether the first lower PPO/GAE update improves its surrogate
while worsening paired held-out task return. This is an open question, not an
established cause. No new training, seed extension or post-outcome tuning launched.
