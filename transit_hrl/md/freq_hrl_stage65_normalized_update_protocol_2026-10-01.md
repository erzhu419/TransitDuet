# Stage65 Normalized Critic Native Update

Stage64 mc_normalized passed all32 held-out critic fits. Test whether its reward-unit episode GAE improves one guarded lower actor update, with the same eight roots, periods50/100 and zero_train/joint_ppo. gae_raw is the paired same-episode control; MC actor credit is not introduced. Clone actors and their Adam initialize both treatments identically.

Reuse Stage64 critic checkpoints without refitting: load normalized training weights, Adam and the fixed first-calibration frame together; verify exact public-value export and the original held-out probe. Keep Stage63's common upper actor/value/Adam fixed in both treatments and the frozen-lower baseline. zero_train frozen lower reuses the pure clone execution; joint_ppo frozen lower executes the common learned upper, isolating lower learning from upper gains.

Perform one lower clipped-PPO actor update with unchanged LR, entropy, minibatch order, epochs and Stage60 conditional-mean KL0.02/backtracking. Then continue each critic in its existing units for one matched value budget (GAE/raw or MC/normalized). Separate actor/critic loops must reproduce the original raw-unit core exactly in tests; normalized continuation must retain nonempty Adam and its frame. Save training state and public inference weights with distinct labels.

Evaluate16 fresh shared deterministic native paths per full root; root310001 preflight uses2 paths. No new training rollouts, source recalibration, forecaster fitting or checkpoint selection. All64 treatment updates and negative outcomes retained. Twelve predeclared reward contrasts: candidate minus gae_raw, frozen_lower and pure clone in each period/arm. Equal-root paired path means,65536 bootstrap draws, fixed seed(65,65065), two-sided Bonferroni12 intervals. Repair requires all four raw-control contrasts positive; training gain requires all frozen-lower and clone contrasts positive, with nonzero actors and valid KL.

Full cost:256 first-batch archive episodes,307200 lower/4608 upper reconstructions plus307200 extra critic scalar calls;64 critic checkpoint loads,32 common upper loads,64 paired actor/value updates,32 MC continuations and64 candidate checkpoints. Native evaluation1536 episodes/1843200 primitive steps,27648 upper calls,26112 plan OLS/ridge plus matching audits. Reuse zero_train clone rather than charging duplicate execution. Optimizer/forward/guard retries are counted from the frozen source config and actual guard records. Upstream Stage64 MC calibration remains separately recorded in the source cell.

Dynamic scheduler node001-node006;9CPU/6GB full,2CPU/2GB tests/preflight/qualification. Resource-only amendment before native outcomes: preceding full Stage64 tasks used approximately3.9GB; Stage65 uses fewer worker networks, and the single-process DenseTask tests measured423MB. Separate test/qualification/preflight/full resource histories prevent small fixtures from reducing eight-worker RAM estimates. Scientific roster, budgets and endpoints remain unchanged. Only logs and compact JSON pulled. Full launch follows unit/integration tests and mechanical preflight.

## Limitations

One update on reused teacher-initialized development roots does not establish full-training stability, frequency-specific superiority, OOD or equal-FLOPs gains. The common upper is controlled rather than newly optimized; deterministic native evaluation does not establish stochastic deployment gains.

## Execution

Scientific preregistration d5ab6f0fc2 precedes native evaluation; later resource/accounting amendments retain all roots, treatments, endpoints and budgets. Initial fixture comparison errors are retained in t116804/t116810. Four Stage64 regression tests passed in t116810; four new unit/integration tests passed in t116814/node006 (62.141s), including exact raw-core/Adam continuation, serialized normalized Adam recovery, public reward-unit GAE, shared upper and archive-to-native cost accounting. Actual MuJoCo preflight t116815 is registered; full native reward results remain pending.
