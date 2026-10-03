# Stage102: Separate-Credit Joint Mean Learning

Stage99 supports lower credit conditioned on common upper innovations; Stage101 supports upper refinement with a frozen lower. Neither proves the integrated joint learner. Test both actor means from original Stage96 U0/L0, without Stage98-101 trained donors.

- Two arms: joint-independent and joint-conditioned. Both use separate upper/lower scenario roles, 64 on-policy paths per actor/update, eight simultaneous updates and fixed-final evaluation32; preflight uses eight paths per actor, two updates and evaluation4. Collect both actors' credit before either actor changes.
- Upper pairs always have independent upper/lower noise. Only the conditioned arm's lower pairs share upper innovations, retaining independent lower noise and causal upper feedback. Never use these action-dependent lower baselines for the upper gradient. Evaluation always uses the original independent noise mapping.
- Both arms have the same native episodes, samples per actor and decision-call-weighted nominal KL0.001/update: upper0.0005, lower0.001-0.0005/period. Extra lower upper-replay forwards are recorded separately, not called equal total compute.
- Keep both stds, values, source Adam, forecaster, decoder and original teachers fixed. Only actor means learn; no fitted critic, checkpoint choice, radius tuning or frequency-superiority claim.
- Retain both complete joint policies, both crossed-actor compositions, base and zero. All20 reward contrasts use equal-root bootstrap65,536 / Bonferroni20. Require conditioned-minus-independent and conditioned-minus-original-base corrected CI lower bounds to be positive at both periods. Crossed-actor diagnostics cannot replace these four primary endpoints.
- Fresh Stage102 preflight/full/actor/noise/evaluation roles; no upstream or cross-stage pooling. Mechanical preflight, not reward, admits the full eight-root trial. Dynamic node001-006, completion/compact JSON only; checkpoints remain server-only.
- Native budget: preflight176 episodes / 52,800 steps, 16 actor-mean updates and 72 extra upper-replay forwards; full eight-root cohort35,840 episodes / 43,008,000 steps, 512 actor-mean updates, 73,728 extra upper-replay forwards and 32 final checkpoint writes. Preflight3 CPU / 3 GB; full9 CPU / 8 GB; qualifier1 CPU / 2 GB.

This is joint MC mean-learning integration on the same native task. Stage101's refinement result and failed matched-upper specialization, Stage100 negatives and Stage67 critic HOLD remain unchanged.
