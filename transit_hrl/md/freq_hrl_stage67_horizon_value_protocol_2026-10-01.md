# Stage67: Finite-Horizon Value Parameterization

Stage65 reward intervals crossed zero; Stage66 found positive last-decile value bias in all32 normalized-MC cases despite a correct remaining-time input. Test one critic intervention before another actor/native trial.

Keep eight development roots, periods50/100, zero_train/joint_ppo, Stage57 calibration/probe paths, gamma, LR, epochs, minibatches, shuffle, value coefficient and gradient clip unchanged. Both actors, upper critic and their Adam states remain frozen. Compare exact Stage64 mc_normalized against mc_factored. Use two reconstruction workers plus one learner (3CPU/3GB), scheduler dynamic placement on node001-node006. No native sampling or forecaster fitting.

Candidate: V=m_gamma(n)*(mu+sigma*f(s,t)), n=round(H*existing_remaining_fraction), m_gamma(n)=sum(gamma**k,k=0..n-1). The terminal multiplier is zero. Normalize MC/m_gamma with fixed mean/std from the first calibration batch only. Start from the same clone hidden weights and empty critic Adam; rebase the scalar head at the full-horizon mass. Consequently initial candidate values taper by m_gamma(n)/m_gamma(H), not identical public values at every time. The intervention changes both initial time shape and regression weighting. Save explicit factored weights/frame/Adam; never export them as ordinary public ValueNet weights.

Full fit gate: every case has EV>=0.10 and lower global MSE, last-decile MSE and absolute last-decile bias than control. Credit gate: candidate mean-gradient cosine is positive in every case; all four equal-root groups have no higher GAE/MC sign disagreement and no lower mean/log_std credit-gradient cosines. Compare pre-update actual normalized GAE against MC-minus-the-same-value with unchanged entropy; no actor optimization. All cases and failures retained. Preflight checks implementation, not effectiveness; both full gates are required before a guarded actor/native trial. No posthoc root/LR/gate sweep.

## Limitations

Finite-horizon archive fit/credit is not reward improvement or frequency superiority. Reused teacher-initialized development roots are not a generalization test; lower and upper causes remain separated. Empirical MC gradients are not true policy gradients. Pull completion markers and compact JSON only; all source traces and critic weights stay on the servers.
