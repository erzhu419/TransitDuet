# Stage170: Joint Duration Calibration Before Policy Selection

Stage169 restores the window primitive but still loses total cost to window on
both roots. Most rolling choices see identical predicted fleet penalties versus
the incumbent. Stage166 samples durations independently across trips, although
Stage164 per-direction adjacent duration errors after causal EWMA have correlation
0.698-0.831 across fit/check episodes. Test joint occupancy modeling, not more
policy weights or seeds.

Use the same five fit scenes, paired directions and stable native trip identities
to construct an empirical path copula. Reorder each trip's original 64 residual
samples by ranks from a common sampled fitting path, with a common tie order.
Every trip's marginal sample values, direction prior and ensemble size remain
exactly unchanged; only their member alignment changes. Local RNG seed170000+root
does not affect native RNG. No check outcome is fitted.

Before deploying this in control, compare expected nonlinear peak penalties with
actual full-episode peaks on all ten independent Stage164 check episodes, using
their actual launches as a shared conditional input. Zero additional native
episodes or training. All 524 native trip marginals reproduce exactly.
Cost MAE worsens from 0.113281 to 0.170833 (root397) and 0.109635 to 0.120313
(root401), with zero improvements in ten check episodes. This fit-only copula
does not improve calibration; do not dispatch it into native control. Next, collect
passive in-flight remaining-time errors, separating distance-progress bias from
duration sampling. Two coupling/source tests and 20 joint/incumbent/duration
focused tests pass. No native policy/training budget was spent on this negative
calibration result.

## Limitations

Known launches are retrospective conditional calibration, not prospective fleet
prediction or counterfactual action value. Five fit paths constrain the empirical
dependence model; unchanged marginals do not fix biased active remaining-time
estimates or duration responses to changed departure plans. This is not learned
upper validation or evidence for complete Freq-HRL.
