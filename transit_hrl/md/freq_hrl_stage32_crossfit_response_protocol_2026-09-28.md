# Stage-32 Path-Cross-Fitted Plan Response

Freeze roots 209011/209061 (preflight 208001), Stage-26 controller, Stage-30
motion and Stage-31 fit designs/labels only. No query labels enter fitting.
The Stage-31 history train/query rate MSE is 0.06836/0.07944 and 0.05064/0.08374;
this motivates a generalization probe, not a demonstrated nonlinear mechanism.

Retain the exact unit-ridge linear predictor. For every training path, predict
it with a linear fit excluding that entire path. Fit its out-of-fold rate
residual with a centered Gaussian kernel on training-standardized features:
K(z,z')=exp(-||z-z'||^2/(2*d)), d=49 (forecast) or 31 (raw). Unit ridge and
unpenalized residual intercept are fixed. No bandwidth/alpha/window search.
Seven cross-fit views and their seven unchanged linear heads share cached
fit designs and fresh candidate proposals; full-view coefficient counts are
1805/1715 (250/160 linear, 1550 kernel dual, five residual intercepts).

Fresh query bases 3319000/3320000/3321000, offsets 101-108 (101-102 preflight).
Reuse the Stage-31 grid/RNG [root,path,31031], 15 checks/path (one preflight),
full observed prefixes, keep100/settle150 and unchanged full lower feedback.
Primary crossfit_history must beat zero-value and all 13 learned controls
in settled-rate MSE, and all 15 decision controls in mean settled ISE benefit
on both roots. Ties fail; both gates required. Descriptive path-block CI:
4096 resamples, [root,31039]. No reinterpretation of the failed Stage-31 gate.

Full totals: 620 reused fit/240 fresh pairs, 339600 new primitive steps,
240 new/620 reused candidate calls, 238 linear plus 14 kernel solves (1260 RHS).
Preflight: 1330 steps, two new/two reused calls, 21 linear+seven kernel solves.
Dynamic node001-node006, 16 workers/17 CPU/24 GB; preflight 1/2 CPU/3 GB.
34 tests pass. Sync JSON only; arrays and kernel coefficients stay remote.
Implementation `6ff7076cd4`: preflight `t101747` completed on node006;
independent recomputation matched all 1330 steps, held-path fits and metrics.
Both gates failed. Full `t101748/101749` retain the frozen configuration.
Limitations: reused optimizer roots and local paired decisions, not independent
root confirmation, episode reward or planning-cost savings. Stage-9 stays the
performance reference. Next-root confirmation follows only after qualification.
