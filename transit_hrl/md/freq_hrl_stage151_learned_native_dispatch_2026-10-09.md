# Stage151: Learned Native Dispatch and Credit

Stage150 qualified causal signed departures and preserved HIRO/zero controls,
but fixed advance had mixed outcomes. Now test learned control, not a fixed
departure rule or a reused HIRO actor presented as a channels-trained policy.

Cross HIRO target-headway versus signed departure control with legacy versus
service-interval credit. All four methods retain the physical lower encoder,
16/33 state dimensions, native RE-SAC, estimator/priors, holding reward, action
bounds, fleet limits and leakage penalties. Credit uses wait/fleet weights1,
headway weight0 and scale100, as in Stage148.

New roots241/257/269/281, 300 training episodes each, 30 upper warmup episodes,
2,700 upper and 9,000 lower updates per cell. Last checkpoint only, no warmstart
or selection. Each worker first qualifies two short training episodes plus
three frozen controls, then resets seeds and starts full training from scratch.
Full evaluation pairs four scenes in each of five regimes, at fleet12 and a
fixed 61,380-second clock, with learned/zero-upper/zero-holding deployments.
The sixteen jobs use 4,800 training and 960 frozen episodes (353,548,800 ticks),
plus 432,000 short qualification ticks; one CPU/3 GB each, no node binding.

Compare equal-regime cost, restricted wait, reward and fleet tradeoffs. Measure
actual launch shifts and subsecond commands to distinguish learned authority
from one-second actuator quantization. Test learned upper against its own zero
control; a trained actor alone does not establish useful upper control.
The 37 focused training/dispatch/frontier/authority/diagnostic tests pass.
Run `native_transit_dispatch_training_stage151_development_20261009_r1` is
submitted as `t139086-t139101` through scheduler, dynamically eligible for
node001-006. Only code is staged; checkpoints and native CSV remain server-only.
All sixteen are running on node001/004/005/006. Sampled root241 logs for all
four methods show completed qualification and fresh full training at episode0;
their first full warmup cost is identically 1.333908.

## Limitations

This is four-root descriptive mechanism development. Frequency remains fixed
to correct routing; neither a favorable cost contrast nor encoder sensitivity
alone establishes frequency attribution, full plan curves or learned promotion.
The coupling comparison includes its native decision timing: nominal for HIRO,
120 seconds before nominal for signed dispatch, not an action-only contrast.
Expand routing ablations/independent confirmation only after learned authority.
