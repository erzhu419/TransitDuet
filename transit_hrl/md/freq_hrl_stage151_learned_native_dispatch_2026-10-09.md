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
Run `native_transit_dispatch_training_stage151_development_20261009_r1`,
`t139086-t139101`, is complete. All sixteen cells pass completeness, paired
demand/fleet/clock, parameter-count, update-count and frozen-network checks.
Only compact JSON was pulled; checkpoints and native CSV remain server-only.

## Results

Equal-regime, root-averaged dispatch-minus-HIRO cost is -0.014853 with legacy
credit and -0.003754 with service credit; both have mixed root signs. Legacy
dispatch wait changes by -0.019575 minutes and episode reward by +197.831.
These whole-training contrasts are not evidence of useful learned upper control.

Against its own frozen zero-upper deployment, learned dispatch changes cost by
+0.000545 and wait by +0.006513 minutes. Service-credit dispatch changes them by
+0.000386 and +0.004513 minutes. Legacy dispatch roots257/269 emit entirely
subsecond commands, which truncate to zero and reproduce zero-upper outcomes
exactly. None of the eight learned dispatch deployments advances a trip; the
other roots mostly learn small delays. Service credit does not resolve this.

The lower remains physically useful but has a fleet tradeoff: removing holding
from legacy dispatch increases wait by +0.170450 minutes and headway CV by
+0.139909, while reducing peak fleet by 0.8625 and cost by 0.177280. The primary
cost is therefore not interchangeable with the native lower episode reward.

Next isolate the upper critic's action coordinates: its normalized state is
currently concatenated with raw +/-120-second actions. Keep reward, bounds,
physical execution, regularization and training budget fixed. The LPF window
only defines a diagnostic, not a smoothed actuator or replay-action mismatch.

## Limitations

This is four-root descriptive mechanism development. Frequency remains fixed
to correct routing; neither a favorable cost contrast nor encoder sensitivity
alone establishes frequency attribution, full plan curves or learned promotion.
The coupling comparison includes its native decision timing: nominal for HIRO,
120 seconds before nominal for signed dispatch, not an action-only contrast.
Expand routing ablations/independent confirmation only after learned authority.
