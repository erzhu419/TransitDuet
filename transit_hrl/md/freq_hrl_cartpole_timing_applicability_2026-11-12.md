# Native Cart-Pole Setpoint Timing Qualification

[Frozen plan](../protocols/plan_validity_cartpole_timing_applicability_2026-11-12.json).
Screen a different plant before fitting any renewal trigger. Use the
[Gymnasium InvertedPendulum-v5](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)
cart/pole dynamics and native termination, with frame skip one and the
integration timestep set to 10ms. This is a derived cart-position tracking
task, not the stock balance score. Visible commands alternate direction with
random 100-180 step dwells and 0.07-0.12m amplitudes, independently of call
clocks. A small 8-step actuator disturbance is present. The pole is stabilized
by fixed model-derived LQR, with two predeclared position weights (10 and 30).
Upper proposals copy the current command; no learned model is fitted.

Use the Stage9 execution clock: 1,200 steps, 24 calls, 50-step bins, optional
checks 0/5/10/15/20 and deadline 25. Select the best constant offset on eight
separate paths per configuration. On 24 new paths per configuration, compare
it with the same clock using the previous waypoint, a visible-command-change
trigger, and a seven-bin rotation of the trigger offsets. The trigger reads
only the current command at each eligible check and the last issued command.
The retrospective rotation matches the offset histogram, not deployability.

Require 2% pooled ISE gain, both path halves positive, and a positive nominal
paired 95% lower bound in each configuration for timely vs lagged waypoint
(episode and latter option halves) and trigger vs both clock and rotation
(episode). Count 345,600 rollout steps plus 20 model-linearization steps.
An incomplete or fallen episode prevents qualification. The two fixed gains
are not trained seeds. A pass qualifies this designed task for a separately
registered Stage9 transfer experiment; it is not cross-task algorithm evidence.
Reacher and all existing HOLD decisions are unchanged.

## Completed Result

[Result](../results/cartpole_timing_applicability_20261112/summary.json) and
[exit records](../results/cartpole_timing_applicability_20261112/completion.json).
t214883-t214884 both exited 0: 345,600 rollout steps, 6,912 planner calls,
20 linearization steps, zero policy fits and 43.05 CPU seconds. Only 5,690
summary bytes were retrieved.

| ISE gain (positive is better) | LQR position weight 10 | LQR position weight 30 |
| --- | ---: | ---: |
| Current vs lagged waypoint, complete episode | -8.30% | +4.41% |
| Current vs lagged waypoint, latter option halves | -8.12% | +4.81% |
| Visible command change vs selection-best clock | +5.13% | +10.03% |
| Visible command change vs rotated offsets | +5.03% | +9.77% |

The event rule passes both timing contrasts in both configurations, including
both path halves. The matched-clock waypoint comparison reverses: its total
gain has nominal paired 95% intervals [-0.02355, -0.00999] and
[0.00139, 0.01989]. In configuration 0 the lagged clock also has lower
episode ISE than the visible rule (0.20192 vs 0.20747) as a descriptive
comparison, not a registered gate. The joint task qualification fails.

Retain both configurations. The gain, selected clock (20 vs 5 steps), and
development path bank vary together; this does not identify why lag helps.
If investigating further, cross both fixed LQR gains with both offsets on one
shared fresh path bank, pairing timely/lagged waypoints. That separates the
gain/offset effects without fitting a trigger or choosing the favorable
configuration. No Stage9 transfer or extra jobs were released here.
