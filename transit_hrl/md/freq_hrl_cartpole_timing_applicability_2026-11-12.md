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
