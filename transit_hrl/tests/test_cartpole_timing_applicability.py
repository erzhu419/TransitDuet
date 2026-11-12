from collections import Counter
import unittest

import numpy as np

from freq_hrl.domains.mujoco.cartpole_setpoint import CartpoleSetpointTask, command_path
from freq_hrl.experiments.compute_budget import ComputeLedger
from scripts.run_cartpole_timing_applicability import CartpoleQualificationController, visible_schedule, rotate_schedule


class CartpoleApplicabilityTest(unittest.TestCase):
    def test_commands_have_persistent_holds_and_nonclock_transitions(self):
        target = command_path(98, 1200)
        changes = np.flatnonzero(np.diff(target)) + 1
        np.testing.assert_array_equal(target, command_path(98, 1200))
        self.assertTrue(np.all(np.diff(np.r_[0, changes]) >= 100))
        self.assertTrue(any(t % 5 for t in changes))

    def test_visible_schedule_is_prefix_causal_and_rotates_the_same_offsets(self):
        target = np.zeros(1201)
        target[53:] = .1
        target[235:] = -.1
        visible = visible_schedule(1200, target.__getitem__)
        self.assertEqual(visible[:6], (0, 55, 125, 175, 225, 250))
        alternative = target.copy()
        alternative[300:] = .3
        self.assertEqual([s for s in visible if s < 300],
                         [s for s in visible_schedule(1200, alternative.__getitem__) if s < 300])
        rotated = rotate_schedule(visible)
        self.assertEqual(Counter(s % 50 for s in visible), Counter(s % 50 for s in rotated))
        self.assertEqual([s // 50 for s in rotated], list(range(24)))

    def test_native_step_clock_lqr_linearization_and_observation(self):
        task = CartpoleSetpointTask(seed=98, horizon=6)
        try:
            obs = task.reset()
            ledger = ComputeLedger()
            with ledger.recording():
                controller = CartpoleQualificationController(task.environment.unwrapped.model, 10.)
            self.assertEqual(ledger.summary()["counts"], {"linearization_native_steps": 10})
            self.assertLess(controller.spectral_radius, 1.)
            self.assertAlmostEqual(task.dt_seconds, .01)
            self.assertEqual(obs.task_measurement.shape, (6,))
            for i in range(6):
                target = obs.target[0]
                obs, _, terminated, truncated, info = task.step(np.zeros(1))
                self.assertAlmostEqual(info["tracking_distance"], abs(obs.achieved_goal[0] - target), places=7)
                self.assertFalse(terminated)
                self.assertEqual(truncated, i == 5)
        finally:
            task.environment.close()
