import unittest

import torch

from freq_hrl.rl.dual_actor_critic import GaussianActor
from scripts.probe_mujoco_radial_restoration import (
    _parse_gains,
    _parse_router_strengths,
    _v14_17_anchor_profile,
    scale_actor_output_head,
)


class MujocoRadialRestorationProbeTest(unittest.TestCase):
    def test_output_head_scaling_preserves_distribution_scale(self):
        actor = GaussianActor(3, 2, 8, -0.7)
        weight = actor.net[-1].weight.detach().clone()
        bias = actor.net[-1].bias.detach().clone()
        log_std = actor.log_std.detach().clone()
        scale_actor_output_head(actor, 0.97)
        self.assertTrue(torch.allclose(actor.net[-1].weight, 0.97 * weight))
        self.assertTrue(torch.allclose(actor.net[-1].bias, 0.97 * bias))
        self.assertTrue(torch.equal(actor.log_std, log_std))

    def test_gain_registry_is_strict(self):
        self.assertEqual(_parse_gains("1,.99,.95"), (1.0, 0.99, 0.95))
        for invalid in ("", "0", "1.1", "nan", "1,1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _parse_gains(invalid)

    def test_router_strength_registry_includes_closed_interval(self):
        self.assertEqual(
            _parse_router_strengths("0,.5,1"),
            (0.0, 0.5, 1.0),
        )
        for invalid in ("", "-0.1", "1.1", "nan", ".5,.5"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _parse_router_strengths(invalid)

    def test_v14_17_anchor_profile_uses_frozen_crossed_guard(self):
        configured, paths = _v14_17_anchor_profile({
            "environment": "HalfCheetah-v5",
            "lower_action_router_strength": 0.0,
        })
        self.assertEqual(configured["lower_action_router_strength"], 0.5)
        self.assertEqual(
            configured["deployment_frequency_closed_loop_risk_mode"],
            "mode_cvar",
        )
        self.assertEqual(len(paths), 16)
        self.assertEqual(
            {path["disturbance_mode"] for path in paths},
            {"standard", "low_frequency", "high_frequency", "mixed"},
        )
        self.assertEqual(
            len({(path["disturbance_mode"], path["seed"]) for path in paths}),
            len(paths),
        )


if __name__ == "__main__":
    unittest.main()
