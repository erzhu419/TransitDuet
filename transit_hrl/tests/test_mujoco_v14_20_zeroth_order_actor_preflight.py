import unittest
from argparse import Namespace
from types import SimpleNamespace

import numpy as np

from freq_hrl.rl.dual_actor_critic import GaussianActor
from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as spec
from scripts.probe_mujoco_zeroth_order_actor_restoration import (
    actor_output_head_vector,
    antithetic_directions,
    apply_actor_output_head_delta,
    ranked_antithetic_gradient,
)
from scripts.submit_mujoco_v14_20_zeroth_order_actor_preflight_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


def _snapshot(merit, reward=0):
    return {
        "reward_violation_count": reward,
        "frequency_violation_merit": merit,
        "worst_frequency_violation": merit,
        "frequency_violation_count": 1,
    }


class MujocoV1420ZerothOrderActorPreflightTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_20_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=["node001", "node002", "node003", "node004", "node005", "node006"],
        )

    def test_output_head_delta_is_aligned_and_exact(self):
        model = SimpleNamespace(
            upper_actor=GaussianActor(5, 2, 8, -0.7),
            lower_actor=GaussianActor(5, 2, 8, -0.7),
        )
        before = actor_output_head_vector(model)
        delta = np.linspace(-1e-4, 1e-4, before.size)
        apply_actor_output_head_delta(model, delta)
        np.testing.assert_allclose(
            actor_output_head_vector(model), before + delta, atol=1e-8
        )
        with self.assertRaises(ValueError):
            apply_actor_output_head_delta(model, delta[:-1])

    def test_rank_gradient_points_away_from_worse_antithetic_side(self):
        directions = [np.array([1.0, -1.0]), np.array([1.0, 1.0])]
        pairs = [
            (_snapshot(4.0), _snapshot(1.0)),
            (_snapshot(3.0), _snapshot(2.0)),
        ]
        gradient = ranked_antithetic_gradient(directions, pairs)
        self.assertGreater(float(np.dot(gradient, directions[0])), 0.0)
        self.assertAlmostEqual(float(np.sqrt(np.mean(gradient ** 2))), 1.0)

    def test_launcher_freezes_disjoint_paths_and_dynamic_one_core_cells(self):
        self.assertEqual(len(selected_cells()), 3)
        self.assertFalse(set(spec.DESIGN_ROOTS) & set(spec.VALIDATION_ROOTS))
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        self.assertIn("--direction-count 8", command)
        self.assertIn("--perturb-rms 1e-06", command)
        self.assertTrue(command.endswith("&& echo DONE"))
        scheduler = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], 1)
        self.assertEqual(set(scheduler["allowed_nodes"]), set(self._args().nodes))

    def test_directions_are_reproducible_rademacher_vectors(self):
        first = antithetic_directions(7, 3, spec.DIRECTION_SEED)
        second = antithetic_directions(7, 3, spec.DIRECTION_SEED)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(set(left), {-1.0, 1.0})


if __name__ == "__main__":
    unittest.main()
