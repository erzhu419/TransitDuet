import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts import mujoco_v14_24_paired_intervention_critic_preflight_spec as v14_24
from scripts import mujoco_v14_25_bias_matched_critic_preflight_spec as spec
from scripts.probe_mujoco_action_cost_critic_restoration import (
    actor_output_bias_vector,
    apply_actor_output_bias_delta,
)
from scripts.submit_mujoco_v14_25_bias_matched_critic_preflight_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


class _ToyActor(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Tanh(),
            torch.nn.Linear(4, output_dim),
        )


class _ToyModel:
    def __init__(self):
        self.upper_actor = _ToyActor(2)
        self.lower_actor = _ToyActor(3)


class MujocoV1425BiasMatchedCriticPreflightTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_25_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=[
                "node001", "node002", "node003", "node004", "node005",
                "node006",
            ],
        )

    def test_bias_delta_changes_only_output_biases(self):
        torch.manual_seed(3)
        model = _ToyModel()
        before = actor_output_bias_vector(model)
        weights = [
            actor.net[-1].weight.detach().clone()
            for actor in (model.upper_actor, model.lower_actor)
        ]
        delta = np.linspace(-1e-4, 1e-4, before.size)
        apply_actor_output_bias_delta(model, delta)
        np.testing.assert_allclose(
            actor_output_bias_vector(model), before + delta,
            rtol=2e-5, atol=2e-8,
        )
        for actor, weight in zip(
            (model.upper_actor, model.lower_actor), weights, strict=True
        ):
            torch.testing.assert_close(actor.net[-1].weight, weight)

    def test_frozen_roots_are_fresh_and_bias_steps_are_registered(self):
        roles = (
            spec.CRITIC_TRAIN_ROOTS,
            spec.CRITIC_HOLDOUT_ROOTS,
            spec.DESIGN_ROOTS,
            spec.VALIDATION_ROOTS,
        )
        flattened = [root for role in roles for root in role]
        self.assertEqual(len(flattened), 44)
        self.assertEqual(len(set(flattened)), 44)
        previous = set(
            v14_24.CRITIC_TRAIN_ROOTS + v14_24.CRITIC_HOLDOUT_ROOTS
            + v14_24.DESIGN_ROOTS + v14_24.VALIDATION_ROOTS
        )
        self.assertFalse(previous & set(flattened))
        self.assertEqual(spec.ACTOR_UPDATE_SCOPE, "output_bias")
        self.assertEqual(spec.ACTOR_STEP_RMS_VALUES[-1], 1e-4)

    def test_launcher_freezes_bias_scope_and_dynamic_resources(self):
        self.assertEqual(len(selected_cells()), 3)
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        self.assertIn("--actor-update-scope output_bias", command)
        self.assertIn("--critic-collection-mode paired_output_bias", command)
        self.assertIn(f"--probe-version {spec.PROBE_VERSION}", command)
        scheduler = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], 24)
        self.assertEqual(scheduler["ram_mb"], 16384)
        self.assertEqual(set(scheduler["allowed_nodes"]), set(self._args().nodes))

    def test_launcher_is_directly_executable(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "scripts"
            / "submit_mujoco_v14_25_bias_matched_critic_preflight_scheduleurm.py"
        )
        process = subprocess.run(
            [sys.executable, str(launcher), "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--run-name", process.stdout)


if __name__ == "__main__":
    unittest.main()
