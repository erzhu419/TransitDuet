import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts import mujoco_v14_23_frequency_horizon_critic_preflight_spec as v14_23
from scripts import mujoco_v14_24_paired_intervention_critic_preflight_spec as spec
from scripts.probe_mujoco_action_cost_critic_restoration import (
    apply_actor_output_bias_intervention,
    paired_output_bias_interventions,
)
from scripts.submit_mujoco_v14_24_paired_intervention_critic_preflight_scheduleurm import (
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


class MujocoV1424PairedInterventionCriticPreflightTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_24_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=[
                "node001", "node002", "node003", "node004", "node005",
                "node006",
            ],
        )

    def test_paired_interventions_are_level_isolated_and_antithetic(self):
        model = _ToyModel()
        interventions = paired_output_bias_interventions(
            model, seed=17, bias_rms=0.25
        )
        self.assertEqual(
            sorted(item["variant"] for item in interventions),
            list(spec.CRITIC_INTERVENTION_VARIANTS),
        )
        by_name = {item["variant"]: item for item in interventions}
        np.testing.assert_allclose(
            by_name["upper_plus"]["upper_bias"],
            -by_name["upper_minus"]["upper_bias"],
        )
        np.testing.assert_allclose(
            by_name["lower_plus"]["lower_bias"],
            -by_name["lower_minus"]["lower_bias"],
        )
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.square(by_name["upper_plus"]["upper_bias"])))),
            0.25,
        )
        lower_before = model.lower_actor.net[-1].bias.detach().clone()
        apply_actor_output_bias_intervention(
            model, upper_bias=by_name["upper_plus"]["upper_bias"]
        )
        torch.testing.assert_close(model.lower_actor.net[-1].bias, lower_before)

    def test_frozen_roots_are_fresh_and_expansion_counts_are_exact(self):
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
            v14_23.CRITIC_TRAIN_ROOTS + v14_23.CRITIC_HOLDOUT_ROOTS
            + v14_23.DESIGN_ROOTS + v14_23.VALIDATION_ROOTS
        )
        self.assertFalse(previous & set(flattened))
        self.assertEqual(spec.EXPECTED_CRITIC_TRAIN_PATH_COUNT, 160)
        self.assertEqual(spec.EXPECTED_CRITIC_HOLDOUT_PATH_COUNT, 80)

    def test_launcher_freezes_interventions_and_dynamic_resources(self):
        self.assertEqual(len(selected_cells()), 3)
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        self.assertIn("--critic-collection-mode paired_output_bias", command)
        self.assertIn("--critic-intervention-bias-rms 0.25", command)
        self.assertIn(f"--probe-version {spec.PROBE_VERSION}", command)
        scheduler = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], 24)
        self.assertEqual(scheduler["ram_mb"], 16384)
        self.assertEqual(set(scheduler["allowed_nodes"]), set(self._args().nodes))

    def test_launcher_is_directly_executable(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "scripts"
            / "submit_mujoco_v14_24_paired_intervention_critic_preflight_scheduleurm.py"
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
