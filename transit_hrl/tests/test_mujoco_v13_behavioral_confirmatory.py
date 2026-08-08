import argparse
import unittest

import numpy as np

from scripts import mujoco_v13_behavioral_confirmatory_spec as spec
from scripts import mujoco_v12_confirmatory_spec as v12_spec
from scripts.analyze_mujoco_v13_behavioral_confirmatory import _bootstrap_environment
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_mujoco_v13_behavioral_confirmatory_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    cell_relative_dir,
    experiment_cells,
)


def args_fixture(arm="transfer_full_method"):
    return argparse.Namespace(
        run_name="unit_mujoco_v13_behavioral",
        arm=arm,
        nodes=list(LINUX_CPU_NODES),
        python_executable="/compute/python",
        priority="normal",
        runtime_revision="a" * 40,
        launcher_sha256="b" * 64,
        runtime_sha256="c" * 64,
        spec_sha256="d" * 64,
    )


class MujocoV13BehavioralConfirmatoryTest(unittest.TestCase):
    def test_frozen_design_uses_disjoint_roles_and_24_replicates(self):
        spec.validate_frozen_design()
        self.assertEqual(len(spec.OPTIMIZER_SEEDS), 24)
        roles = [
            set(spec.OPTIMIZER_SEEDS),
            set(spec.TRAIN_SEEDS),
            set(spec.CHECKPOINT_SELECTION_SEEDS),
            set(spec.SAFETY_SELECTION_SEEDS),
            set(spec.HELDOUT_EVALUATION_SEEDS),
        ]
        for index, left in enumerate(roles):
            for right in roles[index + 1:]:
                self.assertFalse(left & right)
        v12_seeds = set().union(
            v12_spec.OPTIMIZER_SEEDS,
            v12_spec.TRAIN_SEEDS,
            v12_spec.CHECKPOINT_SELECTION_SEEDS,
            v12_spec.SAFETY_SELECTION_SEEDS,
            v12_spec.HELDOUT_EVALUATION_SEEDS,
        )
        self.assertFalse(set().union(*roles) & v12_seeds)
        self.assertEqual(spec.PRIMARY_GATE_COUNT, 12)

    def test_each_arm_has_72_unpinned_one_core_cells(self):
        cells = experiment_cells()
        self.assertEqual(len(cells), 72)
        self.assertEqual({cell[0] for cell in cells}, set(spec.ENVIRONMENTS))
        args = args_fixture()
        scheduler_spec = build_scheduler_spec(
            args,
            environment="HalfCheetah-v5",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        self.assertEqual(scheduler_spec["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertIsNone(scheduler_spec["require_node"])
        self.assertEqual(scheduler_spec["cpu"], 1)
        self.assertNotIn("jtl110cpu", str(scheduler_spec))

    def test_full_method_command_is_source_bound_and_seed_frozen(self):
        args = args_fixture()
        output = cell_relative_dir(
            args.run_name,
            arm=args.arm,
            environment="Hopper-v5",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        command = build_training_command(
            args,
            environment="Hopper-v5",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=output,
        )
        self.assertIn("--method freq_hrl_safe_selector", command)
        self.assertIn("--responsibility-mode causal_lf_transfer", command)
        self.assertIn(
            "--code-revision " + spec.FROZEN_ALGORITHM_REVISION,
            command,
        )
        self.assertIn(
            "--source-manifest-sha256 " + spec.FROZEN_SOURCE_MANIFEST_SHA256,
            command,
        )
        self.assertIn(
            "--eval-seeds " + " ".join(map(str, spec.HELDOUT_EVALUATION_SEEDS)),
            command,
        )

    def test_familywise_bootstrap_passes_clear_no_tradeoff_case(self):
        count = len(spec.OPTIMIZER_SEEDS)
        row = _bootstrap_environment(
            environment="synthetic_pass",
            baseline_return=np.linspace(90.0, 110.0, count),
            full_return=np.linspace(100.0, 120.0, count),
            baseline_drift=np.linspace(0.09, 0.11, count),
            full_drift=np.linspace(0.045, 0.055, count),
            baseline_raw_lower_drift=np.linspace(0.18, 0.22, count),
            full_raw_lower_drift=np.linspace(0.08, 0.11, count),
            full_upper_hf_power=np.full(count, 0.05 ** 2),
        )
        self.assertTrue(row["return_noninferiority_pass"])
        self.assertTrue(row["minimum_responsibility_drift_reduction_pass"])
        self.assertTrue(row["minimum_raw_lower_drift_reduction_pass"])
        self.assertTrue(row["upper_hf_budget_pass"])

    def test_familywise_bootstrap_rejects_tradeoff(self):
        count = len(spec.OPTIMIZER_SEEDS)
        row = _bootstrap_environment(
            environment="synthetic_fail",
            baseline_return=np.full(count, 100.0),
            full_return=np.full(count, 90.0),
            baseline_drift=np.full(count, 0.10),
            full_drift=np.full(count, 0.095),
            baseline_raw_lower_drift=np.full(count, 0.20),
            full_raw_lower_drift=np.full(count, 0.19),
            full_upper_hf_power=np.full(count, 0.15 ** 2),
        )
        self.assertFalse(row["return_noninferiority_pass"])
        self.assertFalse(row["minimum_responsibility_drift_reduction_pass"])
        self.assertFalse(row["minimum_raw_lower_drift_reduction_pass"])
        self.assertFalse(row["upper_hf_budget_pass"])


if __name__ == "__main__":
    unittest.main()
