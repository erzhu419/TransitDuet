import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

from scripts import mujoco_v14_27_orthogonal_paired_fd_preflight_spec as v14_27
from scripts import mujoco_v14_28_mechanism_portfolio_preflight_spec as spec
from scripts.probe_mujoco_action_cost_critic_restoration import (
    _unit_interval_floats,
    build_design_fold_contracts,
    fold_guarded_design_eligibility,
)
from scripts.submit_mujoco_v14_28_mechanism_portfolio_preflight_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


def _snapshot(merit: float, *, reward_violations: int = 0):
    return {
        "reward_violation_count": reward_violations,
        "frequency_violation_merit": merit,
        "worst_frequency_violation": merit,
    }


class MujocoV1428MechanismPortfolioPreflightTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_28_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=[
                "node001", "node002", "node003", "node004", "node005",
                "node006",
            ],
        )

    def test_fold_gate_rejects_pooled_only_improvement(self):
        baseline = _snapshot(1.0)
        eligible, flags = fold_guarded_design_eligibility(
            _snapshot(0.8),
            baseline,
            [_snapshot(0.7), _snapshot(1.1)],
            [baseline, baseline],
            minimum_reduction=0.01,
            funnel_multiplier=3.0,
        )
        self.assertFalse(eligible)
        self.assertEqual(flags, [True, False])
        eligible, flags = fold_guarded_design_eligibility(
            _snapshot(0.8),
            baseline,
            [_snapshot(0.8), _snapshot(0.9)],
            [baseline, baseline],
            minimum_reduction=0.01,
            funnel_multiplier=3.0,
        )
        self.assertTrue(eligible)
        self.assertEqual(flags, [True, True])

    def test_each_fold_builds_a_path_matched_snapshot_contract(self):
        rows = [{"path": index} for index in range(4)]
        factory_paths = []

        def factory(baseline_rows):
            expected = [row["path"] for row in baseline_rows]
            factory_paths.append(expected)

            def snapshot(candidate_rows):
                actual = [row["path"] for row in candidate_rows]
                if actual != expected:
                    raise ValueError("path mismatch")
                return _snapshot(1.0)

            return snapshot

        functions, baselines = build_design_fold_contracts(
            rows, [slice(0, 2), slice(2, 4)], factory
        )
        self.assertEqual(factory_paths, [[0, 1], [2, 3]])
        self.assertEqual(len(functions), 2)
        self.assertEqual(len(baselines), 2)
        self.assertEqual(functions[0](rows[:2]), _snapshot(1.0))
        self.assertEqual(functions[1](rows[2:]), _snapshot(1.0))

    def test_router_parser_accepts_zero_and_rejects_duplicates(self):
        self.assertEqual(_unit_interval_floats("0,0.6,1"), (0.0, 0.6, 1.0))
        with self.assertRaises(ValueError):
            _unit_interval_floats("0.6,0.6")

    def test_frozen_roles_are_fresh_and_two_fold_power_is_preserved(self):
        roles = (
            spec.CRITIC_TRAIN_ROOTS,
            spec.CRITIC_HOLDOUT_ROOTS,
            spec.DESIGN_ROOTS,
            spec.VALIDATION_ROOTS,
        )
        flattened = [root for role in roles for root in role]
        self.assertEqual(len(flattened), 96)
        self.assertEqual(len(set(flattened)), 96)
        previous = set(
            v14_27.CRITIC_TRAIN_ROOTS + v14_27.CRITIC_HOLDOUT_ROOTS
            + v14_27.DESIGN_ROOTS + v14_27.VALIDATION_ROOTS
        )
        self.assertFalse(previous & set(flattened))
        self.assertEqual(spec.EXPECTED_DESIGN_FOLD_PATH_COUNT, 64)
        self.assertEqual(spec.EXPECTED_CANDIDATE_COUNT, 15)
        self.assertNotIn(
            spec.BASELINE_ROUTER_STRENGTH, spec.ROUTER_STRENGTH_VALUES
        )

    def test_launcher_freezes_portfolio_and_dynamic_resources(self):
        self.assertEqual(len(selected_cells()), 3)
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        strengths = ",".join(map(str, spec.ROUTER_STRENGTH_VALUES))
        self.assertIn(f"--router-strength-values {strengths}", command)
        self.assertIn("--design-fold-count 2", command)
        self.assertIn(
            "--paired-direction-estimator orthogonal_least_squares", command
        )
        self.assertIn(f"--probe-version {spec.PROBE_VERSION}", command)
        scheduler = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], 24)
        self.assertEqual(scheduler["ram_mb"], 16384)
        self.assertEqual(set(scheduler["allowed_nodes"]), set(self._args().nodes))
        staged = {Path(path).name for path in scheduler["stage_input_paths"]}
        self.assertIn("scripts", staged)
        self.assertIn("freq_hrl", staged)

    def test_launcher_is_directly_executable(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "scripts"
            / "submit_mujoco_v14_28_mechanism_portfolio_preflight_scheduleurm.py"
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
