import argparse
import unittest

from freq_hrl.experiments.trading import full_method_budget_plan_v74 as plan
from scripts.submit_full_method_budget_v74_scheduleurm import (
    LINUX_CPU_NODES,
    POOL_CPU_CAPACITY,
    build_parser,
    build_scheduler_spec,
    build_training_command,
    cell_relative_dir,
    experiment_cells,
    normalize_args,
)


def args_fixture() -> argparse.Namespace:
    return argparse.Namespace(
        run_name="unit_budget_v74",
        train_seeds=[170003],
        promotion_calibration_seeds=[180001],
        checkpoint_validation_seeds=[190027],
        tuning_validation_seeds=[200003],
        steps=120,
        assets=3,
        nodes=list(LINUX_CPU_NODES),
        python_executable="/compute/python",
        launch_subdir=".",
        project="Freq-HRL-v7.4-Budget",
        ppo_ram_mb=1024,
        offpolicy_ram_mb=1536,
        priority="normal",
        code_revision="a" * 40,
        source_manifest_sha256="b" * 64,
        stage_input_paths=[],
        skip_launch_staging=False,
        allow_duplicate=False,
    )


class ScheduleurmFullMethodBudgetV74SubmitTest(unittest.TestCase):
    def test_default_ladder_has_45_one_core_cells(self):
        args = normalize_args(build_parser().parse_args([
            "--run-name", "unit_budget_v74",
        ]))
        self.assertEqual(args.budgets, [192])
        self.assertEqual(len(experiment_cells(args.budgets)), 45)
        self.assertEqual(args.nodes, list(LINUX_CPU_NODES))
        self.assertEqual(POOL_CPU_CAPACITY, 1152)

    def test_command_is_support_only_and_source_bound(self):
        args = args_fixture()
        output = cell_relative_dir(
            args.run_name,
            192,
            "freq_hrl_full_v7",
            "v73_balanced_margin",
            7207,
        )
        command = build_training_command(
            args,
            budget=192,
            variant_id="freq_hrl_full_v7",
            candidate_id="v73_balanced_margin",
            replicate_seed=7207,
            output_dir=output,
        )
        self.assertIn("--iterations 192", command)
        self.assertIn("--code-revision " + "a" * 40, command)
        self.assertIn("--source-manifest-sha256 " + "b" * 64, command)
        self.assertNotIn("ood_period", command)
        self.assertNotIn("promotion_recovery", command)
        self.assertNotIn("jtl110cpu", command)

    def test_scheduler_uses_dynamic_six_node_pool(self):
        args = args_fixture()
        spec = build_scheduler_spec(
            args,
            budget=192,
            variant_id="flat_td3_matched_v7",
            candidate_id="off_lr1e3_w4096_b64",
            replicate_seed=7207,
        )
        self.assertEqual(spec["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertIsNone(spec["require_node"])
        self.assertEqual(spec["cpu"], 1)
        self.assertEqual(spec["ram_mb"], 1536)
        self.assertTrue(spec["reroute_on_node_down"])

    def test_windows_nodes_are_rejected(self):
        with self.assertRaisesRegex(SystemExit, "invalid budget nodes"):
            normalize_args(build_parser().parse_args([
                "--run-name", "unit_budget_bad_node",
                "--nodes", "jtl110cpu",
            ]))


if __name__ == "__main__":
    unittest.main()
