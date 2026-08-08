import argparse
import unittest

from freq_hrl.experiments.trading import full_method_confirmatory_plan_v732 as plan
from freq_hrl.experiments.trading import full_method_hpo_v7 as hpo
from scripts.submit_full_method_confirmatory_v732_scheduleurm import (
    LINUX_CPU_NODES,
    POOL_CPU_CAPACITY,
    build_scheduler_spec,
    build_training_command,
    cell_relative_dir,
    experiment_cells,
)


def args_fixture() -> argparse.Namespace:
    return argparse.Namespace(
        run_name="unit_confirm_v732",
        frozen_config_sha256="a" * 64,
        frozen_config_b64="e30=",
        heldout_seeds=list(plan.DEFAULT_HELDOUT_SEEDS),
        nodes=list(LINUX_CPU_NODES),
        python_executable="/compute/python",
        launch_subdir=".",
        project="Freq-HRL-v7.3.2-Confirmatory",
        ppo_ram_mb=1024,
        offpolicy_ram_mb=1536,
        priority="normal",
        save_checkpoints=False,
        stage_input_paths=[],
        skip_launch_staging=False,
        allow_duplicate=False,
    )


class ScheduleurmFullMethodConfirmatoryV732SubmitTest(unittest.TestCase):
    def test_registered_matrix_has_288_independent_training_cells(self):
        cells = experiment_cells(
            list(hpo.ALL_VARIANT_IDS),
            list(plan.DEFAULT_CONFIRMATORY_REPLICATES),
        )
        self.assertEqual(len(cells), 288)
        self.assertEqual(POOL_CPU_CAPACITY, 1152)

    def test_command_materializes_frozen_config_and_exact_heldout_registry(self):
        args = args_fixture()
        output = cell_relative_dir(
            args.run_name, "freq_hrl_full_v7", plan.DEFAULT_CONFIRMATORY_REPLICATES[0]
        )
        command = build_training_command(
            args,
            variant_id="freq_hrl_full_v7",
            replicate=plan.DEFAULT_CONFIRMATORY_REPLICATES[0],
            output_dir=output,
        )
        self.assertIn("base64.b64decode", command)
        self.assertIn("/tmp/freq_hrl_v732_", command)
        self.assertIn(
            "--heldout-seeds "
            + " ".join(map(str, plan.DEFAULT_HELDOUT_SEEDS)),
            command,
        )
        self.assertNotIn("--save-checkpoint", command)
        self.assertTrue(command.endswith("&& echo DONE"))

    def test_cells_are_dynamic_one_core_linux_tasks(self):
        args = args_fixture()
        for variant, expected_ram in (
            ("freq_hrl_full_v7", 1024),
            ("flat_td3_matched_v7", 1536),
        ):
            spec = build_scheduler_spec(
                args,
                variant_id=variant,
                replicate=plan.DEFAULT_CONFIRMATORY_REPLICATES[0],
            )
            self.assertEqual(spec["allowed_nodes"], list(LINUX_CPU_NODES))
            self.assertIsNone(spec["require_node"])
            self.assertEqual(spec["cpu"], 1)
            self.assertEqual(spec["ram_mb"], expected_ram)
            self.assertEqual(spec["ckpt_glob"], "cell_summary.json")


if __name__ == "__main__":
    unittest.main()
