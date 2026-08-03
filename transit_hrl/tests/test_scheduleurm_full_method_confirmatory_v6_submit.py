import argparse
import unittest

from freq_hrl.experiments.trading import full_method_confirmatory_v6 as confirm
from freq_hrl.experiments.trading import full_method_hpo_v6 as hpo
from scripts.submit_full_method_confirmatory_v6_scheduleurm import (
    LINUX_CPU_NODES,
    POOL_CPU_CAPACITY,
    build_scheduler_spec,
    build_training_command,
    cell_relative_dir,
    experiment_cells,
)


def args_fixture() -> argparse.Namespace:
    return argparse.Namespace(
        run_name="unit_confirm_v6",
        frozen_config_sha256="a" * 64,
        frozen_config_b64="e30=",
        heldout_seeds=[31415, 27182],
        nodes=list(LINUX_CPU_NODES),
        python_executable="/compute/python",
        launch_subdir=".",
        project="Freq-HRL-v6-Confirmatory",
        ppo_ram_mb=768,
        offpolicy_ram_mb=1536,
        priority="normal",
        save_checkpoints=False,
        stage_input_paths=[],
        skip_launch_staging=False,
        allow_duplicate=False,
    )


class ScheduleurmFullMethodConfirmatoryV6SubmitTest(unittest.TestCase):
    def test_default_matrix_has_200_independent_training_cells(self):
        cells = experiment_cells(
            list(hpo.ALL_VARIANT_IDS),
            list(confirm.DEFAULT_CONFIRMATORY_REPLICATES),
        )
        self.assertEqual(len(cells), 200)
        self.assertEqual(POOL_CPU_CAPACITY, 1152)

    def test_command_materializes_frozen_data_in_tmp_without_staging_results(self):
        args = args_fixture()
        output = cell_relative_dir(
            "unit_confirm_v6", "freq_hrl_full_v6", 7001
        )
        command = build_training_command(
            args,
            variant_id="freq_hrl_full_v6",
            replicate=7001,
            output_dir=output,
        )
        self.assertIn("base64.b64decode", command)
        self.assertIn("/tmp/freq_hrl_v6_", command)
        self.assertIn("--heldout-seeds 31415 27182", command)
        self.assertNotIn("--save-checkpoint", command)
        self.assertTrue(command.endswith("&& echo DONE"))

    def test_scheduler_uses_dynamic_nodes_and_summary_as_success_artifact(self):
        args = args_fixture()
        for variant, expected_ram in (
            ("freq_hrl_full_v6", 768),
            ("flat_td3_matched_v6", 1536),
        ):
            spec = build_scheduler_spec(
                args, variant_id=variant, replicate=7001
            )
            self.assertEqual(spec["allowed_nodes"], list(LINUX_CPU_NODES))
            self.assertIsNone(spec["require_node"])
            self.assertEqual(spec["cpu"], 1)
            self.assertEqual(spec["ram_mb"], expected_ram)
            self.assertEqual(spec["ckpt_glob"], "cell_summary.json")
            self.assertTrue(spec["reroute_on_node_down"])


if __name__ == "__main__":
    unittest.main()
