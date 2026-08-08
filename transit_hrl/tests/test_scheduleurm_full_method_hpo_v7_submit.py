import argparse
import unittest

from freq_hrl.experiments.trading import full_method_hpo_v7 as hpo
from scripts.submit_full_method_hpo_v7_scheduleurm import (
    DEFAULT_LINUX_PYTHON,
    LINUX_CPU_NODES,
    POOL_CPU_CAPACITY,
    build_parser,
    build_preflight_spec,
    build_scheduler_spec,
    build_training_command,
    cell_relative_dir,
    experiment_cells,
    normalize_args,
)


def args_fixture() -> argparse.Namespace:
    return argparse.Namespace(
        run_name="unit_v7_hpo",
        train_seeds=[42],
        promotion_calibration_seeds=[140001],
        checkpoint_validation_seeds=[57721],
        tuning_validation_seeds=[68207],
        steps=24,
        assets=2,
        iterations=1,
        nodes=list(LINUX_CPU_NODES),
        python_executable=DEFAULT_LINUX_PYTHON,
        launch_subdir=".",
        project="Freq-HRL-v7.3.1",
        ppo_ram_mb=768,
        offpolicy_ram_mb=1536,
        priority="normal",
        skip_launch_staging=False,
        stage_input_paths=[],
        allow_duplicate=False,
        code_revision="a" * 40,
        source_manifest_sha256="b" * 64,
    )


class ScheduleurmFullMethodHPOV7SubmitTest(unittest.TestCase):
    def test_default_pilot_has_126_single_checkpoint_cells(self):
        args = normalize_args(build_parser().parse_args([
            "--run-name", "unit_v7_hpo",
        ]))
        cells = experiment_cells(
            args.variant_ids, args.candidate_ids, args.optimizer_seeds
        )
        self.assertEqual(len(cells), 126)
        self.assertEqual(args.nodes, list(LINUX_CPU_NODES))
        self.assertEqual(POOL_CPU_CAPACITY, 1152)
        self.assertEqual(args.iterations, 12)
        self.assertEqual(args.steps, 120)
        self.assertEqual(args.project, "Freq-HRL-v7.3.1")
        self.assertEqual(
            tuple(args.optimizer_seeds), hpo.DEFAULT_PILOT_OPTIMIZER_SEEDS
        )
        self.assertEqual(tuple(args.train_seeds), hpo.DEFAULT_TRAIN_SEEDS)
        self.assertEqual(
            tuple(args.promotion_calibration_seeds),
            hpo.DEFAULT_PROMOTION_CALIBRATION_SEEDS,
        )
        self.assertEqual(
            tuple(args.checkpoint_validation_seeds),
            hpo.DEFAULT_CHECKPOINT_VALIDATION_SEEDS,
        )
        self.assertEqual(
            tuple(args.tuning_validation_seeds), hpo.DEFAULT_TUNING_SEEDS
        )

    def test_final_hpo_uses_optimizer_seeds_disjoint_from_pilot(self):
        args = normalize_args(build_parser().parse_args([
            "--run-name", "unit_v7_hpo_final", "--stage", "final",
        ]))
        self.assertEqual(
            tuple(args.optimizer_seeds), hpo.DEFAULT_FINAL_HPO_OPTIMIZER_SEEDS
        )
        self.assertFalse(
            set(args.optimizer_seeds).intersection(
                hpo.DEFAULT_PILOT_OPTIMIZER_SEEDS
            )
        )

    def test_seed_roles_must_be_disjoint_before_submission(self):
        with self.assertRaisesRegex(SystemExit, "seed roles"):
            normalize_args(build_parser().parse_args([
                "--run-name", "unit_v7_overlap",
                "--train-seeds", "170003",
                "--promotion-calibration-seeds", "170003",
            ]))

    def test_command_has_no_per_scenario_training_axis_or_heldout_seed(self):
        args = args_fixture()
        output = cell_relative_dir(
            "unit_v7_hpo", "freq_hrl_full_v7", "v73_balanced_margin", 2026
        )
        command = build_training_command(
            args,
            variant_id="freq_hrl_full_v7",
            candidate_id="v73_balanced_margin",
            replicate_seed=2026,
            output_dir=output,
        )
        self.assertNotIn("--scenario", command)
        self.assertNotIn("--eval-seeds", command)
        self.assertIn("--checkpoint-validation-seeds 57721", command)
        self.assertIn("--promotion-calibration-seeds 140001", command)
        self.assertIn("--tuning-validation-seeds 68207", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=", command)
        self.assertTrue(command.endswith("&& echo DONE"))

    def test_cells_use_dynamic_six_node_pool_and_one_physical_core(self):
        args = args_fixture()
        for variant_id, candidate_id, expected_ram in (
            ("freq_hrl_full_v7", "v73_balanced_margin", 768),
            ("flat_sac_matched_v7", "off_lr1e4_w1024_b64", 1536),
        ):
            spec = build_scheduler_spec(
                args,
                variant_id=variant_id,
                candidate_id=candidate_id,
                replicate_seed=2026,
            )
            self.assertEqual(spec["allowed_nodes"], list(LINUX_CPU_NODES))
            self.assertIsNone(spec["require_node"])
            self.assertEqual(spec["cpu"], 1)
            self.assertEqual(spec["ram_mb"], expected_ram)
            self.assertTrue(spec["reroute_on_node_down"])

    def test_preflight_targets_each_compute_node_without_login_install(self):
        args = args_fixture()
        for node in LINUX_CPU_NODES:
            spec = build_preflight_spec(args, node=node)
            self.assertEqual(spec["allowed_nodes"], [node])
            self.assertIsNone(spec["require_node"])
            self.assertIn("environment.json", spec["ckpt_glob"])
            self.assertIn(DEFAULT_LINUX_PYTHON, spec["cmd"])
            self.assertNotIn("pip install", spec["cmd"])
            self.assertNotIn("conda install", spec["cmd"])


if __name__ == "__main__":
    unittest.main()
