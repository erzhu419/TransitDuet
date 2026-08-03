import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.submit_full_method_hpo_scheduleurm import (
    DEFAULT_LINUX_PYTHON,
    LINUX_CPU_NODES,
    POOL_CPU_CAPACITY,
    build_parser,
    build_preflight_spec,
    build_scheduler_spec,
    build_training_command,
    cell_relative_dir,
    cells_without_local_summary,
    experiment_cells,
    normalize_args,
)


def args_fixture() -> argparse.Namespace:
    return argparse.Namespace(
        run_name="unit_full_hpo",
        train_seeds=[42],
        checkpoint_validation_seeds=[57721],
        tuning_validation_seeds=[68207],
        steps=24,
        assets=2,
        iterations=1,
        nodes=list(LINUX_CPU_NODES),
        python_executable=DEFAULT_LINUX_PYTHON,
        launch_subdir=".",
        project="Freq-HRL-Full",
        ppo_ram_mb=512,
        offpolicy_ram_mb=1024,
        priority="normal",
        skip_launch_staging=False,
        stage_input_paths=[],
        allow_duplicate=False,
        code_revision="a" * 40,
        source_manifest_sha256="b" * 64,
    )


class ScheduleurmFullMethodHPOSubmitTest(unittest.TestCase):
    def test_default_pilot_matrix_is_540_independent_cells(self):
        args = normalize_args(build_parser().parse_args([
            "--run-name",
            "unit_full_hpo",
        ]))
        cells = experiment_cells(
            args.variant_ids,
            args.candidate_ids,
            args.scenarios,
            args.optimizer_seeds,
        )
        self.assertEqual(len(cells), 540)
        self.assertEqual(args.nodes, list(LINUX_CPU_NODES))
        self.assertEqual(args.python_executable, DEFAULT_LINUX_PYTHON)
        self.assertEqual(POOL_CPU_CAPACITY, 1152)
        self.assertEqual(args.iterations, 8)
        self.assertEqual(args.steps, 120)

    def test_training_command_contains_only_nested_validation_roles(self):
        args = args_fixture()
        command = build_training_command(
            args,
            variant_id="freq_hrl_full_v4",
            candidate_id="freq_lr1e4_std15_conservative",
            scenario="persistent_shift",
            replicate_seed=2026,
            output_dir=cell_relative_dir(
                "unit_full_hpo",
                "freq_hrl_full_v4",
                "freq_lr1e4_std15_conservative",
                "persistent_shift",
                2026,
            ),
        )
        self.assertIn("--checkpoint-validation-seeds 57721", command)
        self.assertIn("--tuning-validation-seeds 68207", command)
        self.assertNotIn("--eval-seeds", command)
        self.assertNotIn("31415", command)
        self.assertIn("--code-revision " + "a" * 40, command)
        self.assertIn("--source-manifest-sha256 " + "b" * 64, command)
        self.assertIn("CUDA_VISIBLE_DEVICES=", command)
        self.assertTrue(command.endswith("&& echo DONE"))

    def test_cells_use_dynamic_six_node_pool_without_hard_pins(self):
        args = args_fixture()
        ppo_spec = build_scheduler_spec(
            args,
            variant_id="freq_hrl_full_v4",
            candidate_id="freq_lr1e4_std15_conservative",
            scenario="persistent_shift",
            replicate_seed=2026,
        )
        offpolicy_spec = build_scheduler_spec(
            args,
            variant_id="flat_sac_matched_v4",
            candidate_id="off_lr1e4_w1024_b64",
            scenario="persistent_shift",
            replicate_seed=2026,
        )
        for spec in (ppo_spec, offpolicy_spec):
            self.assertEqual(spec["allowed_nodes"], list(LINUX_CPU_NODES))
            self.assertIsNone(spec["require_node"])
            self.assertEqual(spec["cpu"], 1)
            self.assertTrue(spec["allow_cpu_training"])
        self.assertEqual(ppo_spec["ram_mb"], 512)
        self.assertEqual(offpolicy_spec["ram_mb"], 1024)

    def test_preflight_uses_one_singleton_allowed_pool_per_node(self):
        args = args_fixture()
        for node in LINUX_CPU_NODES:
            spec = build_preflight_spec(args, node=node)
            self.assertEqual(spec["allowed_nodes"], [node])
            self.assertIsNone(spec["require_node"])
            self.assertIn("environment.json", spec["ckpt_glob"])
            self.assertIn(DEFAULT_LINUX_PYTHON, str(spec["cmd"]))
            self.assertIn("torch_num_threads", str(spec["cmd"]))

    def test_skip_launch_staging_registers_only_freq_hrl_source(self):
        args = normalize_args(build_parser().parse_args([
            "--run-name",
            "unit_full_hpo",
            "--skip-launch-staging",
        ]))
        self.assertEqual(
            args.stage_input_paths,
            [str((Path(__file__).resolve().parents[1] / "freq_hrl").resolve())],
        )

    def test_missing_cell_filter_skips_materialized_summaries(self):
        cells = [
            (
                "freq_hrl_full_v4",
                "freq_lr1e4_std15_conservative",
                "persistent_shift",
                2026,
            ),
            (
                "freq_hrl_full_v4",
                "freq_lr1e4_std15_conservative",
                "persistent_shift",
                2039,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = root / cell_relative_dir("unit_full_hpo", *cells[0])
            completed.mkdir(parents=True)
            (completed / "cell_summary.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                cells_without_local_summary(
                    cells,
                    run_name="unit_full_hpo",
                    root=root,
                ),
                [cells[1]],
            )


if __name__ == "__main__":
    unittest.main()
