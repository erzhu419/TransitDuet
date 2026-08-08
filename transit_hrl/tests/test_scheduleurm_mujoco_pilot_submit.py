import argparse
import unittest

from scripts.submit_mujoco_pilot_scheduleurm import (
    LINUX_CPU_NODES,
    PILOT_OPTIMIZER_SEEDS,
    build_parser,
    build_scheduler_spec,
    build_training_command,
    experiment_cells,
    normalize_args,
)


def args_fixture() -> argparse.Namespace:
    return argparse.Namespace(
        run_name="unit_mujoco_pilot",
        stage="pilot",
        environments=["HalfCheetah-v5"],
        methods=["freq_hrl"],
        training_disturbance_modes=[
            "standard", "low_frequency", "high_frequency", "mixed"
        ],
        evaluation_disturbance_modes=["standard", "ood_chirp"],
        train_seeds=[31013],
        selection_seeds=[32003],
        safety_selection_seeds=[32503],
        eval_seeds=[33013],
        steps=512,
        episode_horizon=1000,
        iterations=64,
        upper_period=16,
        hidden_dim=64,
        learning_rate=3e-4,
        lower_lf_rms_budget=0.05,
        upper_action_scale=0.35,
        lower_action_scale=1.0,
        responsibility_mode="causal_lf_transfer",
        leakage_constraint_scope="joint_behavior",
        upper_hf_rms_budget=0.1,
        upper_hf_penalty_coef=2.0,
        lower_constraint_update_mode="reward_guarded_adam_projection",
        checkpoint_smoothing_window=8,
        checkpoint_min_delta=1e-3,
        checkpoint_evaluation_interval=4,
        nodes=list(LINUX_CPU_NODES),
        python_executable="/compute/python",
        launch_subdir=".",
        project="Freq-HRL-MuJoCo-Pilot",
        ram_mb=2048,
        priority="normal",
        code_revision="a" * 40,
        source_manifest_sha256="b" * 64,
        stage_input_paths=[],
        wait_for_files=["/tmp/preflight/cell_summary.json"],
        skip_launch_staging=False,
        allow_duplicate=False,
    )


class ScheduleurmMujocoPilotSubmitTest(unittest.TestCase):
    def test_pilot_matrix_has_36_dynamic_one_core_cells(self):
        cells = experiment_cells(
            stage="pilot",
            environments=["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"],
            methods=["freq_hrl", "freq_hrl_no_leakage", "generic_hrl", "flat_ppo"],
        )
        self.assertEqual(len(cells), 36)
        self.assertEqual({cell[2] for cell in cells}, set(PILOT_OPTIMIZER_SEEDS))

    def test_preflight_normalization_caps_compute_and_seed_roles(self):
        args = normalize_args(build_parser().parse_args([
            "--run-name", "unit_preflight",
            "--stage", "preflight",
        ]))
        self.assertEqual(args.steps, 64)
        self.assertEqual(args.episode_horizon, 64)
        self.assertEqual(args.iterations, 2)
        self.assertEqual(len(args.train_seeds), 1)
        self.assertEqual(args.training_disturbance_modes, ["standard"])
        self.assertEqual(args.evaluation_disturbance_modes, ["standard"])

    def test_command_is_headless_source_bound_and_multi_condition(self):
        args = args_fixture()
        command = build_training_command(
            args,
            environment="HalfCheetah-v5",
            method="freq_hrl",
            optimizer_seed=35107,
            output_dir="results/unit",
        )
        self.assertIn("MUJOCO_GL=egl", command)
        self.assertIn("--code-revision " + "a" * 40, command)
        self.assertIn(
            "--training-disturbance-modes standard low_frequency "
            "high_frequency mixed",
            command,
        )
        self.assertIn("--evaluation-disturbance-modes standard ood_chirp", command)
        self.assertIn("--safety-selection-seeds 32503", command)
        self.assertIn("--episode-horizon 1000", command)
        self.assertIn("--lower-lf-rms-budget 0.05", command)
        self.assertIn("--upper-action-scale 0.35", command)
        self.assertIn("--lower-action-scale 1.0", command)
        self.assertIn("--responsibility-mode causal_lf_transfer", command)
        self.assertIn("--leakage-constraint-scope joint_behavior", command)
        self.assertIn("--upper-hf-rms-budget 0.1", command)
        self.assertIn("--upper-hf-penalty-coef 2.0", command)
        self.assertIn(
            "--lower-constraint-update-mode reward_guarded_adam_projection",
            command,
        )
        self.assertNotIn("jtl110cpu", command)

    def test_scheduler_uses_unpinned_linux_pool(self):
        args = args_fixture()
        spec = build_scheduler_spec(
            args,
            environment="HalfCheetah-v5",
            method="freq_hrl",
            optimizer_seed=35107,
        )
        self.assertEqual(spec["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertIsNone(spec["require_node"])
        self.assertEqual(spec["cpu"], 1)
        self.assertEqual(spec["ram_mb"], 2048)
        self.assertEqual(
            spec["wait_for_files"], ["/tmp/preflight/cell_summary.json"]
        )


if __name__ == "__main__":
    unittest.main()
