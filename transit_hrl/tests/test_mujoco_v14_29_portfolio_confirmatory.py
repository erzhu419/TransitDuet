from argparse import Namespace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from freq_hrl.experiments.reproducibility import (
    current_freq_hrl_source_manifest_sha256,
)
from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as v14_20
from scripts import mujoco_v14_21_distributional_actor_preflight_spec as v14_21
from scripts import mujoco_v14_22_action_cost_critic_preflight_spec as v14_22
from scripts import mujoco_v14_23_frequency_horizon_critic_preflight_spec as v14_23
from scripts import mujoco_v14_24_paired_intervention_critic_preflight_spec as v14_24
from scripts import mujoco_v14_25_bias_matched_critic_preflight_spec as v14_25
from scripts import mujoco_v14_26_robust_paired_fd_preflight_spec as v14_26
from scripts import mujoco_v14_27_orthogonal_paired_fd_preflight_spec as v14_27
from scripts import mujoco_v14_28_mechanism_portfolio_preflight_spec as v14_28
from scripts import mujoco_v14_29_fresh_anchor_spec as anchor_spec
from scripts import mujoco_v14_29_portfolio_confirmatory_spec as spec
from scripts import (
    submit_mujoco_v14_29_portfolio_confirmatory_scheduleurm
    as portfolio_launcher,
)
from scripts.analyze_mujoco_v14_29_fresh_anchors import _qualify_anchor
from scripts.analyze_mujoco_v14_29_portfolio_confirmatory import wilson_interval
from scripts.submit_mujoco_v14_29_fresh_anchors_scheduleurm import (
    build_parser as build_anchor_parser,
    build_scheduler_spec as build_anchor_scheduler_spec,
    normalize_args as normalize_anchor_args,
    selected_experiment_cells,
)
from scripts.submit_mujoco_v14_29_portfolio_confirmatory_scheduleurm import (
    build_parser as build_portfolio_parser,
    build_scheduler_spec as build_portfolio_scheduler_spec,
    selected_cells,
)


class MujocoV1429PortfolioConfirmatoryTest(unittest.TestCase):
    def _anchor_args(self):
        return normalize_anchor_args(build_anchor_parser().parse_args([
            "--run-name", "anchor-test",
            "--phases", "anchor",
            "--nodes", "node001,node002,node003,node004,node005,node006",
        ]))

    def _portfolio_args(self):
        return Namespace(
            run_name="portfolio-test",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=[
                "node001", "node002", "node003",
                "node004", "node005", "node006",
            ],
        )

    def test_frozen_source_and_all_seed_roles_are_fresh(self):
        self.assertEqual(
            current_freq_hrl_source_manifest_sha256(),
            anchor_spec.FROZEN_SOURCE_MANIFEST_SHA256,
        )
        anchor_roles = set(
            anchor_spec.OPTIMIZER_SEEDS
            + anchor_spec.PRETRAIN_SEEDS
            + anchor_spec.PRETRAIN_SELECTION_SEEDS
            + anchor_spec.DEVELOPMENT_EVALUATION_SEEDS
        )
        portfolio_roles = set(
            spec.CRITIC_TRAIN_ROOTS
            + spec.CRITIC_HOLDOUT_ROOTS
            + spec.DESIGN_ROOTS
            + spec.VALIDATION_ROOTS
        )
        self.assertFalse(anchor_roles & portfolio_roles)
        previous = set()
        for module in (
            v14_20, v14_21, v14_22, v14_23, v14_24,
            v14_25, v14_26, v14_27, v14_28,
        ):
            for role in (
                "CRITIC_TRAIN_ROOTS", "CRITIC_HOLDOUT_ROOTS",
                "DESIGN_ROOTS", "VALIDATION_ROOTS",
            ):
                previous.update(getattr(module, role, ()))
        self.assertFalse(previous & portfolio_roles)
        self.assertEqual(len(anchor_spec.OPTIMIZER_SEEDS), 16)
        self.assertEqual(spec.EXPECTED_CELL_COUNT, 48)

    def test_anchor_launcher_is_anchor_only_and_dynamically_placed(self):
        args = self._anchor_args()
        cells = selected_experiment_cells(args)
        self.assertEqual(len(cells), 48)
        self.assertEqual({cell[0] for cell in cells}, {"anchor"})
        phase, environment, arm, seed = cells[0]
        scheduler = build_anchor_scheduler_spec(
            args,
            phase=phase,
            environment=environment,
            arm=arm,
            optimizer_seed=seed,
        )
        self.assertEqual(scheduler["cpu"], 1)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["wait_for_files"], [])
        self.assertIn("--lower-action-router-strength 0.5", scheduler["cmd"])
        self.assertIn(
            f"--code-revision {anchor_spec.FROZEN_ALGORITHM_REVISION}",
            scheduler["cmd"],
        )
        self.assertIn(
            f"--source-manifest-sha256 {anchor_spec.FROZEN_SOURCE_MANIFEST_SHA256}",
            scheduler["cmd"],
        )
        staged = {Path(path).name for path in scheduler["stage_input_paths"]}
        self.assertIn("scripts", staged)
        self.assertIn("freq_hrl", staged)
        with self.assertRaises(SystemExit):
            normalize_anchor_args(build_anchor_parser().parse_args([
                "--run-name", "bad",
                "--phases", "continuation",
            ]))

    def test_anchor_qualification_rejects_seed_protocol_drift(self):
        checkpoint = b"v14.29-anchor-checkpoint"
        summary = {
            "environment": spec.ENVIRONMENTS[0],
            "optimizer_seed": anchor_spec.OPTIMIZER_SEEDS[0],
            "method": "freq_hrl",
            "code_revision": anchor_spec.FROZEN_ALGORITHM_REVISION,
            "source_manifest_sha256": (
                anchor_spec.FROZEN_SOURCE_MANIFEST_SHA256
            ),
            "lower_action_router_mode": "causal_joint_band_projection",
            "lower_action_router_strength": 0.5,
            "lower_action_router_function_preserving": True,
            "checkpoint_selection_mode": anchor_spec.CHECKPOINT_SELECTION_MODE,
            "iterations": anchor_spec.PRETRAIN_ITERATIONS,
            "rollout_seed_roots": list(anchor_spec.PRETRAIN_SEEDS),
            "checkpoint_selection_seed_roots": list(
                anchor_spec.PRETRAIN_SELECTION_SEEDS
            ),
            "eval_seeds": list(anchor_spec.DEVELOPMENT_EVALUATION_SEEDS),
            "training_disturbance_modes": list(
                anchor_spec.TRAINING_DISTURBANCE_MODES
            ),
            "evaluation_disturbance_modes": list(
                anchor_spec.EVALUATION_DISTURBANCE_MODES
            ),
            "steps": anchor_spec.STEPS,
            "upper_period": anchor_spec.UPPER_PERIOD,
            "lower_action_router_training_schedule": "constant",
            "source_identity_status": "verified",
            "checkpoint_file_sha256": hashlib.sha256(checkpoint).hexdigest(),
            "frozen_parameter_sha256": "a" * 64,
            "checkpoint_has_eligible_selection": True,
            "checkpoint_selection_score": 1.0,
            "selected_checkpoint_iteration": (
                anchor_spec.PRETRAIN_CHECKPOINT_MINIMUM_ITERATION
            ),
            "evaluation_row_count": (
                len(anchor_spec.DEVELOPMENT_EVALUATION_SEEDS)
                * len(anchor_spec.EVALUATION_DISTURBANCE_MODES)
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "checkpoint.pt").write_bytes(checkpoint)
            (path / "cell_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            row = _qualify_anchor(
                path, spec.ENVIRONMENTS[0], anchor_spec.OPTIMIZER_SEEDS[0]
            )
            self.assertTrue(row["qualified"])
            summary["eval_seeds"] = list(reversed(summary["eval_seeds"]))
            (path / "cell_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                _qualify_anchor(
                    path,
                    spec.ENVIRONMENTS[0],
                    anchor_spec.OPTIMIZER_SEEDS[0],
                )

    def test_portfolio_launcher_freezes_full_dynamic_matrix(self):
        args = self._portfolio_args()
        parser = build_portfolio_parser()
        self.assertIn(spec.DEVELOPMENT_PROTOCOL_VERSION, parser.description)
        self.assertNotIn("v14.18", parser.description)
        cells = selected_cells()
        self.assertEqual(len(cells), 48)
        environment, seed = cells[0]
        scheduler = build_portfolio_scheduler_spec(args, environment, seed)
        self.assertEqual(scheduler["cpu"], 24)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(
            set(scheduler["allowed_nodes"]),
            {"node001", "node002", "node003", "node004", "node005", "node006"},
        )
        staged = {Path(path).name for path in scheduler["stage_input_paths"]}
        self.assertIn("scripts", staged)
        self.assertIn("freq_hrl", staged)
        self.assertIn(f"--probe-version {spec.PROBE_VERSION}", scheduler["cmd"])

    def test_portfolio_launcher_requires_all_anchors_to_qualify(self):
        args = self._portfolio_args()
        with (
            mock.patch.object(
                portfolio_launcher, "BASE_NORMALIZE_ARGS", return_value=args
            ),
            mock.patch.object(
                portfolio_launcher,
                "analyze_anchor_run",
                return_value={
                    "status": "fresh_anchor_bank_qualified",
                    "qualified_anchor_count": spec.EXPECTED_CELL_COUNT,
                },
            ),
        ):
            self.assertIs(portfolio_launcher._normalize_args(args), args)
        with (
            mock.patch.object(
                portfolio_launcher, "BASE_NORMALIZE_ARGS", return_value=args
            ),
            mock.patch.object(
                portfolio_launcher,
                "analyze_anchor_run",
                return_value={
                    "status": "fresh_anchor_bank_not_qualified",
                    "qualified_anchor_count": spec.EXPECTED_CELL_COUNT - 1,
                },
            ),
            self.assertRaisesRegex(SystemExit, "all 48 fresh anchors"),
        ):
            portfolio_launcher._normalize_args(args)
        wrong_run = self._portfolio_args()
        wrong_run.anchor_run_name = "post-outcome-anchor-substitution"
        with (
            mock.patch.object(
                portfolio_launcher,
                "BASE_NORMALIZE_ARGS",
                return_value=wrong_run,
            ),
            self.assertRaisesRegex(SystemExit, "anchor run is frozen"),
        ):
            portfolio_launcher._normalize_args(wrong_run)

    def test_launchers_are_directly_executable_with_correct_identity(self):
        root = Path(__file__).resolve().parents[1]
        launchers = (
            (
                "submit_mujoco_v14_29_fresh_anchors_scheduleurm.py",
                anchor_spec.DEVELOPMENT_PROTOCOL_VERSION,
            ),
            (
                "submit_mujoco_v14_29_portfolio_confirmatory_scheduleurm.py",
                spec.DEVELOPMENT_PROTOCOL_VERSION,
            ),
        )
        for filename, protocol in launchers:
            process = subprocess.run(
                [sys.executable, str(root / "scripts" / filename), "--help"],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn(protocol, process.stdout)
            self.assertNotIn("v14.18", process.stdout)

    def test_wilson_gate_requires_twelve_of_sixteen(self):
        lower_12, _ = wilson_interval(12, 16)
        lower_11, _ = wilson_interval(11, 16)
        self.assertGreater(lower_12, spec.SUCCESS_RATE_NULL)
        self.assertLess(lower_11, spec.SUCCESS_RATE_NULL)


if __name__ == "__main__":
    unittest.main()
