"""Focused tests for the preregistered V33 timescale-stability screen."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.audit_protocol_v6_v33_timescale_screen import (
    CANDIDATES,
    CONFIRMATION_EVAL_SEEDS,
    CONFIRMATION_TRAIN_SEEDS,
    CURRENT_MAIN,
    DISCOVERY_EVAL_SEEDS,
    DISCOVERY_TRAIN_SEEDS,
    EXPECTED_CONFIGS,
    confirmation_configs,
    evaluate_timescale_confirmation,
    evaluate_timescale_screen,
    validate_candidate_contracts,
)
from scripts.submit_freqduet_protocol_v6_v33_confirmation_scheduleurm import (
    build_submit_command as build_confirmation_submit_command,
)
from scripts.submit_freqduet_protocol_v6_v33_timescale_scheduleurm import (
    DEFAULT_NODES,
    build_submit_command as build_development_submit_command,
)
from scripts.validate_freqduet_protocol_v6_configs import validate


class ProtocolV6V33TimescaleTest(unittest.TestCase):
    def make_artifacts(
        self,
        root: Path,
        *,
        directory: str = "aggregate",
        configs: list[str] | None = None,
        train_seeds: list[int] | None = None,
        eval_seeds: list[int] | None = None,
        stage: str = "exploratory",
        source_sha: str = "c" * 64,
        inconsistent_direction: bool = False,
    ) -> Path:
        aggregate = root / directory
        aggregate.mkdir()
        configs = list(configs or EXPECTED_CONFIGS)
        train_seeds = list(train_seeds or DISCOVERY_TRAIN_SEEDS)
        eval_seeds = list(eval_seeds or DISCOVERY_EVAL_SEEDS)
        expected_rollouts = (
            len(configs) * len(train_seeds) * len(eval_seeds)
        )
        (aggregate / "matrix_manifest.json").write_text(json.dumps({
            "strict_complete": True,
            "common_random_numbers_verified": True,
            "run_manifests_verified": True,
            "stage": stage,
            "independent_confirmation": stage == "confirmation",
            "configs": configs,
            "train_seeds": train_seeds,
            "eval_seeds": eval_seeds,
            "train_episodes": 200,
            "checkpoint_ep": 199,
            "expected_rollouts": expected_rollouts,
            "reference": "F_freqduet_protocol_v6_noguard_hiro",
            "run_source_fingerprint": {"sha256": source_sha},
            "scenario_contract": {"sha256": "d" * 64},
            "launch_analysis_sha256": "e" * 64,
            "run_git_provenance": {
                "commit": "a" * 40,
                "tracked_dirty": False,
            },
        }))
        rows = []
        selected = CANDIDATES[0]
        for config in configs:
            for train_seed in train_seeds:
                for eval_seed in eval_seeds:
                    value = 0.22
                    if config == selected:
                        value = 0.18
                        if inconsistent_direction and train_seed != (
                                train_seeds[0]):
                            value = 0.24
                    rows.append({
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "headway_cv": value,
                    })
        pd.DataFrame(rows).to_csv(
            aggregate / "frozen_per_eval.csv", index=False)
        return aggregate

    @staticmethod
    def selection_result(candidate: str, *, passes: bool) -> dict[str, object]:
        return {
            "status": "unique_pass" if passes else "no_pass",
            "candidate_results": [{
                "candidate": candidate,
                "headway_cv_delta_ci_high": -0.005 if passes else 0.005,
                "journey_delta_ci_high": 0.10,
            }],
        }

    def fake_selection(self, *args, **kwargs):
        candidate = kwargs["candidates"][0]
        return self.selection_result(
            candidate,
            passes=(candidate == CANDIDATES[0]),
        )

    def test_configs_change_only_role_and_registered_freeze_schedule(self):
        checks = validate_candidate_contracts()
        self.assertTrue(all(checks.values()))
        result = validate(EXPECTED_CONFIGS, allow_experimental=True)
        self.assertEqual(result["status"], "valid")

    def test_development_and_confirmation_rosters_are_disjoint(self):
        self.assertTrue(set(DISCOVERY_TRAIN_SEEDS).isdisjoint(
            CONFIRMATION_TRAIN_SEEDS))
        self.assertTrue(set(DISCOVERY_EVAL_SEEDS).isdisjoint(
            CONFIRMATION_EVAL_SEEDS))
        self.assertTrue(set(CONFIRMATION_TRAIN_SEEDS).isdisjoint(
            CONFIRMATION_EVAL_SEEDS))

    def test_gate_selects_first_registered_passing_candidate(self):
        with TemporaryDirectory() as tmp:
            aggregate = self.make_artifacts(Path(tmp))
            with patch(
                "scripts.audit_protocol_v6_v33_timescale_screen."
                "evaluate_selection",
                side_effect=self.fake_selection,
            ):
                result = evaluate_timescale_screen(aggregate)
        self.assertEqual(result["status"], "development_candidate_selected")
        self.assertEqual(result["selected_for_confirmation"], CANDIDATES[0])
        self.assertTrue(result["confirmation_authorized"])
        self.assertFalse(result["claim_eligible"])

    def test_direction_failure_blocks_confirmation(self):
        with TemporaryDirectory() as tmp:
            aggregate = self.make_artifacts(
                Path(tmp), inconsistent_direction=True)
            with patch(
                "scripts.audit_protocol_v6_v33_timescale_screen."
                "evaluate_selection",
                side_effect=self.fake_selection,
            ):
                result = evaluate_timescale_screen(aggregate)
        selected = result["candidate_results"][0]
        self.assertFalse(selected["passes"])
        self.assertFalse(selected["longtrain_gates"][
            "headway_cv_direction_consistent"])
        self.assertEqual(result["status"], "no_pass")
        self.assertFalse(result["confirmation_authorized"])

    def test_submit_command_uses_only_hpc_matrix_entrypoint(self):
        command = build_development_submit_command(
            commit="b" * 40,
            nodes=DEFAULT_NODES,
            dispatch=True,
            dry_run=True,
        )
        rendered = " ".join(command)
        self.assertIn("submit_freqduet_protocol_v2_scheduleurm.py", rendered)
        self.assertIn("--train-episodes 200", rendered)
        self.assertIn("--workers 8", rendered)
        self.assertIn("--result-sync summary", rendered)
        self.assertIn("--allow-experimental-configs", command)
        self.assertIn("--dispatch", command)
        self.assertIn("--dry-run", command)
        self.assertIn(CURRENT_MAIN, rendered)

    def test_confirmation_requires_and_passes_frozen_lineage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            development = self.make_artifacts(
                root, directory="development")
            with patch(
                "scripts.audit_protocol_v6_v33_timescale_screen."
                "evaluate_selection",
                side_effect=self.fake_selection,
            ):
                gate = evaluate_timescale_screen(development)
            gate_path = development / "v33_timescale_gate.json"
            gate_path.write_text(json.dumps(gate))
            selected = gate["selected_for_confirmation"]
            confirmation = self.make_artifacts(
                root,
                directory="confirmation",
                configs=confirmation_configs(selected),
                train_seeds=CONFIRMATION_TRAIN_SEEDS,
                eval_seeds=CONFIRMATION_EVAL_SEEDS,
                stage="confirmation",
            )
            with patch(
                "scripts.audit_protocol_v6_v33_timescale_screen."
                "evaluate_selection",
                side_effect=self.fake_selection,
            ):
                result = evaluate_timescale_confirmation(
                    confirmation,
                    development_dir=development,
                )
            command = build_confirmation_submit_command(
                commit="a" * 40,
                gate_path=gate_path,
                nodes=DEFAULT_NODES,
                dispatch=True,
                dry_run=True,
            )
        self.assertEqual(result["status"], "timescale_stability_confirmed")
        self.assertTrue(result["confirmation_claim_eligible"])
        rendered = " ".join(command)
        self.assertIn("--stage confirmation", rendered)
        self.assertIn("--v33-development-gate", command)
        self.assertIn(selected, rendered)

    def test_confirmation_rejects_changed_model_source(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            development = self.make_artifacts(
                root, directory="development")
            with patch(
                "scripts.audit_protocol_v6_v33_timescale_screen."
                "evaluate_selection",
                side_effect=self.fake_selection,
            ):
                gate = evaluate_timescale_screen(development)
            (development / "v33_timescale_gate.json").write_text(
                json.dumps(gate))
            selected = gate["selected_for_confirmation"]
            confirmation = self.make_artifacts(
                root,
                directory="confirmation",
                configs=confirmation_configs(selected),
                train_seeds=CONFIRMATION_TRAIN_SEEDS,
                eval_seeds=CONFIRMATION_EVAL_SEEDS,
                stage="confirmation",
                source_sha="f" * 64,
            )
            with self.assertRaisesRegex(ValueError, "model_source_unchanged"):
                evaluate_timescale_confirmation(
                    confirmation,
                    development_dir=development,
                )


if __name__ == "__main__":
    unittest.main()
