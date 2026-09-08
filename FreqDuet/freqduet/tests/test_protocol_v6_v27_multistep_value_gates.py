from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.audit_protocol_v6_v27_multistep_value_common import (
    CANDIDATES,
    CANDIDATE_SPECS,
    PRIORITY,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
)
from scripts.audit_protocol_v6_v27_multistep_value_screen import (
    CONFIGS as SCREEN_CONFIGS,
    EVAL_SEEDS as SCREEN_EVAL_SEEDS,
    TRAIN_EPISODES as SCREEN_TRAIN_EPISODES,
    TRAIN_SEEDS as SCREEN_TRAIN_SEEDS,
    evaluate_v27_multistep_value_screen,
    main as screen_main,
)
from scripts.audit_protocol_v6_v27_multistep_value_smoke import (
    CONFIGS as SMOKE_CONFIGS,
    EVAL_SEEDS as SMOKE_EVAL_SEEDS,
    TRAIN_EPISODES as SMOKE_TRAIN_EPISODES,
    TRAIN_SEEDS as SMOKE_TRAIN_SEEDS,
    evaluate_v27_multistep_value_smoke,
    main as smoke_main,
)


class ProtocolV6V27MultiStepValueGateTest(unittest.TestCase):
    @staticmethod
    def _runtime_row(config: str, *, frozen: bool) -> dict[str, object]:
        candidate = config in CANDIDATES
        spec = CANDIDATE_SPECS.get(config, {
            "horizon_steps": 0, "ucb_beta": 0.0})
        ready = float(candidate)
        replay = 800 if candidate else 0
        targets = 900 if candidate else 0
        updates = 35 if candidate else 0
        return {
            "lower_observation_contract": "deployable_apc_avl_v4",
            "headway_reward_mode": "forward_event_only",
            "frequency_observation_source": "apc_boardings",
            "lower_discrete_critic": "continuous_action",
            "lower_causal_guard_enabled": 0.0,
            "lower_causal_guard_adjustment_mean_s": 0.0,
            "follower_target_calibration_enabled": 0.0,
            "follower_target_calibration_mode": "disabled",
            "lower_regularity_policy_enabled": float(candidate),
            "lower_regularity_policy_mode": (
                "causal_multistep_arrival_delta_regret_dual_v12"
                if candidate else "disabled"),
            "lower_regularity_policy_constraint_cost_mode": (
                "downstream_arrival_value_regret_v5"
                if candidate else "disabled"),
            "lower_regularity_policy_constraint_scale_mode": (
                "cost_limit_ratio_v1" if candidate else "raw_cost_v1"),
            "lower_regularity_policy_cost_limit": 0.001 if candidate else 0.0,
            "lower_regularity_policy_valid_fraction": 0.8 if candidate else 0.0,
            "lower_multistep_value_enabled": float(candidate),
            "lower_multistep_value_mode": (
                "discounted_future_arrival_cost_change_v1"
                if candidate else "disabled"),
            "lower_multistep_value_horizon_steps": spec["horizon_steps"],
            "lower_multistep_value_discount": 1.0 if candidate else 0.0,
            "lower_multistep_value_ucb_beta": spec["ucb_beta"],
            "lower_multistep_value_ready": ready,
            "lower_multistep_value_replay_size": replay,
            "lower_multistep_value_targets_emitted": targets,
            "lower_multistep_value_terminal_tails_discarded": (
                10 if candidate else 0),
            "lower_multistep_value_episode_tails_discarded": (
                5 if candidate else 0),
            "lower_multistep_value_critic_updates": updates,
            "lower_multistep_value_critic_loss": (
                0.2 if candidate and not frozen else 0.0),
            "lower_multistep_value_target_mean": (
                0.01 if candidate and not frozen else 0.0),
            "lower_multistep_value_target_std": (
                0.03 if candidate and not frozen else 0.0),
            "lower_multistep_value_prediction_mean": (
                0.01 if candidate and not frozen else 0.0),
            "lower_multistep_value_prediction_std": (
                0.02 if candidate and not frozen else 0.0),
            "lower_multistep_value_grad_norm": (
                0.1 if candidate and not frozen else 0.0),
            "lower_multistep_value_action_span_mean": (
                0.04 if candidate and not frozen else 0.0),
            "lower_multistep_value_advantage_mean": (
                0.01 if candidate and not frozen else 0.0),
            "lower_multistep_value_advantage_std_mean": (
                0.02 if candidate and not frozen else 0.0),
            "lower_multistep_value_positive_regret_mean": (
                0.01 if candidate and not frozen else 0.0),
            "lower_multistep_value_positive_regret_max": (
                0.03 if candidate and not frozen else 0.0),
            "lower_multistep_value_frozen": float(frozen),
        }

    @classmethod
    def _training_row(cls, candidate: str, ep: int) -> dict[str, object]:
        row = cls._runtime_row(candidate, frozen=False)
        row.update({
            "ep": ep,
            "lower_multistep_value_replay_size": 800 + 100 * ep,
            "lower_multistep_value_targets_emitted": 900 + 100 * ep,
            "lower_multistep_value_terminal_tails_discarded": 10 + ep,
            "lower_multistep_value_episode_tails_discarded": 5 + ep,
            "lower_multistep_value_critic_updates": 35 + ep,
        })
        return row

    @classmethod
    def _evaluation_row(
        cls, config: str, train_episodes: int
    ) -> dict[str, object]:
        row = cls._runtime_row(config, frozen=True)
        row["checkpoint_ep"] = train_episodes - 1
        row["lower_policy_frozen"] = 1.0
        row["lower_critic_frozen"] = 1.0
        row["upper_policy_frozen"] = 1.0
        return row

    @staticmethod
    def _outcomes(config: str) -> dict[str, float]:
        if config in CANDIDATES:
            return {
                "headway_cv": 0.1975,
                "restricted_total_journey_horizon_min": 19.94,
                "service_cost": 0.502,
                "passenger_unserved_rate": 0.010,
                "lower_action_mean": 10.20,
                "holding_vehicle_seconds": 990.0,
            }
        if config == V13_ZERO_HOLD_ADVANTAGE:
            return {
                "headway_cv": 0.1995,
                "restricted_total_journey_horizon_min": 20.00,
                "service_cost": 0.500,
                "passenger_unserved_rate": 0.010,
                "lower_action_mean": 10.00,
                "holding_vehicle_seconds": 1000.0,
            }
        return {
            "headway_cv": 0.2000,
            "restricted_total_journey_horizon_min": 20.00,
            "service_cost": 0.500,
            "passenger_unserved_rate": 0.010,
            "lower_action_mean": 10.00,
            "holding_vehicle_seconds": 1000.0,
        }

    @classmethod
    def _write_fixture(
        cls,
        root: Path,
        *,
        configs: list[str],
        train_seeds: list[int],
        eval_seeds: list[int],
        train_episodes: int,
    ) -> tuple[Path, Path]:
        aggregate = root / "aggregate"
        logs = root / "logs"
        aggregate.mkdir()
        logs.mkdir()
        git_record = {
            "commit": "a" * 40,
            "branch": "codex/freqduet-v6-causal-protocol",
            "tracked_dirty": False,
        }
        manifest = {
            "strict_complete": True,
            "run_manifests_verified": True,
            "common_random_numbers_verified": True,
            "stage": "exploratory",
            "independent_confirmation": False,
            "configs": configs,
            "train_seeds": train_seeds,
            "eval_seeds": eval_seeds,
            "train_episodes": train_episodes,
            "checkpoint_ep": train_episodes - 1,
            "reference": V13_ANCHOR,
            "run_git_provenance": git_record,
            "git": git_record,
            "expected_rollouts": (
                len(configs) * len(train_seeds) * len(eval_seeds)),
        }
        (aggregate / "matrix_manifest.json").write_text(json.dumps(manifest))
        (aggregate / "frozen_summary.csv").write_text("config\n")
        (aggregate / "frozen_paired_deltas.csv").write_text(
            "candidate,reference\n")

        evaluation_rows = []
        for config in configs:
            for train_seed in train_seeds:
                for eval_seed in eval_seeds:
                    row = cls._evaluation_row(config, train_episodes)
                    row.update(cls._outcomes(config))
                    row.update({
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "scenario_tape_id": f"tape-{eval_seed}",
                    })
                    evaluation_rows.append(row)
        pd.DataFrame(evaluation_rows).to_csv(
            aggregate / "frozen_per_eval.csv", index=False)

        for candidate in CANDIDATES:
            for train_seed in train_seeds:
                run = logs / f"{candidate}_seed{train_seed}"
                run.mkdir()
                pd.DataFrame([
                    cls._training_row(candidate, ep)
                    for ep in range(train_episodes)
                ]).to_csv(run / "diagnostics.csv", index=False)
        return aggregate, logs

    def test_smoke_passes_only_as_mechanical_evidence(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SMOKE_CONFIGS,
                train_seeds=SMOKE_TRAIN_SEEDS,
                eval_seeds=SMOKE_EVAL_SEEDS,
                train_episodes=SMOKE_TRAIN_EPISODES,
            )
            result = evaluate_v27_multistep_value_smoke(aggregate, [logs])

        self.assertEqual(result["status"], "mechanical_pass")
        self.assertTrue(result["formal_screen_authorized"])
        self.assertFalse(result["effect_evidence"])

    def test_smoke_cli_prints_scheduler_success_marker(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SMOKE_CONFIGS,
                train_seeds=SMOKE_TRAIN_SEEDS,
                eval_seeds=SMOKE_EVAL_SEEDS,
                train_episodes=SMOKE_TRAIN_EPISODES,
            )
            stdout = StringIO()
            with patch("sys.argv", [
                    "v27-smoke", str(aggregate),
                    "--logs-root", str(logs), "--require-pass"]), \
                    redirect_stdout(stdout):
                smoke_main()

        self.assertTrue(stdout.getvalue().rstrip().endswith(
            "DONE V27 causal-multistep-value smoke gate"))

    def test_frozen_unready_objective_blocks_smoke(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SMOKE_CONFIGS,
                train_seeds=SMOKE_TRAIN_SEEDS,
                eval_seeds=SMOKE_EVAL_SEEDS,
                train_episodes=SMOKE_TRAIN_EPISODES,
            )
            path = aggregate / "frozen_per_eval.csv"
            frame = pd.read_csv(path)
            candidate = CANDIDATES[0]
            mask = frame["config"] == candidate
            frame.loc[mask, "lower_multistep_value_ready"] = 0.0
            frame.loc[mask, "lower_multistep_value_replay_size"] = 0
            frame.to_csv(path, index=False)
            result = evaluate_v27_multistep_value_smoke(aggregate, [logs])

        self.assertEqual(result["status"], "no_pass")
        self.assertFalse(result["candidate_results"][candidate][
            "evaluation_checks"]["frozen_value_objective_is_ready"])

    def test_screen_selects_registered_priority_candidate(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SCREEN_CONFIGS,
                train_seeds=SCREEN_TRAIN_SEEDS,
                eval_seeds=SCREEN_EVAL_SEEDS,
                train_episodes=SCREEN_TRAIN_EPISODES,
            )
            result = evaluate_v27_multistep_value_screen(aggregate, [logs])

        self.assertEqual(result["status"], "exploratory_candidate_selected")
        self.assertEqual(result["selected_for_confirmation"], PRIORITY[0])
        self.assertFalse(result["claim_eligible"])

    def test_screen_cli_prints_scheduler_success_marker(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SCREEN_CONFIGS,
                train_seeds=SCREEN_TRAIN_SEEDS,
                eval_seeds=SCREEN_EVAL_SEEDS,
                train_episodes=SCREEN_TRAIN_EPISODES,
            )
            stdout = StringIO()
            with patch("sys.argv", [
                    "v27-screen", str(aggregate),
                    "--logs-root", str(logs), "--require-selection"]), \
                    redirect_stdout(stdout):
                screen_main()

        self.assertTrue(stdout.getvalue().rstrip().endswith(
            "DONE V27 causal-multistep-value screen gate"))

    def test_failed_priority_candidate_falls_through(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SCREEN_CONFIGS,
                train_seeds=SCREEN_TRAIN_SEEDS,
                eval_seeds=SCREEN_EVAL_SEEDS,
                train_episodes=SCREEN_TRAIN_EPISODES,
            )
            path = aggregate / "frozen_per_eval.csv"
            frame = pd.read_csv(path)
            failed = PRIORITY[0]
            frame.loc[frame["config"] == failed, "headway_cv"] = 0.2000
            frame.to_csv(path, index=False)
            result = evaluate_v27_multistep_value_screen(aggregate, [logs])

        self.assertFalse(result["candidate_results"][failed]["passes"])
        self.assertEqual(result["selected_for_confirmation"], PRIORITY[1])

    def test_v19_comparison_failure_blocks_selection(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SCREEN_CONFIGS,
                train_seeds=SCREEN_TRAIN_SEEDS,
                eval_seeds=SCREEN_EVAL_SEEDS,
                train_episodes=SCREEN_TRAIN_EPISODES,
            )
            path = aggregate / "frozen_per_eval.csv"
            frame = pd.read_csv(path)
            frame.loc[
                frame["config"] == V13_ZERO_HOLD_ADVANTAGE,
                "restricted_total_journey_horizon_min",
            ] = 19.70
            frame.to_csv(path, index=False)
            result = evaluate_v27_multistep_value_screen(aggregate, [logs])

        self.assertEqual(result["status"], "no_pass")
        self.assertTrue(all(
            not value["outcome_checks"]["journey_improves_v19"]
            for value in result["candidate_results"].values()))

    def test_manifest_seed_drift_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SCREEN_CONFIGS,
                train_seeds=SCREEN_TRAIN_SEEDS,
                eval_seeds=SCREEN_EVAL_SEEDS,
                train_episodes=SCREEN_TRAIN_EPISODES,
            )
            path = aggregate / "matrix_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["eval_seeds"] = [1, 2, 3, 4]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "strict checks failed"):
                evaluate_v27_multistep_value_screen(aggregate, [logs])

    def test_scenario_tape_drift_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SCREEN_CONFIGS,
                train_seeds=SCREEN_TRAIN_SEEDS,
                eval_seeds=SCREEN_EVAL_SEEDS,
                train_episodes=SCREEN_TRAIN_EPISODES,
            )
            path = aggregate / "frozen_per_eval.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "scenario_tape_id"] = "wrong-tape"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "strict checks failed"):
                evaluate_v27_multistep_value_screen(aggregate, [logs])


if __name__ == "__main__":
    unittest.main()
