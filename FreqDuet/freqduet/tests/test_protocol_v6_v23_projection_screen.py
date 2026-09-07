import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_v23_projection_screen import (
    CANDIDATE,
    CONFIGS,
    CURRENT_MAIN,
    EVAL_SEEDS,
    HARD_MAIN,
    REFERENCE,
    TRAIN_SEEDS,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
    V20_QADV_B080,
    evaluate_v23_projection_screen,
)


class ProtocolV6V23ProjectionScreenTest(unittest.TestCase):
    def _base_row(self, *, candidate: bool) -> dict[str, object]:
        return {
            "lower_discrete_critic": (
                "zero_hold_advantage" if candidate else "continuous_action"),
            "lower_policy_frozen": 1.0,
            "lower_critic_frozen": 1.0,
            "upper_policy_frozen": 1.0,
            "lower_causal_guard_adjustment_mean_s": 0.0,
            "lower_regularity_policy_enabled": float(candidate),
            "lower_regularity_policy_mode": (
                "analytic_two_sided_hf_aggregate_gain_projection_v10"
                if candidate else "disabled"),
            "lower_regularity_policy_constraint_cost_mode": (
                "hf_aggregate_gain_shortfall_v4"
                if candidate else "disabled"),
            "lower_regularity_policy_constraint_scale_mode": (
                "cost_limit_ratio_v1" if candidate else "raw_cost_v1"),
            "lower_regularity_policy_dual_update_mode": (
                "exact_projection_v1" if candidate else "log_adam_v1"),
            "lower_regularity_policy_augmented_lagrangian_rho": 0.0,
            "lower_regularity_policy_cost_limit": 0.05 if candidate else 0.0,
            "lower_regularity_policy_evidence_valid_mean": (
                0.60 if candidate else 0.0),
            "lower_regularity_policy_penalty": 0.0,
            "lower_regularity_policy_augmented_penalty": 0.0,
            "lower_regularity_lambda": 0.0,
            "lower_regularity_gain_floor_enabled": float(candidate),
            "lower_regularity_gain_floor_mode": (
                "causal_hf_aggregate_gain_floor_v2"
                if candidate else "disabled"),
            "lower_regularity_gain_floor_base_fraction": (
                0.30 if candidate else 0.0),
            "lower_regularity_gain_floor_hf_increment": (
                0.30 if candidate else 0.0),
            "lower_regularity_gain_floor_hf_energy_scale": (
                0.04 if candidate else 1.0),
            "lower_regularity_gain_floor_hf_energy_exponent": 1.0,
            "lower_regularity_gain_floor_required_gain_mean": (
                0.002 if candidate else 0.0),
            "lower_regularity_gain_floor_expected_absolute_shortfall_mean": (
                0.00008 if candidate else 0.0),
            "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio": (
                0.04 if candidate else 0.0),
            "lower_regularity_entropy_split_enabled": float(candidate),
            "lower_regularity_entropy_target_fraction": (
                0.25 if candidate else 0.0),
            "lower_regularity_passenger_holding_enabled": float(candidate),
            "lower_regularity_passenger_holding_mode": (
                "causal_apc_person_delay_dual_v1"
                if candidate else "disabled"),
            "lower_regularity_passenger_constraint_scale_mode": (
                "cost_limit_ratio_v1" if candidate else "raw_cost_v1"),
            "lower_regularity_passenger_dual_update_mode": (
                "exact_projection_v1" if candidate else "log_adam_v1"),
            "lower_regularity_passenger_augmented_lagrangian_rho": 0.0,
            "lower_regularity_passenger_cost_limit": (
                0.08 if candidate else 0.0),
            "lower_regularity_passenger_expected_cost_mean": (
                0.07 if candidate else 0.0),
            "lower_regularity_passenger_actor_penalty": 0.0,
            "lower_regularity_passenger_actor_augmented_penalty": 0.0,
            "lower_regularity_passenger_lambda": 0.0,
            "lower_regularity_projection_enabled": float(candidate),
            "lower_regularity_projection_mode": (
                "joint_kl_soft_policy_target_v1"
                if candidate else "disabled"),
            "lower_regularity_projection_applied": 0.0,
            "lower_regularity_projection_regularity_target": (
                0.036 if candidate else 0.0),
            "lower_regularity_projection_passenger_target": (
                0.075 if candidate else 0.0),
        }

    def _write_fixture(
        self,
        root: Path,
        *,
        projection_cost: float = 0.036,
        frozen_cost: float = 0.04,
    ) -> tuple[Path, Path]:
        aggregate = root / "aggregate"
        logs = root / "logs"
        aggregate.mkdir()
        logs.mkdir()
        expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
        manifest = {
            "strict_complete": True,
            "run_manifests_verified": True,
            "common_random_numbers_verified": True,
            "stage": "exploratory",
            "independent_confirmation": False,
            "configs": CONFIGS,
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "train_episodes": 40,
            "checkpoint_ep": 39,
            "reference": V13_ANCHOR,
            "run_git_provenance": {"tracked_dirty": False},
            "expected_rollouts": len(CONFIGS) * expected_pairs,
        }
        (aggregate / "matrix_manifest.json").write_text(json.dumps(manifest))
        (aggregate / "frozen_summary.csv").write_text("config\n")

        values = {
            HARD_MAIN: (20.2, 0.210, 11.0, 115.0, 11.0),
            CURRENT_MAIN: (20.5, 0.202, 11.0, 115.0, 11.0),
            REFERENCE: (21.0, 0.250, 12.0, 120.0, 12.0),
            V13_ANCHOR: (20.0, 0.205, 10.0, 110.0, 10.0),
            V13_ZERO_HOLD_ADVANTAGE: (20.0, 0.203, 10.0, 110.0, 10.0),
            V20_QADV_B080: (19.65, 0.230, 8.0, 95.0, 8.0),
            CANDIDATE: (19.5, 0.180, 9.0, 100.0, 9.0),
        }
        evaluation_rows = []
        for config in CONFIGS:
            candidate = config == CANDIDATE
            journey, cv, action, holding, denied = values[config]
            for train_seed in TRAIN_SEEDS:
                for eval_seed in EVAL_SEEDS:
                    row = self._base_row(candidate=candidate)
                    row.update({
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "restricted_total_journey_horizon_min": journey,
                        "headway_cv": cv,
                        "lower_action_mean": action,
                        "holding_vehicle_seconds": holding,
                        "fleet_denied_dispatch_events": denied,
                    })
                    if candidate:
                        row[
                            "lower_regularity_gain_floor_expected_absolute_shortfall_mean"
                        ] = 0.002 * frozen_cost
                        row[
                            "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio"
                        ] = frozen_cost
                    evaluation_rows.append(row)
        pd.DataFrame(evaluation_rows).to_csv(
            aggregate / "frozen_per_eval.csv", index=False)
        pd.DataFrame([{
            "candidate": CANDIDATE,
            "reference": V13_ANCHOR,
            "n_pairs": expected_pairs,
        }]).to_csv(aggregate / "frozen_paired_deltas.csv", index=False)

        for train_seed in TRAIN_SEEDS:
            run = logs / f"{CANDIDATE}_seed{train_seed}"
            run.mkdir()
            training_rows = []
            for ep in range(40):
                applied = ep >= 1
                row = self._base_row(candidate=True)
                row.update({
                    "ep": ep,
                    "lower_policy_frozen": 0.0,
                    "lower_critic_frozen": 0.0,
                    "upper_policy_frozen": 0.0,
                    "lower_regularity_projection_applied": float(applied),
                    "lower_regularity_projection_converged": float(applied),
                    "lower_regularity_projection_iterations": (
                        4.0 if applied else 0.0),
                    "lower_regularity_projection_valid_count": (
                        128.0 if applied else 0.0),
                    "lower_regularity_projection_base_regularity_cost": (
                        0.06 if applied else 0.0),
                    "lower_regularity_projection_base_passenger_cost": (
                        0.09 if applied else 0.0),
                    "lower_regularity_projection_target_regularity_cost": (
                        projection_cost if applied else 0.0),
                    "lower_regularity_projection_target_passenger_cost": (
                        0.075 if applied else 0.0),
                    "lower_regularity_projection_regularity_multiplier": (
                        0.2 if applied else 0.0),
                    "lower_regularity_projection_passenger_multiplier": (
                        0.3 if applied else 0.0),
                    "lower_regularity_projection_target_kl_from_soft": (
                        0.02 if applied else 0.0),
                    "lower_regularity_projection_target_entropy": (
                        1.2 if applied else 0.0),
                    "lower_regularity_projection_actor_reverse_kl": (
                        0.1 if applied else 0.0),
                    "lower_regularity_projection_base_action_mean_s": (
                        14.0 if applied else 0.0),
                    "lower_regularity_projection_target_action_mean_s": (
                        10.0 if applied else 0.0),
                    "lower_regularity_projection_target_action_change_mean_s": (
                        -4.0 if applied else 0.0),
                    "lower_regularity_projection_max_constraint_violation": 0.0,
                })
                training_rows.append(row)
            pd.DataFrame(training_rows).to_csv(
                run / "diagnostics.csv", index=False)
        return aggregate, logs

    def test_selects_candidate_when_every_registered_gate_passes(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(Path(tmp))
            result = evaluate_v23_projection_screen(aggregate, [logs])

        self.assertEqual(result["status"], "exploratory_candidate_selected")
        self.assertEqual(result["selected_for_confirmation"], CANDIDATE)
        self.assertTrue(all(result["strict_checks"].values()))
        self.assertTrue(all(result["mechanism_checks"].values()))
        self.assertTrue(all(result["outcome_checks"].values()))
        self.assertFalse(result["claim_eligible"])

    def test_replay_target_violation_fails_candidate(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp), projection_cost=0.03600002)
            result = evaluate_v23_projection_screen(aggregate, [logs])

        self.assertEqual(result["status"], "no_pass")
        self.assertFalse(result["mechanism_checks"][
            "every_recorded_projection_meets_replay_targets"])

    def test_frozen_budget_violation_fails_candidate(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp), frozen_cost=0.051)
            result = evaluate_v23_projection_screen(aggregate, [logs])

        self.assertEqual(result["status"], "no_pass")
        self.assertFalse(result["mechanism_checks"][
            "frozen_regularity_budget"])

    def test_manifest_seed_drift_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(Path(tmp))
            manifest_path = aggregate / "matrix_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["eval_seeds"] = [1, 2, 3, 4]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "strict checks failed"):
                evaluate_v23_projection_screen(aggregate, [logs])


if __name__ == "__main__":
    unittest.main()
