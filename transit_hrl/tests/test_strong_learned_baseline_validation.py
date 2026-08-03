import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.strong_learned_baseline_validation import (
    build_paired_checks,
    canonical_hyperparameter_sha256,
    learning_dynamics_summary,
    merge_strong_learned_baseline_shards,
    run_strong_learned_baseline_validation,
    selected_experiment_cells,
    selected_scenario_policy_pairs,
    write_outputs,
)
from freq_hrl.experiments.reproducibility import (
    current_freq_hrl_source_manifest_sha256,
)


class StrongLearnedBaselineValidationTest(unittest.TestCase):
    def test_confirmatory_run_requires_frozen_config_identity(self):
        with self.assertRaisesRegex(ValueError, "frozen_nested_validation"):
            run_strong_learned_baseline_validation(
                scenarios=["persistent_shift"],
                policy_modes=["freq_hrl"],
                train_seeds=[42],
                eval_seeds=[123],
                steps=8,
                assets=2,
                iterations=1,
                optimizer_seed=7,
                min_pairs=1,
                confirmatory=True,
            )
        payload = run_strong_learned_baseline_validation(
            scenarios=["persistent_shift"],
            policy_modes=["freq_hrl"],
            train_seeds=[42],
            eval_seeds=[123],
            steps=8,
            assets=2,
            iterations=1,
            optimizer_seed=7,
            min_pairs=1,
            confirmatory=True,
            hyperparameter_source="frozen_nested_validation",
            frozen_config_sha256="a" * 64,
            selected_candidate_id="ppo_lr1e4_std15",
            frozen_candidate_parameters_sha256=canonical_hyperparameter_sha256({
                "hidden_dim": 64,
                "learning_rate": 3e-4,
                "epochs": 4,
                "minibatch_size": 512,
                "init_log_std": -1.0,
                "reward_scale": 100.0,
            }),
            code_revision="c" * 40,
            expected_source_manifest_sha256=(
                current_freq_hrl_source_manifest_sha256()
            ),
        )
        self.assertEqual(
            payload["summary"]["hyperparameter_protocol_status"],
            "frozen_validation_only",
        )
        self.assertEqual(payload["per_seed"][0]["frozen_config_sha256"], "a" * 64)
        self.assertEqual(payload["summary"]["source_identity_status"], "verified")

    def test_confirmatory_run_rejects_staged_source_drift(self):
        with self.assertRaisesRegex(RuntimeError, "staged source manifest mismatch"):
            run_strong_learned_baseline_validation(
                scenarios=["persistent_shift"],
                policy_modes=["freq_hrl"],
                train_seeds=[42],
                eval_seeds=[123],
                steps=8,
                assets=2,
                iterations=1,
                optimizer_seed=7,
                min_pairs=1,
                confirmatory=True,
                hyperparameter_source="frozen_nested_validation",
                frozen_config_sha256="a" * 64,
                selected_candidate_id="ppo_unit",
                frozen_candidate_parameters_sha256="b" * 64,
                code_revision="c" * 40,
                expected_source_manifest_sha256="0" * 64,
            )

    def test_learning_gate_rejects_random_initialization_as_learned_policy(self):
        unsupported = learning_dynamics_summary([{
            "policy_mode": "freq_hrl",
            "selected_checkpoint_iteration": -1,
            "validation_learning_gain": 0.0,
        }])
        self.assertEqual(unsupported["learning_dynamics_status"], "not_supported")
        supported = learning_dynamics_summary([
            {
                "policy_mode": "freq_hrl",
                "selected_checkpoint_iteration": iteration,
                "validation_learning_gain": 0.01,
            }
            for iteration in range(5)
        ])
        self.assertEqual(supported["learning_dynamics_status"], "supported")

    def test_selected_pairs_support_scheduler_sharding(self):
        pairs = selected_scenario_policy_pairs(
            ["persistent_shift", "localized_burst"],
            ["freq_hrl", "flat_ppo", "generic_hrl_ppo"],
            shard_index=1,
            num_shards=3,
        )
        self.assertEqual(pairs, [
            ("persistent_shift", "flat_ppo"),
            ("localized_burst", "flat_ppo"),
        ])

    def test_selected_cells_shard_independent_training_replicates(self):
        cells = selected_experiment_cells(
            ["persistent_shift"],
            ["freq_hrl", "flat_ppo"],
            [7, 11],
            shard_index=1,
            num_shards=2,
        )
        self.assertEqual(cells, [
            ("persistent_shift", "freq_hrl", 11),
            ("persistent_shift", "flat_ppo", 11),
        ])

    def test_runner_emits_learned_rows_and_budget_tables(self):
        payload = run_strong_learned_baseline_validation(
            scenarios=["persistent_shift"],
            policy_modes=["freq_hrl", "flat_ppo", "generic_hrl_ppo"],
            train_seeds=[42],
            eval_seeds=[123],
            steps=24,
            assets=2,
            iterations=1,
            optimizer_seed=7,
            min_pairs=1,
            shard_index=0,
            num_shards=1,
        )
        self.assertEqual(payload["summary"]["rows"], 3)
        self.assertEqual(
            payload["summary"]["parameter_budget_status"],
            "matched_within_5pct",
        )
        self.assertEqual(
            payload["summary"]["trainer_budget_status"],
            "controlled_by_ppo_family",
        )
        self.assertTrue(any(
            row["trainer"] == "frequency_separated_smdp_ppo_v2"
            for row in payload["per_seed"]
            if row["baseline"] == "freq_hrl"
        ))
        self.assertEqual(len(payload["paired_checks"]), 16)
        self.assertTrue(any(row["baseline"] == "flat_ppo" for row in payload["per_seed"]))
        self.assertTrue(any(row["policy_mode"] == "generic_hrl_ppo" for row in payload["parameter_budget"]))
        parameter_counts = [
            row["parameter_count"] for row in payload["parameter_budget"]
        ]
        self.assertLessEqual(max(parameter_counts) / min(parameter_counts), 1.05)
        by_mode = {row["baseline"]: row for row in payload["per_seed"]}
        self.assertEqual(by_mode["flat_ppo"]["upper_decision_count"], 24)
        self.assertLess(by_mode["generic_hrl_ppo"]["upper_decision_count"], 24)

    def test_runner_executes_real_sac_and_td3_under_shared_step_budget(self):
        payload = run_strong_learned_baseline_validation(
            scenarios=["persistent_shift"],
            policy_modes=["freq_hrl", "flat_sac", "flat_td3"],
            train_seeds=[42],
            eval_seeds=[123],
            steps=40,
            assets=2,
            iterations=1,
            optimizer_seed=7,
            min_pairs=1,
            offpolicy_hidden_dim=16,
            offpolicy_warmup_steps=8,
            offpolicy_batch_size=8,
        )
        self.assertEqual(payload["summary"]["rows"], 3)
        self.assertEqual(payload["summary"]["sac_td3_status"], "complete")
        self.assertEqual(payload["summary"]["environment_step_budget_status"], "matched")
        self.assertNotEqual(
            payload["summary"]["strong_learned_baseline_evidence_status"],
            "supported",
        )
        self.assertEqual(
            payload["summary"]["parameter_budget_status"],
            "controlled_by_algorithm_family",
        )
        by_mode = {row["policy_mode"]: row for row in payload["run_summary"]}
        self.assertGreater(by_mode["flat_sac"]["gradient_updates_train"], 0)
        self.assertGreater(by_mode["flat_td3"]["gradient_updates_train"], 0)
        self.assertEqual(
            {row["environment_steps_train"] for row in payload["sample_efficiency"]},
            {40},
        )
        commands = {row["policy_mode"]: row["command"] for row in payload["experiment_manifest"]}
        self.assertIn("strong_learned_baseline_validation", commands["flat_sac"])
        self.assertIn("--optimizer-seeds 7 --min-pairs 1", commands["flat_td3"])
        self.assertTrue(all("holm_adjusted_p_value" in row for row in payload["paired_checks"]))

    def test_paired_checks_cluster_eval_paths_by_training_replicate(self):
        rows = []
        for replicate in (7, 11):
            for seed in (123, 456):
                for baseline, offset in (("freq_hrl", 1.0), ("flat_ppo", 0.0)):
                    rows.append({
                        "scenario": "persistent_shift",
                        "training_replicate_seed": replicate,
                        "seed": seed,
                        "baseline": baseline,
                        "metric_contract_version": "trading_metrics_v2",
                        "training_path_protocol": "fresh_deterministic_path_per_root_and_iteration_v2",
                        "checkpoint_selection_protocol": "disjoint_validation_paths",
                        "total_return": offset,
                        "episode_information_ratio": offset,
                        "FocusScore": offset,
                        "LowerLFDrift": 1.0 - offset,
                    })
        checks = build_paired_checks(
            rows,
            controls=("flat_ppo",),
            min_pairs=2,
        )
        self.assertEqual(len(checks), 4)
        self.assertTrue(all(row["n_common"] == 4 for row in checks))
        self.assertTrue(all(row["n_independent"] == 2 for row in checks))
        self.assertTrue(all(
            row["cluster_keys"] == ["training_replicate_seed"]
            for row in checks
        ))

    def test_paired_checks_reject_legacy_eval_seed_only_rows(self):
        with self.assertRaisesRegex(ValueError, "legacy eval-seed-only"):
            build_paired_checks([{
                "scenario": "persistent_shift",
                "seed": 123,
                "baseline": "freq_hrl",
                "total_return": 1.0,
            }])

    def test_write_outputs_creates_main_artifacts(self):
        payload = run_strong_learned_baseline_validation(
            scenarios=["persistent_shift"],
            policy_modes=["freq_hrl", "flat_ppo"],
            train_seeds=[42],
            eval_seeds=[123],
            steps=20,
            assets=2,
            iterations=1,
            optimizer_seed=7,
            min_pairs=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_outputs(out, payload)
            self.assertTrue((out / "per_seed.csv").exists())
            self.assertTrue((out / "paired_checks.csv").exists())
            self.assertTrue((out / "parameter_budget.csv").exists())
            self.assertTrue((out / "sample_efficiency.csv").exists())
            self.assertTrue((out / "summary.json").exists())
            checkpoint_files = list((out / "checkpoints").glob("*.pt"))
            self.assertEqual(len(checkpoint_files), 2)

    def test_merge_shards_recomputes_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dirs = []
            for idx, modes in enumerate((["freq_hrl", "flat_ppo"], ["generic_hrl_ppo"])):
                payload = run_strong_learned_baseline_validation(
                    scenarios=["persistent_shift"],
                    policy_modes=modes,
                    train_seeds=[42],
                    eval_seeds=[123],
                    steps=20,
                    assets=2,
                    iterations=1,
                    optimizer_seed=7,
                    min_pairs=1,
                )
                out = root / f"shard_{idx}"
                write_outputs(out, payload)
                shard_dirs.append(out)
            merged = merge_strong_learned_baseline_shards(shard_dirs, min_pairs=1)
            self.assertEqual(merged["summary"]["merge_status"], "merged")
            self.assertEqual(merged["summary"]["rows"], 3)
            self.assertEqual(len(merged["paired_checks"]), 16)
            self.assertEqual(merged["summary"]["training_replicate_count"], 1)
            self.assertEqual(merged["summary"]["matrix_coverage_status"], "complete")
            self.assertEqual(merged["summary"]["training_protocol_status"], "valid")


if __name__ == "__main__":
    unittest.main()
