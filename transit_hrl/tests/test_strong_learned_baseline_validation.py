import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.strong_learned_baseline_validation import (
    merge_strong_learned_baseline_shards,
    run_strong_learned_baseline_validation,
    selected_scenario_policy_pairs,
    write_outputs,
)


class StrongLearnedBaselineValidationTest(unittest.TestCase):
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
        self.assertEqual(payload["summary"]["parameter_budget_status"], "matched")
        self.assertEqual(len(payload["paired_checks"]), 8)
        self.assertTrue(any(row["baseline"] == "flat_ppo" for row in payload["per_seed"]))
        self.assertTrue(any(row["policy_mode"] == "generic_hrl_ppo" for row in payload["parameter_budget"]))

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
                    optimizer_seed=7 + idx,
                    min_pairs=1,
                )
                out = root / f"shard_{idx}"
                write_outputs(out, payload)
                shard_dirs.append(out)
            merged = merge_strong_learned_baseline_shards(shard_dirs, min_pairs=1)
            self.assertEqual(merged["summary"]["merge_status"], "merged")
            self.assertEqual(merged["summary"]["rows"], 3)
            self.assertEqual(len(merged["paired_checks"]), 8)


if __name__ == "__main__":
    unittest.main()
