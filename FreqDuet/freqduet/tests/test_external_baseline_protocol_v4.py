from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

import pandas as pd

from scripts.compare_freqduet_external_frozen import METRICS, V5_METRICS, compare
from scripts.run_freqduet_external_baselines import (
    external_evaluator_fingerprint,
    install_exact_dispatch_schedule,
    make_env_from_config,
    run_one,
)


class ExternalBaselineProtocolV4Test(unittest.TestCase):
    def test_exact_schedule_writes_consecutive_headway_gaps(self):
        trips = []
        turn = 0
        for direction, launches in [
            (False, [0.0, 420.0, 930.0]),
            (True, [60.0, 540.0, 1020.0]),
        ]:
            for launch in launches:
                trips.append(SimpleNamespace(
                    direction=direction,
                    launch_time=launch,
                    launch_turn=turn,
                    target_headway=360.0,
                ))
                turn += 1
        env = SimpleNamespace(timetables=trips)

        install_exact_dispatch_schedule(
            env,
            lambda direction, predecessor_launch, trip: 300.0,
            projection_mode="exact_test_v4",
        )

        for direction in (False, True):
            rows = sorted(
                (trip for trip in trips if trip.direction == direction),
                key=lambda trip: trip._freqduet_scheduled_launch,
            )
            gaps = [
                rows[index]._freqduet_scheduled_launch
                - rows[index - 1]._freqduet_scheduled_launch
                for index in range(1, len(rows))
            ]
            self.assertEqual(gaps, [300.0, 300.0])
            self.assertTrue(all(
                trip._freqduet_terminal_dispatch for trip in rows))
            self.assertTrue(all(
                trip._freqduet_projection_mode == "exact_test_v4"
                for trip in rows))

    def test_v4_baseline_env_uses_deployable_physical_contract(self):
        env, config = make_env_from_config(
            "F_freqduet_protocol_v4_main_hiro")

        self.assertEqual(config["protocol"]["version"], "freqduet-eval-v4")
        self.assertEqual(env.fleet_inventory_mode, "fixed_pool")
        self.assertEqual(env.lower_observation_contract, "deployable_apc_avl_v4")
        self.assertEqual(env.lower_state_input_schema, "causal_forward_v4")
        self.assertEqual(env.headway_reward_mode, "forward_event_only")
        self.assertTrue(env.frequency_enabled)

    def test_v5_baseline_env_uses_the_same_fixed_pool_contract(self):
        env, config = make_env_from_config(
            "F_freqduet_protocol_v5_main_hiro")

        self.assertEqual(config["protocol"]["version"], "freqduet-eval-v5")
        self.assertEqual(env.fleet_inventory_mode, "fixed_pool")
        self.assertEqual(env.lower_observation_contract, "deployable_apc_avl_v4")
        self.assertEqual(config["upper"]["fleet_mode"], "fixed")

    def test_direct_scenario_mode_rejects_episode_averaging(self):
        with self.assertRaisesRegex(ValueError, "requires episodes=1"):
            run_one(
                config="F_freqduet_protocol_v4_main_hiro",
                variant="fixed_headway",
                seed=31013,
                episodes=2,
                logs_dir=Path("unused"),
                direct_scenario_seed=True,
            )

    def test_evaluator_fingerprint_covers_rule_and_driver(self):
        fingerprint = external_evaluator_fingerprint()
        paths = {item["path"] for item in fingerprint["files"]}

        self.assertEqual(len(fingerprint["sha256"]), 64)
        self.assertIn("scripts/run_freqduet_external_baselines.py", paths)
        self.assertIn("run_baseline_rule.py", paths)

    def test_external_comparison_requires_and_uses_common_tapes(self):
        learned_rows = []
        for train_seed in (1, 2):
            for eval_seed in (11, 12):
                row = {
                    "config": "main",
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "scenario_tape_id": f"tape-{eval_seed}",
                    "protocol_version": "freqduet-eval-v4",
                }
                row.update({metric: 0.8 for metric in METRICS})
                learned_rows.append(row)
        baseline_rows = []
        for eval_seed in (11, 12):
            row = {
                "config": "main",
                "method": "fixed_headway",
                "eval_seed": eval_seed,
                "scenario_tape_id": f"tape-{eval_seed}",
                "protocol_version": "freqduet-eval-v4",
            }
            row.update({metric: 1.0 for metric in METRICS})
            baseline_rows.append(row)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned_path = root / "learned.csv"
            baseline_path = root / "baseline.csv"
            out_dir = root / "out"
            pd.DataFrame(learned_rows).to_csv(learned_path, index=False)
            pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
            source_sha = "a" * 64
            (root / "matrix_manifest.json").write_text(json.dumps({
                "protocol_version": "freqduet-eval-v4",
                "strict_complete": True,
                "common_random_numbers_verified": True,
                "run_manifests_verified": True,
                "run_source_fingerprint": {"sha256": source_sha},
            }))
            (root / "external_baselines_summary.json").write_text(json.dumps({
                "direct_scenario_seeds": True,
                "run_manifests_verified": True,
                "core_source_sha256": source_sha,
                "evaluator_source_sha256": "b" * 64,
            }))

            compare(
                learned_path,
                baseline_path,
                out_dir,
                learned_config="main",
            )

            summary = pd.read_csv(
                out_dir / "learned_vs_external_summary.csv")
            self.assertAlmostEqual(
                summary.iloc[0]["delta_service_cost_restricted_mean"], -0.2)
            self.assertEqual(int(summary.iloc[0]["n_pairs"]), 4)

            broken = pd.DataFrame(baseline_rows)
            broken.loc[0, "scenario_tape_id"] = "wrong"
            broken.to_csv(baseline_path, index=False)
            with self.assertRaisesRegex(ValueError, "tape mismatch"):
                compare(
                    learned_path,
                    baseline_path,
                    root / "broken",
                    learned_config="main",
                )

    def test_external_comparison_rejects_core_source_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned_path = root / "learned.csv"
            baseline_path = root / "baseline.csv"
            pd.DataFrame([{
                "config": "main", "train_seed": 1, "eval_seed": 11,
                "scenario_tape_id": "tape-11",
                "protocol_version": "freqduet-eval-v4",
                **{metric: 1.0 for metric in METRICS},
            }]).to_csv(learned_path, index=False)
            pd.DataFrame([{
                "config": "main", "method": "fixed_headway",
                "eval_seed": 11, "scenario_tape_id": "tape-11",
                "protocol_version": "freqduet-eval-v4",
                **{metric: 1.0 for metric in METRICS},
            }]).to_csv(baseline_path, index=False)
            (root / "matrix_manifest.json").write_text(json.dumps({
                "protocol_version": "freqduet-eval-v4",
                "strict_complete": True,
                "common_random_numbers_verified": True,
                "run_manifests_verified": True,
                "run_source_fingerprint": {"sha256": "a" * 64},
            }))
            (root / "external_baselines_summary.json").write_text(json.dumps({
                "direct_scenario_seeds": True,
                "run_manifests_verified": True,
                "core_source_sha256": "c" * 64,
                "evaluator_source_sha256": "b" * 64,
            }))

            with self.assertRaisesRegex(ValueError, "fingerprints differ"):
                compare(
                    learned_path,
                    baseline_path,
                    root / "out",
                    learned_config="main",
                )

    def test_external_comparison_accepts_v5_normalized_safety_metrics(self):
        learned_rows = []
        for train_seed in (1, 2):
            for eval_seed in (11, 12):
                learned_rows.append({
                    "config": "main",
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "scenario_tape_id": f"tape-{eval_seed}",
                    "protocol_version": "freqduet-eval-v5",
                    **{metric: 0.8 for metric in V5_METRICS},
                })
        baseline_rows = [{
            "config": "main",
            "method": "fixed_headway",
            "eval_seed": eval_seed,
            "scenario_tape_id": f"tape-{eval_seed}",
            "protocol_version": "freqduet-eval-v5",
            **{metric: 1.0 for metric in V5_METRICS},
        } for eval_seed in (11, 12)]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned_path = root / "learned.csv"
            baseline_path = root / "baseline.csv"
            pd.DataFrame(learned_rows).to_csv(learned_path, index=False)
            pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
            source_sha = "a" * 64
            (root / "matrix_manifest.json").write_text(json.dumps({
                "protocol_version": "freqduet-eval-v5",
                "strict_complete": True,
                "common_random_numbers_verified": True,
                "run_manifests_verified": True,
                "run_source_fingerprint": {"sha256": source_sha},
            }))
            (root / "external_baselines_summary.json").write_text(json.dumps({
                "direct_scenario_seeds": True,
                "run_manifests_verified": True,
                "core_source_sha256": source_sha,
                "evaluator_source_sha256": "b" * 64,
            }))

            compare(
                learned_path,
                baseline_path,
                root / "out",
                learned_config="main",
            )

            summary = pd.read_csv(
                root / "out" / "learned_vs_external_summary.csv")
            self.assertAlmostEqual(
                summary.iloc[0][
                    "delta_holding_passenger_min_per_generated_mean"],
                -0.2,
            )


if __name__ == "__main__":
    unittest.main()
