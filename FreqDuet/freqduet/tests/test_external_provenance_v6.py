from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pandas as pd

from scripts.analysis_provenance import (
    canonical_json_sha256,
    csv_artifact_record,
    validate_csv_artifact,
)
from scripts.compare_freqduet_external_frozen import (
    V6_LEARNED_PRIMARY_KEY,
    V6_METRICS,
    compare,
)
from scripts.run_freqduet_external_baselines import (
    BASELINE_VARIANTS,
    EXTERNAL_BASELINE_MANIFEST,
    V6_COMMAND_CONTRACT_VERSION,
    V6_DIAGNOSTICS_PRIMARY_KEY,
    V6_PER_SEED_PRIMARY_KEY,
    V6_PROTOCOL_VERSION,
    V6_RUN_MANIFEST_VERSION,
    V6_SAFETY_METRICS,
    V6_SUMMARY_MANIFEST_VERSION,
    aggregate,
    config_fingerprint,
    external_command_contract,
    external_evaluator_fingerprint,
    git_provenance,
    locked_runtime_environment,
    scenario_contract_fingerprint,
    source_fingerprint,
    run_one,
    validate_direct_baseline_run,
)
from scripts.run_freqduet_protocol_v2_matrix import MATRIX_MANIFEST_VERSION


CONFIG = "F_freqduet_protocol_v6_main_hiro"


def fingerprint(payload: dict) -> dict:
    return {
        "version": "test-contract-v1",
        "sha256": canonical_json_sha256(payload),
        "payload": payload,
    }


class ExternalProvenanceV6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core_source = source_fingerprint()
        cls.evaluator_source = external_evaluator_fingerprint()
        cls.config_record = config_fingerprint(CONFIG)
        cls.scenario_record = scenario_contract_fingerprint(CONFIG)
        cls.runtime_record = locked_runtime_environment()
        cls.git_record = git_provenance()

    def diagnostic_row(self, method: str, eval_seed: int) -> dict:
        row = {
            "ep": 0,
            "variant": method,
            "config": CONFIG,
            "domain": "terminal",
            "seed": eval_seed,
            "scenario_seed": eval_seed,
            "eval_seed": eval_seed,
            "protocol_version": V6_PROTOCOL_VERSION,
            "scenario_tape_id": f"tape-{eval_seed}",
            "avg_wait_min": 1.0,
            "peak_fleet": 12.0,
            "headway_cv": 0.1,
            "fleet_overshoot": 0.0,
            "composite": 1.0,
            "avg_wait_observed_min": 1.0,
            "restricted_wait_horizon_min": 1.0,
            "avg_in_vehicle_observed_min": 1.0,
            "restricted_in_vehicle_horizon_min": 1.0,
            "avg_total_journey_observed_min": 1.0,
            "restricted_total_journey_horizon_min": 1.0,
            "service_cost_observed": 1.0,
            "service_cost_restricted": 1.0,
            "passenger_unserved_rate": 0.0,
            "trip_launch_rate": 1.0,
            "trip_completion_rate": 1.0,
            "target_peak": 360.0,
            "target_offpeak": 360.0,
            "target_transition": 360.0,
            "N_fleet": 12,
            "physical_vehicle_count": 12,
            "fleet_inventory_mode": "fixed_pool",
            "lower_observation_contract": "deployable_apc_avl_v4",
            "headway_reward_mode": "forward_event_only",
            "baseline_schedule_projection_mode": (
                "exact_fixed_headway_schedule_v6"),
            "frequency_share_max_error": 0.0,
        }
        row.update({metric: 0.0 for metric in V6_SAFETY_METRICS})
        return row

    def write_external_run(
        self,
        logs_dir: Path,
        method: str,
        eval_seed: int,
        *,
        rows: list[dict] | None = None,
    ) -> Path:
        run_dir = logs_dir / f"{CONFIG}_{method}_seed{eval_seed}"
        run_dir.mkdir(parents=True)
        frame = pd.DataFrame(rows or [self.diagnostic_row(method, eval_seed)])
        diagnostics = run_dir / "diagnostics.csv"
        frame.to_csv(diagnostics, index=False)
        argv = ["run_freqduet_external_baselines.py", "--direct-scenario-seeds"]
        manifest = {
            "manifest_version": V6_RUN_MANIFEST_VERSION,
            "protocol_version": V6_PROTOCOL_VERSION,
            "config_name": CONFIG,
            "config_fingerprint": self.config_record,
            "core_source_fingerprint": self.core_source,
            "evaluator_source_fingerprint": self.evaluator_source,
            "variant": method,
            "method": method,
            "seed": eval_seed,
            "direct_scenario_seed": True,
            "scenario_seed": eval_seed,
            "scenario_seeds": [eval_seed],
            "episodes": 1,
            "scenario_contract": self.scenario_record,
            "runtime_environment": self.runtime_record,
            "git": self.git_record,
            "launch_argv": {
                "sha256": canonical_json_sha256(argv),
                "argv": argv,
            },
            "command_contract": external_command_contract(
                config=CONFIG,
                variant=method,
                seed=eval_seed,
                episodes=1,
                direct_scenario_seed=True,
                worker_threads=1,
                route_headway_target_s=None,
            ),
            "diagnostics_artifact": csv_artifact_record(
                diagnostics, frame, V6_DIAGNOSTICS_PRIMARY_KEY),
        }
        self.assertEqual(
            manifest["command_contract"]["payload"]["version"],
            V6_COMMAND_CONTRACT_VERSION,
        )
        (run_dir / EXTERNAL_BASELINE_MANIFEST).write_text(
            json.dumps(manifest, indent=2) + "\n")
        return run_dir

    def write_comparison_package(
        self,
        root: Path,
        *,
        methods: tuple[str, ...] = BASELINE_VARIANTS,
        learned_source: str | None = None,
        baseline_source: str | None = None,
        baseline_scenario: dict | None = None,
    ) -> tuple[Path, Path, Path, Path]:
        train_seeds = [1, 2]
        eval_seeds = [11, 12]
        learned_source = learned_source or self.core_source["sha256"]
        baseline_source = baseline_source or learned_source
        scenario = self.scenario_record
        baseline_contract = baseline_scenario or scenario
        learned_rows = []
        for train_seed in train_seeds:
            for eval_seed in eval_seeds:
                learned_rows.append({
                    "config": CONFIG,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "scenario_tape_id": f"tape-{eval_seed}",
                    "scenario_contract_sha256": scenario["sha256"],
                    "protocol_version": V6_PROTOCOL_VERSION,
                    **{metric: 0.8 for metric in V6_METRICS},
                })
        baseline_rows = []
        for method in methods:
            for eval_seed in eval_seeds:
                baseline_rows.append({
                    "config": CONFIG,
                    "method": method,
                    "eval_seed": eval_seed,
                    "scenario_tape_id": f"tape-{eval_seed}",
                    "scenario_contract_sha256": baseline_contract["sha256"],
                    "protocol_version": V6_PROTOCOL_VERSION,
                    **{metric: 1.0 for metric in V6_METRICS},
                })
        learned = pd.DataFrame(learned_rows)
        baseline = pd.DataFrame(baseline_rows)
        learned_path = root / "frozen_per_eval.csv"
        baseline_path = root / "external_baselines_per_seed.csv"
        learned.to_csv(learned_path, index=False)
        baseline.to_csv(baseline_path, index=False)
        config_record = {"sha256": "d" * 64, "lineage": [CONFIG]}
        learned_manifest = {
            "manifest_version": MATRIX_MANIFEST_VERSION,
            "protocol_version": V6_PROTOCOL_VERSION,
            "strict_complete": True,
            "common_random_numbers_verified": True,
            "run_manifests_verified": True,
            "run_source_fingerprint": {"sha256": learned_source},
            "run_git_provenance": self.git_record,
            "git": self.git_record,
            "configs": [CONFIG],
            "train_seeds": train_seeds,
            "eval_seeds": eval_seeds,
            "config_fingerprints": {CONFIG: config_record},
            "scenario_contract": scenario,
            "artifacts": {
                "frozen_per_eval.csv": csv_artifact_record(
                    learned_path, learned, V6_LEARNED_PRIMARY_KEY),
            },
        }
        baseline_manifest = {
            "manifest_version": V6_SUMMARY_MANIFEST_VERSION,
            "protocol_version": V6_PROTOCOL_VERSION,
            "protocol_versions": [V6_PROTOCOL_VERSION],
            "strict_complete": True,
            "direct_scenario_seeds": True,
            "run_manifests_verified": True,
            "core_source_sha256": baseline_source,
            "evaluator_source_sha256": self.evaluator_source["sha256"],
            "core_source_fingerprint": {"sha256": baseline_source},
            "evaluator_source_fingerprint": self.evaluator_source,
            "run_git_provenance": self.git_record,
            "git": self.git_record,
            "configs": [CONFIG],
            "methods": list(methods),
            "eval_seeds": eval_seeds,
            "config_fingerprints": {CONFIG: config_record},
            "scenario_contract_fingerprints": {
                CONFIG: baseline_contract},
            "artifacts": {
                "external_baselines_per_seed.csv": csv_artifact_record(
                    baseline_path, baseline, V6_PER_SEED_PRIMARY_KEY),
            },
        }
        learned_manifest_path = root / "matrix_manifest.json"
        baseline_manifest_path = root / "external_baselines_summary.json"
        learned_manifest_path.write_text(
            json.dumps(learned_manifest, indent=2) + "\n")
        baseline_manifest_path.write_text(
            json.dumps(baseline_manifest, indent=2) + "\n")
        return (
            learned_path,
            baseline_path,
            learned_manifest_path,
            baseline_manifest_path,
        )

    def test_v6_aggregate_binds_inputs_and_outputs_and_rejects_tamper(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            for method in BASELINE_VARIANTS:
                self.write_external_run(logs, method, 11)
            out = root / "aggregate"
            aggregate(
                logs,
                out,
                1,
                configs=[CONFIG],
                variants=list(BASELINE_VARIANTS),
                eval_seeds=[11],
            )
            manifest = json.loads(
                (out / "external_baselines_summary.json").read_text())
            self.assertTrue(manifest["strict_complete"])
            self.assertEqual(len(manifest["input_diagnostics"]), 3)
            self.assertEqual(
                manifest["run_git_provenance"]["commit"],
                manifest["git"]["commit"],
            )
            validate_csv_artifact(
                out / "external_baselines_per_seed.csv",
                manifest["artifacts"]["external_baselines_per_seed.csv"],
                expected_primary_key=V6_PER_SEED_PRIMARY_KEY,
            )

            victim = next(logs.glob("*/diagnostics.csv"))
            tampered = pd.read_csv(victim)
            tampered.loc[0, "composite"] = 99.0
            tampered.to_csv(victim, index=False)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                aggregate(
                    logs,
                    root / "tampered",
                    1,
                    configs=[CONFIG],
                    variants=list(BASELINE_VARIANTS),
                    eval_seeds=[11],
                )

    def test_v6_direct_run_rejects_multiple_eval_rows(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.diagnostic_row("fixed_headway", 11)
            second = self.diagnostic_row("fixed_headway", 12)
            run_dir = self.write_external_run(
                root, "fixed_headway", 11, rows=[first, second])
            with self.assertRaisesRegex(ValueError, "exactly one unique eval seed"):
                validate_direct_baseline_run(
                    run_dir, pd.read_csv(run_dir / "diagnostics.csv"))

    def test_v6_run_requires_direct_seed_mode_before_creating_output(self):
        with TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            with self.assertRaisesRegex(ValueError, "requires direct"):
                run_one(
                    config=CONFIG,
                    variant="fixed_headway",
                    seed=11,
                    episodes=1,
                    logs_dir=logs,
                    direct_scenario_seed=False,
                )
            self.assertFalse(logs.exists())

    def test_v6_direct_run_requires_execution_metrics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.diagnostic_row("fixed_headway", 11)
            row.pop("terminal_dispatch_execution_error_abs_mean_s")
            run_dir = self.write_external_run(
                root, "fixed_headway", 11, rows=[row])
            with self.assertRaisesRegex(ValueError, "missing frozen safety"):
                validate_direct_baseline_run(
                    run_dir, pd.read_csv(run_dir / "diagnostics.csv"))

    def test_v6_direct_run_rejects_missing_manifest(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            frame = pd.DataFrame([
                self.diagnostic_row("fixed_headway", 11)])
            frame.to_csv(run_dir / "diagnostics.csv", index=False)
            with self.assertRaisesRegex(ValueError, "missing external_baseline"):
                validate_direct_baseline_run(run_dir, frame)

    def test_v6_aggregate_rejects_incomplete_method_grid(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            for method in ("fixed_headway", "rule_holding"):
                self.write_external_run(logs, method, 11)
            with self.assertRaisesRegex(ValueError, "methods do not match"):
                aggregate(
                    logs,
                    root / "out",
                    1,
                    configs=[CONFIG],
                    variants=list(BASELINE_VARIANTS),
                    eval_seeds=[11],
                )

    def test_v6_compare_accepts_locked_artifacts_and_binds_outputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned, baseline, _, _ = self.write_comparison_package(root)
            out = root / "comparison"
            compare(learned, baseline, out, learned_config=CONFIG)
            manifest = json.loads(
                (out / "learned_vs_external_manifest.json").read_text())
            self.assertEqual(
                manifest["baseline_methods"], list(BASELINE_VARIANTS))
            self.assertEqual(
                set(manifest["artifacts"]), {
                    "learned_vs_external_per_pair.csv",
                    "learned_vs_external_summary.csv",
                })

    def test_v6_compare_rejects_tampered_csv(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned, baseline, _, _ = self.write_comparison_package(root)
            frame = pd.read_csv(baseline)
            frame.loc[0, "service_cost_restricted"] = 99.0
            frame.to_csv(baseline, index=False)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                compare(learned, baseline, root / "out", learned_config=CONFIG)

    def test_v6_compare_rejects_missing_method(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned, baseline, _, _ = self.write_comparison_package(
                root, methods=("fixed_headway", "rule_holding"))
            with self.assertRaisesRegex(ValueError, "exactly fixed_headway"):
                compare(learned, baseline, root / "out", learned_config=CONFIG)

    def test_v6_compare_rejects_extra_method(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned, baseline, _, _ = self.write_comparison_package(
                root,
                methods=BASELINE_VARIANTS + ("fixed_headway_330",),
            )
            with self.assertRaisesRegex(ValueError, "exactly fixed_headway"):
                compare(learned, baseline, root / "out", learned_config=CONFIG)

    def test_v6_compare_rejects_wrong_scenario_contract(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned, baseline, _, _ = self.write_comparison_package(
                root,
                baseline_scenario=fingerprint({"scenario": "different"}),
            )
            with self.assertRaisesRegex(ValueError, "scenario contract"):
                compare(learned, baseline, root / "out", learned_config=CONFIG)

    def test_v6_compare_rejects_wrong_source(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned, baseline, _, _ = self.write_comparison_package(
                root, learned_source="a" * 64, baseline_source="c" * 64)
            with self.assertRaisesRegex(ValueError, "fingerprints differ"):
                compare(learned, baseline, root / "out", learned_config=CONFIG)

    def test_v6_compare_rejects_mismatched_git_commit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned, baseline, _, baseline_manifest_path = (
                self.write_comparison_package(root))
            manifest = json.loads(baseline_manifest_path.read_text())
            manifest["run_git_provenance"]["commit"] = "b" * 40
            baseline_manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Git provenance differs"):
                compare(learned, baseline, root / "out", learned_config=CONFIG)


if __name__ == "__main__":
    unittest.main()
