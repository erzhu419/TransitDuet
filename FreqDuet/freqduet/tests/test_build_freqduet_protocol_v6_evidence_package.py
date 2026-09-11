import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.build_freqduet_protocol_v6_evidence_package import (
    CONFIRMED_SOURCE_CONFIG,
    NOGUARD_REFERENCE,
    PAPER_CONTROLLER,
    V8_EVAL_SEEDS,
    V8_SOURCE_COMMIT,
    V8_TRAIN_SEEDS,
    V9_EVAL_SEEDS,
    V9_SOURCE_COMMIT,
    V9_TRAIN_SEEDS,
    V33_CANDIDATE_CONTRACTS,
    V33_CANDIDATES,
    V33_CONFIGS,
    V33_CONFIRM_EVAL_SEEDS,
    V33_CONFIRM_TRAIN_SEEDS,
    V33_EVAL_SEEDS,
    V33_GATE_VERSION,
    V33_RUN_NAME,
    V33_SOURCE_COMMIT,
    V33_TRAIN_SEEDS,
    build_package,
    config_fingerprint,
    sha256_file,
)


PAIR_FIELDS = [
    "restricted_total_journey_horizon_min",
    "restricted_wait_horizon_min",
    "restricted_in_vehicle_horizon_min",
    "headway_cv",
    "passenger_unserved_rate",
    "holding_vehicle_seconds_per_launched_trip",
    "fleet_denied_trip_rate",
    "service_cost_restricted",
]
def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n")


def pair_row(candidate: str, n_pairs: int) -> dict[str, str]:
    row = {
        "candidate": candidate,
        "reference": NOGUARD_REFERENCE,
        "n_pairs": str(n_pairs),
    }
    for index, metric in enumerate(PAIR_FIELDS, start=1):
        row[f"delta_{metric}_mean"] = str(-0.01 * index)
        row[f"delta_{metric}_ci_low"] = str(-0.02 * index)
        row[f"delta_{metric}_ci_high"] = "0.001"
        row[f"delta_{metric}_signflip_p"] = "0.03125"
        row[f"delta_{metric}_signflip_p_holm"] = "0.0625"
    return row


def write_pair_csv(path: Path, candidate: str, n_pairs: int) -> None:
    write_pair_rows(path, [pair_row(candidate, n_pairs)])


def write_pair_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_external_csv(path: Path) -> None:
    rows = []
    for method in ("fixed_headway", "rule_holding", "rule_mpc"):
        row = {
            "learned_config": PAPER_CONTROLLER,
            "baseline_method": method,
            "n_pairs": "64",
        }
        for index, metric in enumerate(PAIR_FIELDS, start=1):
            row[f"{metric}_learned_mean"] = "1.0"
            row[f"{metric}_baseline_mean"] = "2.0"
            row[f"delta_{metric}_mean"] = str(-0.1 * index)
            row[f"delta_{metric}_ci_low"] = str(-0.2 * index)
            row[f"delta_{metric}_ci_high"] = "-0.01"
            row[f"delta_{metric}_signflip_p"] = "0.01"
            row[f"delta_{metric}_signflip_p_holm"] = "0.03"
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ProtocolV6EvidencePackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.v8 = self.root / "v8"
        self.v9 = self.root / "v9"
        self.v33 = self.root / "v33"
        self.external = self.root / "external"
        self.config_root = self.root / "config_root"
        for path in (self.v8, self.v9, self.v33, self.external):
            path.mkdir()
        configs = self.config_root / "configs_freqduet"
        configs.mkdir(parents=True)
        (self.config_root / "config_v2.yaml").write_text("_name: base\n")
        config_chain = (
            ("F_freqduet_protocol_v6_main_hiro", "../config_v2.yaml"),
            (NOGUARD_REFERENCE, "F_freqduet_protocol_v6_main_hiro.yaml"),
            ("F_freqduet_protocol_v6_avlcompact_hiro", f"{NOGUARD_REFERENCE}.yaml"),
            (
                CONFIRMED_SOURCE_CONFIG,
                "F_freqduet_protocol_v6_avlcompact_hiro.yaml",
            ),
            (PAPER_CONTROLLER, f"{CONFIRMED_SOURCE_CONFIG}.yaml"),
            *(
                (candidate, f"{PAPER_CONTROLLER}.yaml")
                for candidate in V33_CANDIDATES
            ),
        )
        for name, parent in config_chain:
            (configs / f"{name}.yaml").write_text(
                f"_extends: {parent}\n_name: {name}\n"
            )

        write_json(self.v8 / "confirmation_gate.json", {
            "primary": CONFIRMED_SOURCE_CONFIG,
            "primary_claim_eligible": True,
            "primary_result": {"status": "unique_pass"},
        })
        write_pair_csv(
            self.v8 / "frozen_paired_deltas.csv",
            CONFIRMED_SOURCE_CONFIG,
            24,
        )

        write_json(self.v9 / "confirmed_longtrain_gate.json", {
            "candidate": PAPER_CONTROLLER,
            "status": "longtrain_not_confirmed",
            "longtrain_claim_eligible": False,
            "base_confirmation_result": {
                "matrix_provenance": {
                    "run_git_provenance": {"commit": V9_SOURCE_COMMIT}
                }
            },
        })
        write_pair_csv(self.v9 / "frozen_paired_deltas.csv", PAPER_CONTROLLER, 64)
        write_pair_rows(
            self.v33 / "frozen_paired_deltas.csv",
            [pair_row(candidate, 16) for candidate in V33_CANDIDATES],
        )
        write_external_csv(self.external / "learned_vs_external_summary.csv")

        for directory, names in (
            (self.v8, ("frozen_per_eval.csv", "frozen_summary.csv")),
            (self.v9, ("frozen_per_eval.csv", "frozen_summary.csv")),
            (self.v33, ("frozen_per_eval.csv", "frozen_summary.csv")),
            (self.external, (
                "learned_vs_external_per_pair.csv",
                "external_baselines_per_seed.csv",
                "external_baselines_summary.csv",
            )),
        ):
            for name in names:
                (directory / name).write_text("value\n1\n")
        write_json(self.external / "external_baselines_summary.json", {})

        def artifact_records(directory: Path) -> dict:
            return {
                name: {"sha256": sha256_file(directory / name)}
                for name in (
                    "frozen_per_eval.csv",
                    "frozen_summary.csv",
                    "frozen_paired_deltas.csv",
                )
            }

        main_fingerprint = config_fingerprint(
            "F_freqduet_protocol_v6_main_hiro", self.config_root
        )
        noguard_fingerprint = config_fingerprint(
            NOGUARD_REFERENCE, self.config_root
        )
        avl_fingerprint = config_fingerprint(
            "F_freqduet_protocol_v6_avlcompact_hiro", self.config_root
        )
        source_fingerprint = config_fingerprint(
            CONFIRMED_SOURCE_CONFIG, self.config_root
        )
        controller_fingerprint = config_fingerprint(
            PAPER_CONTROLLER, self.config_root
        )
        v33_fingerprints = {
            candidate: config_fingerprint(candidate, self.config_root)
            for candidate in V33_CANDIDATES
        }
        scenario_sha = "c" * 64
        source_sha = "d" * 64
        write_json(self.v8 / "matrix_manifest.json", {
            "manifest_version": "freqduet-matrix-manifest-v2",
            "protocol_version": "freqduet-eval-v6",
            "stage": "confirmation",
            "independent_confirmation": True,
            "reference": NOGUARD_REFERENCE,
            "strict_complete": True,
            "common_random_numbers_verified": True,
            "run_git_provenance": {
                "commit": V8_SOURCE_COMMIT,
                "tracked_dirty": False,
            },
            "train_episodes": 40,
            "checkpoint_ep": 39,
            "train_seeds": list(V8_TRAIN_SEEDS),
            "eval_seeds": list(V8_EVAL_SEEDS),
            "configs": [
                "F_freqduet_protocol_v6_main_hiro",
                NOGUARD_REFERENCE,
                CONFIRMED_SOURCE_CONFIG,
            ],
            "expected_rollouts": 72,
            "config_fingerprints": {
                "F_freqduet_protocol_v6_main_hiro": main_fingerprint,
                NOGUARD_REFERENCE: noguard_fingerprint,
                CONFIRMED_SOURCE_CONFIG: source_fingerprint,
            },
            "artifacts": artifact_records(self.v8),
        })
        write_json(self.v9 / "matrix_manifest.json", {
            "manifest_version": "freqduet-matrix-manifest-v2",
            "protocol_version": "freqduet-eval-v6",
            "stage": "confirmation",
            "independent_confirmation": True,
            "reference": NOGUARD_REFERENCE,
            "strict_complete": True,
            "common_random_numbers_verified": True,
            "run_git_provenance": {
                "commit": V9_SOURCE_COMMIT,
                "tracked_dirty": False,
            },
            "train_episodes": 200,
            "checkpoint_ep": 199,
            "train_seeds": list(V9_TRAIN_SEEDS),
            "eval_seeds": list(V9_EVAL_SEEDS),
            "configs": [
                "F_freqduet_protocol_v6_main_hiro",
                NOGUARD_REFERENCE,
                "F_freqduet_protocol_v6_avlcompact_hiro",
                PAPER_CONTROLLER,
            ],
            "expected_rollouts": 256,
            "run_source_fingerprint": {"sha256": source_sha},
            "scenario_contract": {"sha256": scenario_sha},
            "config_fingerprints": {
                "F_freqduet_protocol_v6_main_hiro": main_fingerprint,
                NOGUARD_REFERENCE: noguard_fingerprint,
                "F_freqduet_protocol_v6_avlcompact_hiro": avl_fingerprint,
                PAPER_CONTROLLER: controller_fingerprint,
            },
            "artifacts": artifact_records(self.v9),
        })
        write_json(self.v33 / "matrix_manifest.json", {
            "manifest_version": "freqduet-matrix-manifest-v2",
            "protocol_version": "freqduet-eval-v6",
            "stage": "exploratory",
            "independent_confirmation": False,
            "reference": NOGUARD_REFERENCE,
            "strict_complete": True,
            "common_random_numbers_verified": True,
            "run_manifests_verified": True,
            "run_git_provenance": {
                "commit": V33_SOURCE_COMMIT,
                "tracked_dirty": False,
            },
            "train_episodes": 200,
            "checkpoint_ep": 199,
            "train_seeds": list(V33_TRAIN_SEEDS),
            "eval_seeds": list(V33_EVAL_SEEDS),
            "configs": list(V33_CONFIGS),
            "expected_rollouts": 112,
            "config_fingerprints": {
                "F_freqduet_protocol_v6_main_hiro": main_fingerprint,
                NOGUARD_REFERENCE: noguard_fingerprint,
                "F_freqduet_protocol_v6_avlcompact_hiro": avl_fingerprint,
                PAPER_CONTROLLER: controller_fingerprint,
                **v33_fingerprints,
            },
            "artifacts": artifact_records(self.v33),
        })
        write_json(self.v33 / "v33_timescale_gate.json", {
            "gate_version": V33_GATE_VERSION,
            "status": "no_pass",
            "claim_eligible": False,
            "confirmation_authorized": False,
            "selected_for_confirmation": None,
            "passing_candidates": [],
            "candidate_priority": list(V33_CANDIDATES),
            "candidate_contracts": V33_CANDIDATE_CONTRACTS,
            "candidate_results": [
                {
                    "candidate": candidate,
                    "passes": False,
                }
                for candidate in V33_CANDIDATES
            ],
            "strict_checks": {"strict_complete": True},
            "contract_checks": {"candidate_contracts_match": True},
            "development_design": {
                "checkpoint_ep": 199,
                "configs": list(V33_CONFIGS),
                "eval_seeds": list(V33_EVAL_SEEDS),
                "run_name": V33_RUN_NAME,
                "train_episodes": 200,
                "train_seeds": list(V33_TRAIN_SEEDS),
            },
            "confirmation_design": {
                "checkpoint_ep": 199,
                "eval_seeds": list(V33_CONFIRM_EVAL_SEEDS),
                "single_use": True,
                "train_episodes": 200,
                "train_seeds": list(V33_CONFIRM_TRAIN_SEEDS),
            },
            "thresholds": {
                "base_gate": "unchanged V8 effect and mechanism thresholds",
                "max_journey_ci_high_min": 0.15,
                "min_negative_train_seed_fraction": 0.75,
            },
            "input_artifacts": {
                "manifest": {
                    "sha256": sha256_file(self.v33 / "matrix_manifest.json")
                },
                "per_eval": {
                    "sha256": sha256_file(self.v33 / "frozen_per_eval.csv")
                },
            },
            "unfrozen_control_result": {
                "candidate_results": [{
                    "headway_cv_delta_vs_reference": -0.021,
                    "headway_cv_delta_ci_low": -0.045,
                    "headway_cv_delta_ci_high": -0.003,
                }],
            },
        })
        write_json(
            self.external / "learned_vs_external_manifest.json",
            {
                "manifest_version": "freqduet-external-comparison-v6",
                "protocol_version": "freqduet-eval-v6",
                "strict_complete": True,
                "common_random_numbers_verified": True,
                "learned_config": PAPER_CONTROLLER,
                "baseline_config": PAPER_CONTROLLER,
                "baseline_methods": ["fixed_headway", "rule_holding", "rule_mpc"],
                "required_external_method_family": [
                    "fixed_headway",
                    "rule_holding",
                    "rule_mpc",
                ],
                "source_provenance": {
                    "git": {"commit": V9_SOURCE_COMMIT},
                    "core_source_sha256": source_sha,
                    "scenario_contract_sha256": scenario_sha,
                },
                "input_artifacts": {
                    "learned": {
                        "sha256": sha256_file(self.v9 / "frozen_per_eval.csv")
                    },
                    "external": {
                        "sha256": sha256_file(
                            self.external / "external_baselines_per_seed.csv"
                        )
                    },
                },
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_builds_balanced_package_with_failed_longtrain_visible(self) -> None:
        out = self.root / "out"
        manifest = build_package(
            self.v8, self.v9, self.v33, self.external, out, self.config_root
        )

        self.assertFalse(manifest["submission_ready"])
        self.assertEqual(manifest["submission_blocker"], "v9_longtrain_not_confirmed")
        status = json.loads((out / "evidence_status.json").read_text())
        self.assertEqual(status["v8_confirmation_status"], "unique_pass")
        self.assertEqual(status["v9_longtrain_status"], "longtrain_not_confirmed")
        self.assertEqual(status["v33_timescale_status"], "no_pass")
        self.assertFalse(status["v33_confirmation_authorized"])
        results = (out / "manuscript" / "current_best_results.md").read_text()
        self.assertIn("does not outperform fixed headway", results)
        self.assertIn("combined-policy comparison", results)
        self.assertIn("configurations disable the legacy", results)
        self.assertIn("must not be relabelled as submission-ready", results)
        self.assertIn("V33 status is `no_pass`", results)
        self.assertTrue((out / "source_artifacts" / "v8" / "frozen_per_eval.csv").is_file())
        self.assertTrue(
            (out / "source_artifacts" / "v33_timescale" / "v33_timescale_gate.json").is_file()
        )
        self.assertTrue(
            (out / "tables" / "table5_v33_timescale_development.csv").is_file()
        )
        self.assertTrue((out / "README.md").is_file())
        self.assertTrue(
            (out / "configs" / "config_snapshot_manifest.json").is_file()
        )
        self.assertTrue(manifest["config_fingerprints_verified"])

    def test_rejects_nonconfirmed_v8_source(self) -> None:
        gate = json.loads((self.v8 / "confirmation_gate.json").read_text())
        gate["primary_claim_eligible"] = False
        write_json(self.v8 / "confirmation_gate.json", gate)

        with self.assertRaisesRegex(ValueError, "not claim eligible"):
            build_package(
                self.v8,
                self.v9,
                self.v33,
                self.external,
                self.root / "out",
                self.config_root,
            )

    def test_rejects_missing_external_comparator(self) -> None:
        with (self.external / "learned_vs_external_summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        with (self.external / "learned_vs_external_summary.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows[:-1])

        with self.assertRaisesRegex(ValueError, "exactly fixed_headway"):
            build_package(
                self.v8,
                self.v9,
                self.v33,
                self.external,
                self.root / "out",
                self.config_root,
            )

    def test_rejects_config_drift_from_recorded_experiment(self) -> None:
        path = (
            self.config_root
            / "configs_freqduet"
            / f"{PAPER_CONTROLLER}.yaml"
        )
        path.write_text(path.read_text() + "lower:\n  changed: true\n")

        with self.assertRaisesRegex(ValueError, "config fingerprint mismatch"):
            build_package(
                self.v8,
                self.v9,
                self.v33,
                self.external,
                self.root / "out",
                self.config_root,
            )

    def test_rejects_v33_confirmation_authorization_after_no_pass(self) -> None:
        path = self.v33 / "v33_timescale_gate.json"
        gate = json.loads(path.read_text())
        gate["confirmation_authorized"] = True
        write_json(path, gate)

        with self.assertRaisesRegex(ValueError, "confirmation was not authorized"):
            build_package(
                self.v8,
                self.v9,
                self.v33,
                self.external,
                self.root / "out",
                self.config_root,
            )


if __name__ == "__main__":
    unittest.main()
