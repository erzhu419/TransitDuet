#!/usr/bin/env python3
"""Audit the preregistered V33 timescale-stability development matrix."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_protocol_v6_incremental_selection import (  # noqa: E402
    DEFAULT_MAIN,
    DEFAULT_REFERENCE,
    evaluate_selection,
    sha256_file,
)
from scripts.run_freqduet_protocol_v2_matrix import resolved_config  # noqa: E402


GATE_VERSION = "freqduet-v33-timescale-stability-development-v1"
CONFIRMATION_GATE_VERSION = (
    "freqduet-v33-timescale-stability-confirmation-v1")
RUN_NAME = "protocol_v6_v33_timescale_ep200_s4_e4"
CONFIRMATION_RUN_NAME = "protocol_v6_v33_timescale_confirm_ep200_s8_e8"
MATCHED_CONTEXT = "F_freqduet_protocol_v6_avlcompact_hiro"
CURRENT_MAIN = "F_freqduet_protocol_v6_confirmed_main_hiro"
TRAIN_EPISODES = 200
CHECKPOINT_EP = 199

# Priority favors the latest lower-only freeze, then earlier lower-only freezes.
# Upper-only and joint freezes remain causal mechanism controls.
CANDIDATE_CONTRACTS = {
    "F_freqduet_protocol_v6_v33_lowerfreeze100_hiro": {
        "freeze_lower_policy_after_ep": 100,
        "freeze_lower_critic_after_ep": -1,
        "freeze_upper_after_ep": -1,
    },
    "F_freqduet_protocol_v6_v33_lowerfreeze80_hiro": {
        "freeze_lower_policy_after_ep": 80,
        "freeze_lower_critic_after_ep": -1,
        "freeze_upper_after_ep": -1,
    },
    "F_freqduet_protocol_v6_v33_lowerfreeze40_hiro": {
        "freeze_lower_policy_after_ep": 40,
        "freeze_lower_critic_after_ep": -1,
        "freeze_upper_after_ep": -1,
    },
    "F_freqduet_protocol_v6_v33_upperfreeze100_hiro": {
        "freeze_lower_policy_after_ep": -1,
        "freeze_lower_critic_after_ep": -1,
        "freeze_upper_after_ep": 100,
    },
    "F_freqduet_protocol_v6_v33_bothfreeze100_hiro": {
        "freeze_lower_policy_after_ep": 100,
        "freeze_lower_critic_after_ep": -1,
        "freeze_upper_after_ep": 100,
    },
}
CANDIDATES = list(CANDIDATE_CONTRACTS)
EXPECTED_CONFIGS = [
    DEFAULT_MAIN,
    DEFAULT_REFERENCE,
    MATCHED_CONTEXT,
    CURRENT_MAIN,
    *CANDIDATES,
]

DISCOVERY_TRAIN_SEEDS = [37013, 37031, 37057, 37081]
DISCOVERY_EVAL_SEEDS = [72011, 72029, 72047, 72071]

# Frozen before the development matrix is opened. Only one selected candidate
# may use this roster, and only after an unchanged V33 development pass.
CONFIRMATION_TRAIN_SEEDS = [
    38013, 38037, 38057, 38081, 38101, 38119, 38143, 38167,
]
CONFIRMATION_EVAL_SEEDS = [
    73011, 73029, 73047, 73071, 73101, 73119, 73143, 73167,
]

MAX_JOURNEY_CI_HIGH_MIN = 0.15
MIN_NEGATIVE_TRAIN_SEED_FRACTION = 0.75


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _without_timescale_contract(config: dict[str, object]) -> dict[str, object]:
    payload = deepcopy(config)
    payload.pop("_name", None)
    protocol = payload.get("protocol", {}) or {}
    protocol.pop("role", None)
    training = payload.get("training", {}) or {}
    training.pop("longtrain_stability", None)
    return payload


def validate_candidate_contracts() -> dict[str, bool]:
    base = resolved_config(CURRENT_MAIN)
    checks: dict[str, bool] = {}
    for name, expected_stability in CANDIDATE_CONTRACTS.items():
        candidate = resolved_config(name)
        stability = (candidate.get("training", {}) or {}).get(
            "longtrain_stability", {}) or {}
        role = str((candidate.get("protocol", {}) or {}).get("role", ""))
        checks[f"{name}:exact_freeze_schedule"] = (
            stability == expected_stability)
        checks[f"{name}:lower_critic_keeps_training"] = (
            stability.get("freeze_lower_critic_after_ep") == -1)
        checks[f"{name}:explicit_exploratory_role"] = (
            role.startswith("exploratory_v33_"))
        checks[f"{name}:only_role_and_schedule_change"] = (
            _without_timescale_contract(candidate)
            == _without_timescale_contract(base))
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise ValueError(f"V33 candidate contract failed: {failed}")
    return checks


def _train_seed_direction_fraction(
    per_eval: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
    metric: str,
    expected_train_seeds: list[int],
) -> tuple[float, dict[int, float]]:
    keys = ["train_seed", "eval_seed"]
    candidate_rows = per_eval.loc[
        per_eval["config"] == candidate, [*keys, metric]
    ].rename(columns={metric: "candidate_value"})
    reference_rows = per_eval.loc[
        per_eval["config"] == reference, [*keys, metric]
    ].rename(columns={metric: "reference_value"})
    paired = candidate_rows.merge(
        reference_rows,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    paired["delta"] = (
        pd.to_numeric(paired["candidate_value"], errors="raise")
        - pd.to_numeric(paired["reference_value"], errors="raise")
    )
    by_seed = paired.groupby("train_seed")["delta"].mean()
    if list(by_seed.index.astype(int)) != expected_train_seeds:
        raise ValueError("V33 direction check has an incomplete train-seed grid")
    values = {int(seed): float(value) for seed, value in by_seed.items()}
    return float((by_seed < 0.0).mean()), values


def _candidate_result(
    aggregate_dir: Path,
    per_eval: pd.DataFrame,
    candidate: str,
    expected_stage: str,
    expected_train_seeds: list[int],
) -> dict[str, object]:
    base = evaluate_selection(
        aggregate_dir,
        candidates=[candidate],
        main=DEFAULT_MAIN,
        reference=DEFAULT_REFERENCE,
        matched_context=MATCHED_CONTEXT,
        expected_stage=expected_stage,
    )
    result = base["candidate_results"][0]
    direction_fraction, direction_values = _train_seed_direction_fraction(
        per_eval,
        candidate=candidate,
        reference=DEFAULT_REFERENCE,
        metric="headway_cv",
        expected_train_seeds=expected_train_seeds,
    )
    longtrain_gates = {
        "v8_effect_and_mechanism_gate_passes": base["status"] == "unique_pass",
        "headway_cv_ci_excludes_zero": (
            float(result["headway_cv_delta_ci_high"]) < 0.0),
        "journey_ci_is_noninferior": (
            float(result["journey_delta_ci_high"])
            <= MAX_JOURNEY_CI_HIGH_MIN),
        "headway_cv_direction_consistent": (
            direction_fraction >= MIN_NEGATIVE_TRAIN_SEED_FRACTION),
    }
    return {
        "candidate": candidate,
        "passes": bool(all(longtrain_gates.values())),
        "longtrain_gates": longtrain_gates,
        "headway_cv_negative_train_seed_fraction": direction_fraction,
        "headway_cv_delta_by_train_seed": direction_values,
        "base_selection_result": base,
    }


def evaluate_timescale_screen(aggregate_dir: Path) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    manifest_path = aggregate_dir / "matrix_manifest.json"
    per_eval_path = aggregate_dir / "frozen_per_eval.csv"
    manifest = _load_json(manifest_path)
    per_eval = pd.read_csv(per_eval_path)
    expected_rollouts = (
        len(EXPECTED_CONFIGS)
        * len(DISCOVERY_TRAIN_SEEDS)
        * len(DISCOVERY_EVAL_SEEDS)
    )
    strict_checks = {
        "strict_complete": manifest.get("strict_complete") is True,
        "common_random_numbers_verified": (
            manifest.get("common_random_numbers_verified") is True),
        "run_manifests_verified": (
            manifest.get("run_manifests_verified") is True),
        "development_stage_is_not_confirmation": (
            manifest.get("stage") == "exploratory"
            and manifest.get("independent_confirmation") is False),
        "exact_configs": manifest.get("configs") == EXPECTED_CONFIGS,
        "exact_train_seeds": (
            manifest.get("train_seeds") == DISCOVERY_TRAIN_SEEDS),
        "exact_eval_seeds": (
            manifest.get("eval_seeds") == DISCOVERY_EVAL_SEEDS),
        "exact_200_episode_checkpoint": (
            manifest.get("train_episodes") == TRAIN_EPISODES
            and manifest.get("checkpoint_ep") == CHECKPOINT_EP),
        "expected_rollout_count": (
            manifest.get("expected_rollouts") == expected_rollouts
            and len(per_eval) == expected_rollouts),
        "clean_identified_source": (
            (manifest.get("run_git_provenance", {}) or {}).get(
                "tracked_dirty") is False
            and len(str((manifest.get("run_git_provenance", {}) or {}).get(
                "commit", ""))) == 40),
    }
    if not all(strict_checks.values()):
        raise ValueError(f"V33 strict checks failed: {strict_checks}")
    contract_checks = validate_candidate_contracts()

    unfrozen_control = evaluate_selection(
        aggregate_dir,
        candidates=[CURRENT_MAIN],
        main=DEFAULT_MAIN,
        reference=DEFAULT_REFERENCE,
        matched_context=MATCHED_CONTEXT,
        expected_stage="exploratory",
    )
    candidate_results = [
        _candidate_result(
            aggregate_dir,
            per_eval,
            candidate,
            expected_stage="exploratory",
            expected_train_seeds=DISCOVERY_TRAIN_SEEDS,
        )
        for candidate in CANDIDATES
    ]
    passing = [
        item["candidate"] for item in candidate_results if item["passes"]
    ]
    selected = next((name for name in CANDIDATES if name in passing), None)
    return {
        "gate_version": GATE_VERSION,
        "status": (
            "development_candidate_selected" if selected else "no_pass"),
        "claim_eligible": False,
        "confirmation_authorized": selected is not None,
        "selected_for_confirmation": selected,
        "passing_candidates": passing,
        "candidate_priority": CANDIDATES,
        "candidate_contracts": CANDIDATE_CONTRACTS,
        "strict_checks": strict_checks,
        "contract_checks": contract_checks,
        "thresholds": {
            "max_journey_ci_high_min": MAX_JOURNEY_CI_HIGH_MIN,
            "min_negative_train_seed_fraction": (
                MIN_NEGATIVE_TRAIN_SEED_FRACTION),
            "base_gate": "unchanged V8 effect and mechanism thresholds",
        },
        "development_design": {
            "run_name": RUN_NAME,
            "configs": EXPECTED_CONFIGS,
            "train_seeds": DISCOVERY_TRAIN_SEEDS,
            "eval_seeds": DISCOVERY_EVAL_SEEDS,
            "train_episodes": TRAIN_EPISODES,
            "checkpoint_ep": CHECKPOINT_EP,
        },
        "confirmation_design": {
            "train_seeds": CONFIRMATION_TRAIN_SEEDS,
            "eval_seeds": CONFIRMATION_EVAL_SEEDS,
            "train_episodes": TRAIN_EPISODES,
            "checkpoint_ep": CHECKPOINT_EP,
            "single_use": True,
        },
        "unfrozen_control_result": unfrozen_control,
        "candidate_results": candidate_results,
        "input_artifacts": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "per_eval": {
                "path": str(per_eval_path),
                "sha256": sha256_file(per_eval_path),
            },
        },
    }


def confirmation_configs(candidate: str) -> list[str]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unregistered V33 candidate: {candidate}")
    return [
        DEFAULT_MAIN,
        DEFAULT_REFERENCE,
        MATCHED_CONTEXT,
        CURRENT_MAIN,
        candidate,
    ]


def validate_development_authorization(
    gate: dict[str, object],
) -> str:
    selected = gate.get("selected_for_confirmation")
    passing = gate.get("passing_candidates")
    results = gate.get("candidate_results")
    if not isinstance(selected, str) or selected not in CANDIDATES:
        raise ValueError("V33 development gate has no registered selection")
    if not isinstance(passing, list) or not isinstance(results, list):
        raise ValueError("V33 development gate lacks candidate evidence")
    first_passing = next(
        (candidate for candidate in CANDIDATES if candidate in passing), None)
    selected_rows = [
        row for row in results
        if isinstance(row, dict) and row.get("candidate") == selected
    ]
    checks = {
        "gate_version": gate.get("gate_version") == GATE_VERSION,
        "development_selected": (
            gate.get("status") == "development_candidate_selected"),
        "not_claim_eligible": gate.get("claim_eligible") is False,
        "confirmation_authorized": (
            gate.get("confirmation_authorized") is True),
        "priority_is_frozen": gate.get("candidate_priority") == CANDIDATES,
        "selected_is_first_passing": selected == first_passing,
        "selected_result_passes": (
            len(selected_rows) == 1
            and selected_rows[0].get("passes") is True),
        "confirmation_roster_is_frozen": (
            (gate.get("confirmation_design") or {}).get("train_seeds")
            == CONFIRMATION_TRAIN_SEEDS
            and (gate.get("confirmation_design") or {}).get("eval_seeds")
            == CONFIRMATION_EVAL_SEEDS
            and (gate.get("confirmation_design") or {}).get("train_episodes")
            == TRAIN_EPISODES
            and (gate.get("confirmation_design") or {}).get("checkpoint_ep")
            == CHECKPOINT_EP
            and (gate.get("confirmation_design") or {}).get("single_use")
            is True
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            f"V33 development authorization failed: {checks}")
    return selected


def evaluate_timescale_confirmation(
    aggregate_dir: Path,
    *,
    development_dir: Path,
) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    development_dir = Path(development_dir).resolve()
    paths = {
        "manifest": aggregate_dir / "matrix_manifest.json",
        "per_eval": aggregate_dir / "frozen_per_eval.csv",
        "development_manifest": development_dir / "matrix_manifest.json",
        "development_gate": development_dir / "v33_timescale_gate.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing V33 confirmation artifacts: {missing}")

    manifest = _load_json(paths["manifest"])
    per_eval = pd.read_csv(paths["per_eval"])
    development_manifest = _load_json(paths["development_manifest"])
    development_gate = _load_json(paths["development_gate"])
    selected = validate_development_authorization(development_gate)
    configs = confirmation_configs(selected)
    expected_rollouts = (
        len(configs)
        * len(CONFIRMATION_TRAIN_SEEDS)
        * len(CONFIRMATION_EVAL_SEEDS)
    )
    development_source = (
        development_manifest.get("run_source_fingerprint", {}) or {}).get(
            "sha256")
    confirmation_source = (
        manifest.get("run_source_fingerprint", {}) or {}).get("sha256")
    development_commit = (
        development_manifest.get("run_git_provenance", {}) or {}).get(
            "commit")
    confirmation_commit = (
        manifest.get("run_git_provenance", {}) or {}).get("commit")
    strict_checks = {
        "development_manifest_hash_verified": (
            sha256_file(paths["development_manifest"])
            == (development_gate.get("input_artifacts", {}) or {}).get(
                "manifest", {}).get("sha256")),
        "development_matrix_is_exact": (
            development_manifest.get("strict_complete") is True
            and development_manifest.get("stage") == "exploratory"
            and development_manifest.get("independent_confirmation") is False
            and development_manifest.get("configs") == EXPECTED_CONFIGS
            and development_manifest.get("train_seeds")
            == DISCOVERY_TRAIN_SEEDS
            and development_manifest.get("eval_seeds")
            == DISCOVERY_EVAL_SEEDS
            and development_manifest.get("train_episodes") == TRAIN_EPISODES
            and development_manifest.get("checkpoint_ep") == CHECKPOINT_EP
        ),
        "strict_complete": manifest.get("strict_complete") is True,
        "common_random_numbers_verified": (
            manifest.get("common_random_numbers_verified") is True),
        "run_manifests_verified": (
            manifest.get("run_manifests_verified") is True),
        "independent_confirmation_stage": (
            manifest.get("stage") == "confirmation"
            and manifest.get("independent_confirmation") is True),
        "exact_configs": manifest.get("configs") == configs,
        "exact_train_seeds": (
            manifest.get("train_seeds") == CONFIRMATION_TRAIN_SEEDS),
        "exact_eval_seeds": (
            manifest.get("eval_seeds") == CONFIRMATION_EVAL_SEEDS),
        "seeds_disjoint_from_development": (
            not (set(CONFIRMATION_TRAIN_SEEDS)
                 & set(DISCOVERY_TRAIN_SEEDS))
            and not (set(CONFIRMATION_EVAL_SEEDS)
                     & set(DISCOVERY_EVAL_SEEDS))),
        "exact_200_episode_checkpoint": (
            manifest.get("train_episodes") == TRAIN_EPISODES
            and manifest.get("checkpoint_ep") == CHECKPOINT_EP),
        "reference_is_noguard": (
            manifest.get("reference") == DEFAULT_REFERENCE),
        "expected_rollout_count": (
            manifest.get("expected_rollouts") == expected_rollouts
            and len(per_eval) == expected_rollouts),
        "unique_rollout_keys": not per_eval.duplicated(
            ["config", "train_seed", "eval_seed"]).any(),
        "model_source_unchanged": (
            isinstance(development_source, str)
            and len(development_source) == 64
            and confirmation_source == development_source),
        "scenario_contract_unchanged": (
            (manifest.get("scenario_contract", {}) or {}).get("sha256")
            == (development_manifest.get("scenario_contract", {}) or {}).get(
                "sha256")),
        "analysis_source_unchanged": (
            manifest.get("launch_analysis_sha256")
            == development_manifest.get("launch_analysis_sha256")),
        "git_commit_unchanged": (
            isinstance(development_commit, str)
            and len(development_commit) == 40
            and confirmation_commit == development_commit),
        "source_is_clean": (
            (development_manifest.get("run_git_provenance", {}) or {}).get(
                "tracked_dirty") is False
            and (manifest.get("run_git_provenance", {}) or {}).get(
                "tracked_dirty") is False),
    }
    if not all(strict_checks.values()):
        raise ValueError(
            f"V33 confirmation strict checks failed: {strict_checks}")
    contract_checks = validate_candidate_contracts()

    candidate_result = _candidate_result(
        aggregate_dir,
        per_eval,
        selected,
        expected_stage="confirmation",
        expected_train_seeds=CONFIRMATION_TRAIN_SEEDS,
    )
    unfrozen_control = evaluate_selection(
        aggregate_dir,
        candidates=[CURRENT_MAIN],
        main=DEFAULT_MAIN,
        reference=DEFAULT_REFERENCE,
        matched_context=MATCHED_CONTEXT,
        expected_stage="confirmation",
    )
    confirmed = bool(candidate_result["passes"])
    return {
        "gate_version": CONFIRMATION_GATE_VERSION,
        "status": (
            "timescale_stability_confirmed"
            if confirmed else "timescale_stability_not_confirmed"),
        "confirmation_claim_eligible": confirmed,
        "selected_candidate": selected,
        "strict_checks": strict_checks,
        "contract_checks": contract_checks,
        "thresholds": {
            "max_journey_ci_high_min": MAX_JOURNEY_CI_HIGH_MIN,
            "min_negative_train_seed_fraction": (
                MIN_NEGATIVE_TRAIN_SEED_FRACTION),
            "base_gate": "unchanged V8 effect and mechanism thresholds",
        },
        "confirmation_design": {
            "run_name": CONFIRMATION_RUN_NAME,
            "configs": configs,
            "train_seeds": CONFIRMATION_TRAIN_SEEDS,
            "eval_seeds": CONFIRMATION_EVAL_SEEDS,
            "train_episodes": TRAIN_EPISODES,
            "checkpoint_ep": CHECKPOINT_EP,
        },
        "candidate_result": candidate_result,
        "unfrozen_control_result": unfrozen_control,
        "development_lineage": {
            "directory": str(development_dir),
            "gate_sha256": sha256_file(paths["development_gate"]),
            "manifest_sha256": sha256_file(paths["development_manifest"]),
            "source_commit": development_commit,
        },
        "input_artifacts": {
            "manifest": {
                "path": str(paths["manifest"]),
                "sha256": sha256_file(paths["manifest"]),
            },
            "per_eval": {
                "path": str(paths["per_eval"]),
                "sha256": sha256_file(paths["per_eval"]),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--development-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.development_dir is None:
        result = evaluate_timescale_screen(args.aggregate_dir)
        passed = result["confirmation_authorized"]
        default_name = "v33_timescale_gate.json"
    else:
        result = evaluate_timescale_confirmation(
            args.aggregate_dir,
            development_dir=args.development_dir,
        )
        passed = result["confirmation_claim_eligible"]
        default_name = "v33_timescale_confirmation.json"
    out = args.out or Path(args.aggregate_dir) / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
