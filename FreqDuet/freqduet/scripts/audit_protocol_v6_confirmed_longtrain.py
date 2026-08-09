#!/usr/bin/env python3
"""Audit the preregistered V6 confirmed-main long-training matrix."""

from __future__ import annotations

import argparse
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


PARENT_GATE_SHA256 = (
    "a72da292c10a990552db8291a43f92c0c80097217db30153ecfbdee5a1ae8cd6"
)
PARENT_PRIMARY = "F_freqduet_protocol_v6_avlcompact_w2_hiro"
CONFIRMED_MAIN = "F_freqduet_protocol_v6_confirmed_main_hiro"
MATCHED_CONTEXT = "F_freqduet_protocol_v6_avlcompact_hiro"
EXPECTED_CONFIGS = [
    DEFAULT_MAIN,
    DEFAULT_REFERENCE,
    MATCHED_CONTEXT,
    CONFIRMED_MAIN,
]
EXPECTED_TRAIN_SEEDS = [
    12011, 12037, 12049, 12071, 12097, 12109, 12143, 12161,
]
EXPECTED_EVAL_SEEDS = [
    45007, 45013, 45053, 45061, 45077, 45119, 45131, 45137,
]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _train_seed_direction_fraction(
    per_eval: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
    metric: str,
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
    if len(by_seed) != len(EXPECTED_TRAIN_SEEDS):
        raise ValueError(
            "long-training heterogeneity check has incomplete train seeds")
    values = {int(key): float(value) for key, value in by_seed.items()}
    fraction = float((by_seed < 0.0).mean())
    return fraction, values


def evaluate_confirmed_longtrain(
    aggregate_dir: Path,
    *,
    parent_gate_path: Path,
    expected_parent_gate_sha256: str = PARENT_GATE_SHA256,
    max_journey_ci_high_min: float = 0.15,
    min_negative_train_seed_fraction: float = 0.75,
) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    parent_gate_path = Path(parent_gate_path).resolve()
    parent = _load_json(parent_gate_path)
    manifest_path = aggregate_dir / "matrix_manifest.json"
    per_eval_path = aggregate_dir / "frozen_per_eval.csv"
    manifest = _load_json(manifest_path)
    per_eval = pd.read_csv(per_eval_path)

    parent_provenance = parent["primary_result"]["matrix_provenance"]
    parent_train_seeds = set(
        parent["confirmation_design"]["confirmation_train_seeds"])
    parent_eval_seeds = set(
        parent["confirmation_design"]["confirmation_eval_seeds"])
    observed_train_seeds = set(manifest.get("train_seeds", []))
    observed_eval_seeds = set(manifest.get("eval_seeds", []))
    lineage_checks = {
        "parent_gate_hash_locked": (
            sha256_file(parent_gate_path) == expected_parent_gate_sha256),
        "parent_primary_confirmed": (
            parent.get("gate_version")
            == "freqduet-v8-compact-primary-confirmation-v1"
            and parent.get("status") == "primary_confirmed"
            and parent.get("primary_claim_eligible") is True
            and parent.get("primary") == PARENT_PRIMARY),
        "longtrain_is_independent_confirmation": (
            manifest.get("stage") == "confirmation"
            and manifest.get("independent_confirmation") is True),
        "exact_configs": manifest.get("configs") == EXPECTED_CONFIGS,
        "exact_train_seeds": (
            manifest.get("train_seeds") == EXPECTED_TRAIN_SEEDS),
        "exact_eval_seeds": (
            manifest.get("eval_seeds") == EXPECTED_EVAL_SEEDS),
        "train_seeds_disjoint_from_parent": not (
            parent_train_seeds & observed_train_seeds),
        "eval_seeds_disjoint_from_parent": not (
            parent_eval_seeds & observed_eval_seeds),
        "longtrain_200_episodes": (
            manifest.get("train_episodes") == 200
            and manifest.get("checkpoint_ep") == 199),
        "expected_rollout_count": manifest.get("expected_rollouts") == 256,
        "model_source_fingerprint_unchanged": (
            manifest.get("run_source_fingerprint", {}).get("sha256")
            == parent_provenance.get("source_fingerprint_sha256")),
        "scenario_contract_unchanged": (
            manifest.get("scenario_contract", {}).get("sha256")
            == parent_provenance.get("scenario_contract_sha256")),
        "longtrain_source_is_clean": (
            manifest.get("run_git_provenance", {}).get("tracked_dirty")
            is False),
    }
    if not all(lineage_checks.values()):
        raise ValueError(
            f"confirmed long-training lineage checks failed: {lineage_checks}")

    base_result = evaluate_selection(
        aggregate_dir,
        candidates=[CONFIRMED_MAIN],
        main=DEFAULT_MAIN,
        reference=DEFAULT_REFERENCE,
        matched_context=MATCHED_CONTEXT,
        expected_stage="confirmation",
    )
    candidate = base_result["candidate_results"][0]
    direction_fraction, train_seed_cv_deltas = (
        _train_seed_direction_fraction(
            per_eval,
            candidate=CONFIRMED_MAIN,
            reference=DEFAULT_REFERENCE,
            metric="headway_cv",
        )
    )
    longtrain_gates = {
        "base_confirmation_gate_passes": (
            base_result.get("status") == "unique_pass"),
        "headway_cv_ci_excludes_zero": (
            float(candidate["headway_cv_delta_ci_high"]) < 0.0),
        "journey_ci_is_noninferior": (
            float(candidate["journey_delta_ci_high"])
            <= float(max_journey_ci_high_min)),
        "headway_cv_direction_consistent": (
            direction_fraction >= min_negative_train_seed_fraction),
    }
    claim_eligible = all(longtrain_gates.values())
    return {
        "gate_version": "freqduet-v6-confirmed-longtrain-v1",
        "status": (
            "longtrain_confirmed" if claim_eligible
            else "longtrain_not_confirmed"),
        "longtrain_claim_eligible": claim_eligible,
        "candidate": CONFIRMED_MAIN,
        "reference": DEFAULT_REFERENCE,
        "matched_context": MATCHED_CONTEXT,
        "lineage_checks": lineage_checks,
        "longtrain_gates": longtrain_gates,
        "thresholds": {
            "max_journey_ci_high_min": max_journey_ci_high_min,
            "min_negative_train_seed_fraction": (
                min_negative_train_seed_fraction),
        },
        "headway_cv_negative_train_seed_fraction": direction_fraction,
        "headway_cv_delta_by_train_seed": train_seed_cv_deltas,
        "parent_gate": {
            "path": str(parent_gate_path),
            "sha256": sha256_file(parent_gate_path),
        },
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
        "base_confirmation_result": base_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--parent-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    result = evaluate_confirmed_longtrain(
        args.aggregate_dir,
        parent_gate_path=args.parent_gate,
    )
    out = args.out or Path(args.aggregate_dir) / "longtrain_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and not result["longtrain_claim_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
