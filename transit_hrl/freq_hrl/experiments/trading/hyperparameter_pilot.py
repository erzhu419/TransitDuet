"""Nested-validation hyperparameter pilot for confirmatory trading runs.

The pilot never receives the confirmatory held-out test seeds. Each candidate
uses training rollouts, an inner checkpoint-selection split, and a disjoint
outer tuning-validation split. Candidate ranking is clustered by independently
initialized training replicate and shared exogenous paths are paired across
candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.experiments.reproducibility import (
    validate_evaluation_seed_roles,
    validate_unique_seeds,
)

from .metrics import (
    DEFAULT_TRAINING_REWARD_SCALE,
    METRIC_CONTRACT_VERSION,
    SELECTION_OBJECTIVE_VERSION,
    validation_utility,
)
from .offpolicy_baseline_validation import (
    OFFPOLICY_MODES,
    train_flat_offpolicy_baseline,
)
from .performance_validation import SCENARIOS
from .ppo_actor_critic import (
    LEARNED_BASELINE_IMPLEMENTATION_VERSION,
    POLICY_MODES,
    train_ppo_actor_critic,
)
from .strong_learned_baseline_validation import (
    DEFAULT_OPTIMIZER_SEEDS,
    DEFAULT_ROLLOUT_SEED_ROOTS,
    DEFAULT_SCENARIOS,
    DEFAULT_VALIDATION_SEEDS,
    count_parameters,
    scenario_optimizer_seed,
)


TUNING_PROTOCOL_VERSION = "nested_validation_hpo_v1"
DEFAULT_TUNING_SEEDS = (68207, 68209, 68213, 68219, 68227)
DEFAULT_PILOT_SCENARIOS = (
    "stationary_low_noise",
    "persistent_shift",
    "ood_period",
)
ALL_POLICY_MODES = POLICY_MODES + OFFPOLICY_MODES


@dataclass(frozen=True)
class TuningCandidate:
    candidate_id: str
    family: str
    parameters: dict[str, Any]

    def applies_to(self, policy_mode: str) -> bool:
        if self.family == "ppo":
            return policy_mode in POLICY_MODES
        if self.family == "offpolicy":
            return policy_mode in OFFPOLICY_MODES
        return False


def _ppo_candidate(candidate_id: str, learning_rate: float, init_log_std: float) -> TuningCandidate:
    return TuningCandidate(
        candidate_id=candidate_id,
        family="ppo",
        parameters={
            "hidden_dim": 64,
            "learning_rate": float(learning_rate),
            "epochs": 4,
            "minibatch_size": 512,
            "init_log_std": float(init_log_std),
            "reward_scale": DEFAULT_TRAINING_REWARD_SCALE,
        },
    )


def _offpolicy_candidate(
    candidate_id: str,
    learning_rate: float,
    warmup_steps: int,
    batch_size: int,
) -> TuningCandidate:
    return TuningCandidate(
        candidate_id=candidate_id,
        family="offpolicy",
        parameters={
            "hidden_dim": 64,
            "learning_rate": float(learning_rate),
            "replay_capacity": 100_000,
            "warmup_steps": int(warmup_steps),
            "batch_size": int(batch_size),
            "updates_per_step": 1,
            "reward_scale": DEFAULT_TRAINING_REWARD_SCALE,
        },
    )


TUNING_CANDIDATES = (
    _ppo_candidate("ppo_lr1e4_std15", 1e-4, -1.5),
    _ppo_candidate("ppo_lr1e4_std10", 1e-4, -1.0),
    _ppo_candidate("ppo_lr3e4_std15", 3e-4, -1.5),
    _ppo_candidate("ppo_lr3e4_std10", 3e-4, -1.0),
    _ppo_candidate("ppo_lr1e3_std15", 1e-3, -1.5),
    _ppo_candidate("ppo_lr1e3_std10", 1e-3, -1.0),
    _ppo_candidate("ppo_lr1e4_std05", 1e-4, -0.5),
    _ppo_candidate("ppo_lr3e4_std05", 3e-4, -0.5),
    _offpolicy_candidate("off_lr1e4_w2048_b64", 1e-4, 2048, 64),
    _offpolicy_candidate("off_lr3e4_w2048_b64", 3e-4, 2048, 64),
    _offpolicy_candidate("off_lr1e3_w2048_b64", 1e-3, 2048, 64),
    _offpolicy_candidate("off_lr3e4_w1024_b64", 3e-4, 1024, 64),
    _offpolicy_candidate("off_lr3e4_w4096_b64", 3e-4, 4096, 64),
    _offpolicy_candidate("off_lr3e4_w2048_b128", 3e-4, 2048, 128),
    _offpolicy_candidate("off_lr1e4_w1024_b64", 1e-4, 1024, 64),
    _offpolicy_candidate("off_lr1e3_w4096_b64", 1e-3, 4096, 64),
)
CANDIDATES_BY_ID = {candidate.candidate_id: candidate for candidate in TUNING_CANDIDATES}


def frozen_config_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_frozen_config(
    payload: dict[str, Any],
    *,
    required_policy_modes: Iterable[str] = ALL_POLICY_MODES,
) -> dict[str, Any]:
    """Reject incomplete, stale, or test-contaminated HPO freezes."""

    required_modes = list(required_policy_modes)
    if payload.get("status") != "frozen_from_validation_only":
        raise ValueError("frozen config status must be frozen_from_validation_only")
    if payload.get("stage") != "final" or not bool(payload.get("final_design_complete")):
        raise ValueError("frozen config must come from a complete final HPO design")
    if payload.get("tuning_protocol_version") != TUNING_PROTOCOL_VERSION:
        raise ValueError("frozen config tuning protocol version mismatch")
    if payload.get("selection_objective_version") != SELECTION_OBJECTIVE_VERSION:
        raise ValueError("frozen config selection objective version mismatch")
    if payload.get("learned_baseline_implementation_version") != (
        LEARNED_BASELINE_IMPLEMENTATION_VERSION
    ):
        raise ValueError("frozen config implementation version mismatch")
    if payload.get("heldout_test_access_status") != "not_loaded":
        raise ValueError("frozen config must not access held-out test data")
    if payload.get("heldout_test_seeds"):
        raise ValueError("frozen config contains held-out test seeds")
    checkpoint_seeds = {
        int(seed) for seed in payload.get("checkpoint_validation_seeds", [])
    }
    tuning_seeds = {
        int(seed) for seed in payload.get("tuning_validation_seeds", [])
    }
    if not checkpoint_seeds or not tuning_seeds or checkpoint_seeds & tuning_seeds:
        raise ValueError("frozen config validation splits must be non-empty and disjoint")
    if set(payload.get("scenarios", [])) != set(DEFAULT_SCENARIOS):
        raise ValueError("frozen config must cover the preregistered five scenarios")
    replicate_seeds = [
        int(seed) for seed in payload.get("training_replicate_seeds", [])
    ]
    if len(set(replicate_seeds)) < 5:
        raise ValueError("frozen config requires at least five training replicates")

    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("frozen config selected policies are missing")
    search_budget = payload.get("search_budget_candidates_per_policy", {})
    for mode in required_modes:
        if mode not in ALL_POLICY_MODES:
            raise ValueError(f"unknown required policy mode: {mode}")
        entry = selected.get(mode)
        if not isinstance(entry, dict):
            raise ValueError(f"frozen config is missing policy mode: {mode}")
        if entry.get("learning_gate_status") != "eligible":
            raise ValueError(f"frozen policy failed learning gate: {mode}")
        candidate_id = str(entry.get("candidate_id", ""))
        candidate = CANDIDATES_BY_ID.get(candidate_id)
        if candidate is None or not candidate.applies_to(mode):
            raise ValueError(f"invalid frozen candidate for {mode}: {candidate_id}")
        if entry.get("candidate_family") != candidate.family:
            raise ValueError(f"frozen candidate family mismatch for {mode}")
        if entry.get("parameters") != candidate.parameters:
            raise ValueError(f"frozen candidate parameters drifted for {mode}")
        if int(search_budget.get(mode, 0)) < 2:
            raise ValueError(f"frozen search budget is too small for {mode}")
    return {
        "status": "valid",
        "sha256": frozen_config_sha256(payload),
        "selected": {mode: selected[mode] for mode in required_modes},
        "implementation_version": LEARNED_BASELINE_IMPLEMENTATION_VERSION,
    }


def load_frozen_config(
    path: Path,
    *,
    required_policy_modes: Iterable[str] = ALL_POLICY_MODES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen config must be a JSON object")
    return payload, validate_frozen_config(
        payload, required_policy_modes=required_policy_modes
    )


def candidate_ids_for_mode(policy_mode: str) -> list[str]:
    if policy_mode not in ALL_POLICY_MODES:
        raise ValueError(f"unknown policy_mode: {policy_mode}")
    return [
        candidate.candidate_id
        for candidate in TUNING_CANDIDATES
        if candidate.applies_to(policy_mode)
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_hpo_cell(
    *,
    candidate_id: str,
    policy_mode: str,
    scenario: str,
    training_replicate_seed: int,
    train_seeds: list[int],
    checkpoint_validation_seeds: list[int],
    tuning_validation_seeds: list[int],
    steps: int,
    assets: int,
    iterations: int,
) -> dict[str, Any]:
    if candidate_id not in CANDIDATES_BY_ID:
        raise ValueError(f"unknown candidate_id: {candidate_id}")
    candidate = CANDIDATES_BY_ID[candidate_id]
    if not candidate.applies_to(policy_mode):
        raise ValueError(f"candidate {candidate_id} does not apply to {policy_mode}")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    rollout_seed_roots = validate_unique_seeds(train_seeds, role="rollout_seed_roots")
    checkpoint_seeds, tuning_seeds = validate_evaluation_seed_roles(
        checkpoint_validation_seeds,
        tuning_validation_seeds,
    )
    run_seed = scenario_optimizer_seed(int(training_replicate_seed), scenario)
    started = time.perf_counter()
    params = candidate.parameters
    if policy_mode in POLICY_MODES:
        model_payload, tuning_rows, model = train_ppo_actor_critic(
            train_seeds=rollout_seed_roots,
            validation_seeds=checkpoint_seeds,
            eval_seeds=tuning_seeds,
            steps=int(steps),
            assets=int(assets),
            scenario=scenario,
            iterations=int(iterations),
            seed=run_seed,
            hidden_dim=int(params["hidden_dim"]),
            learning_rate=float(params["learning_rate"]),
            ppo_epochs=int(params["epochs"]),
            minibatch_size=int(params["minibatch_size"]),
            init_log_std=float(params["init_log_std"]),
            resample_training_paths=True,
            policy_mode=policy_mode,
            use_handcrafted_frequency_prior=False,
            evaluation_role="tuning_validation",
            reward_scale=float(params["reward_scale"]),
        )
    else:
        model_payload, tuning_rows, model = train_flat_offpolicy_baseline(
            policy_mode=policy_mode,
            train_seeds=rollout_seed_roots,
            validation_seeds=checkpoint_seeds,
            eval_seeds=tuning_seeds,
            steps=int(steps),
            assets=int(assets),
            scenario=scenario,
            iterations=int(iterations),
            seed=run_seed,
            hidden_dim=int(params["hidden_dim"]),
            learning_rate=float(params["learning_rate"]),
            replay_capacity=int(params["replay_capacity"]),
            warmup_steps=int(params["warmup_steps"]),
            batch_size=int(params["batch_size"]),
            updates_per_step=int(params["updates_per_step"]),
            resample_training_paths=True,
            evaluation_role="tuning_validation",
            reward_scale=float(params["reward_scale"]),
        )
    if model_payload.get("evaluation_role") != "tuning_validation":
        raise RuntimeError("trainer exposed the wrong evaluation role during HPO")
    if model_payload.get("heldout_test_seeds"):
        raise RuntimeError("HPO must not load held-out test seeds")

    annotated_rows: list[dict[str, Any]] = []
    utilities: list[float] = []
    for row in tuning_rows:
        utility = validation_utility(row)
        utilities.append(utility)
        annotated_rows.append({
            **row,
            "candidate_id": candidate_id,
            "candidate_family": candidate.family,
            "policy_mode": policy_mode,
            "scenario": scenario,
            "training_replicate_seed": int(training_replicate_seed),
            "optimizer_seed": int(run_seed),
            "evaluation_role": "tuning_validation",
            "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
            "selection_utility": float(utility),
            "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
            "learned_baseline_implementation_version": (
                LEARNED_BASELINE_IMPLEMENTATION_VERSION
            ),
        })
    if not utilities or not np.all(np.isfinite(utilities)):
        raise RuntimeError("HPO tuning utilities must be finite and non-empty")
    elapsed = float(time.perf_counter() - started)
    summary = {
        "candidate_id": candidate_id,
        "candidate_family": candidate.family,
        "candidate_parameters": dict(candidate.parameters),
        "policy_mode": policy_mode,
        "scenario": scenario,
        "training_replicate_seed": int(training_replicate_seed),
        "optimizer_seed": int(run_seed),
        "rollout_seed_roots": list(rollout_seed_roots),
        "checkpoint_validation_seeds": list(checkpoint_seeds),
        "tuning_validation_seeds": list(tuning_seeds),
        "heldout_test_seeds": [],
        "heldout_test_access_status": "not_loaded",
        "evaluation_role": "tuning_validation",
        "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
        "learned_baseline_implementation_version": (
            LEARNED_BASELINE_IMPLEMENTATION_VERSION
        ),
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
        "training_path_protocol": str(model_payload.get("training_path_protocol", "")),
        "checkpoint_selection_protocol": str(
            model_payload.get("checkpoint_selection_protocol", "")
        ),
        "steps": int(steps),
        "assets": int(assets),
        "iterations": int(iterations),
        "parameter_count": count_parameters(model),
        "environment_steps_train": int(model_payload.get("environment_steps_train", 0)),
        "environment_steps_checkpoint_validation": int(
            model_payload.get("environment_steps_validation", 0)
        ),
        "environment_steps_tuning_validation": int(
            model_payload.get("environment_steps_eval", 0)
        ),
        "tuning_seed_count": len(tuning_seeds),
        "selection_utility_mean": float(np.mean(utilities)),
        "selection_utility_min": float(np.min(utilities)),
        "selection_utility_std": float(np.std(utilities, ddof=1)) if len(utilities) > 1 else 0.0,
        "best_checkpoint_inner_validation_score": float(model_payload.get("best_score", 0.0)),
        "initial_checkpoint_validation_score": float(
            model_payload.get("initial_validation_score", 0.0)
        ),
        "validation_learning_gain": float(
            model_payload.get("validation_learning_gain", 0.0)
        ),
        "selected_checkpoint_iteration": int(
            model_payload.get("selected_checkpoint_iteration", -1)
        ),
        "elapsed_sec": elapsed,
        "cell_status": "valid",
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_payload.get("config", {}),
        "candidate_id": candidate_id,
        "candidate_parameters": dict(candidate.parameters),
        "policy_mode": policy_mode,
        "scenario": scenario,
        "training_replicate_seed": int(training_replicate_seed),
        "checkpoint_validation_seeds": list(checkpoint_seeds),
        "tuning_validation_seeds": list(tuning_seeds),
        "heldout_test_seeds": [],
        "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
        "learned_baseline_implementation_version": (
            LEARNED_BASELINE_IMPLEMENTATION_VERSION
        ),
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
    }
    return {"tuning_rows": annotated_rows, "cell_summary": summary, "checkpoint": checkpoint}


def write_hpo_cell(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "tuning_rows.csv", payload["tuning_rows"])
    with (output_dir / "cell_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["cell_summary"], handle, indent=2, sort_keys=True)
    torch.save(payload["checkpoint"], output_dir / "checkpoint.pt")
    summary = payload["cell_summary"]
    lines = [
        "# Nested-Validation HPO Cell",
        "",
        f"- status: `{summary['cell_status']}`",
        f"- candidate: `{summary['candidate_id']}`",
        f"- policy: `{summary['policy_mode']}`",
        f"- scenario: `{summary['scenario']}`",
        f"- training replicate: `{summary['training_replicate_seed']}`",
        f"- tuning utility: `{summary['selection_utility_mean']:.8f}`",
        f"- held-out test access: `{summary['heldout_test_access_status']}`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bootstrap_mean_ci(values: Iterable[float], *, seed: int, draws: int = 10_000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(draws), array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def merge_hpo_cells(
    input_dirs: list[Path],
    *,
    expected_policy_modes: list[str] | None = None,
    expected_candidate_ids: list[str] | None = None,
    expected_scenarios: list[str] | None = None,
    expected_replicate_seeds: list[int] | None = None,
    top_k: int = 3,
    stage: str = "pilot",
) -> dict[str, Any]:
    cell_summaries: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, str, str, int]] = set()
    for directory in input_dirs:
        base = Path(directory)
        summary_path = base / "cell_summary.json"
        if not summary_path.exists():
            raise ValueError(f"missing HPO cell summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        key = (
            str(summary["policy_mode"]),
            str(summary["candidate_id"]),
            str(summary["scenario"]),
            int(summary["training_replicate_seed"]),
        )
        if key in seen_cells:
            raise ValueError(f"duplicate HPO cell: {key}")
        seen_cells.add(key)
        if summary.get("cell_status") != "valid":
            raise ValueError(f"invalid HPO cell: {key}")
        if summary.get("heldout_test_seeds"):
            raise ValueError(f"HPO cell accessed held-out test seeds: {key}")
        if summary.get("tuning_protocol_version") != TUNING_PROTOCOL_VERSION:
            raise ValueError(f"HPO protocol mismatch: {key}")
        if summary.get("selection_objective_version") != SELECTION_OBJECTIVE_VERSION:
            raise ValueError(f"HPO selection objective mismatch: {key}")
        if summary.get("learned_baseline_implementation_version") != (
            LEARNED_BASELINE_IMPLEMENTATION_VERSION
        ):
            raise ValueError(f"HPO implementation version mismatch: {key}")
        cell_summaries.append(summary)
        tuning_rows.extend(_read_csv(base / "tuning_rows.csv"))

    policy_modes = list(expected_policy_modes or sorted({key[0] for key in seen_cells}))
    scenarios = list(expected_scenarios or sorted({key[2] for key in seen_cells}))
    replicates = list(expected_replicate_seeds or sorted({key[3] for key in seen_cells}))
    requested_candidates = list(expected_candidate_ids or sorted({key[1] for key in seen_cells}))
    expected_cells = {
        (mode, candidate_id, scenario, int(replicate))
        for mode in policy_modes
        for candidate_id in requested_candidates
        if CANDIDATES_BY_ID[candidate_id].applies_to(mode)
        for scenario in scenarios
        for replicate in replicates
    }
    missing = sorted(expected_cells - seen_cells)
    unexpected = sorted(seen_cells - expected_cells)
    coverage_status = "complete" if not missing and not unexpected else "incomplete"
    if coverage_status != "complete":
        preview = ", ".join(map(str, missing[:3] + unexpected[:3]))
        raise ValueError(f"incomplete HPO matrix: {preview}")

    utility_by_cell: dict[tuple[str, str, str, int], list[float]] = {}
    for row in tuning_rows:
        if str(row.get("evaluation_role")) != "tuning_validation":
            raise ValueError("HPO merge found a non-tuning evaluation row")
        key = (
            str(row["policy_mode"]),
            str(row["candidate_id"]),
            str(row["scenario"]),
            int(float(row["training_replicate_seed"])),
        )
        utility_by_cell.setdefault(key, []).append(float(row["selection_utility"]))

    leaderboard: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    top_candidates: dict[str, list[str]] = {}
    for mode in policy_modes:
        mode_candidates = [
            candidate_id for candidate_id in requested_candidates
            if CANDIDATES_BY_ID[candidate_id].applies_to(mode)
        ]
        mode_rows: list[dict[str, Any]] = []
        for candidate_id in mode_candidates:
            replicate_scores = []
            for replicate in replicates:
                scenario_scores = [
                    float(np.mean(utility_by_cell[(mode, candidate_id, scenario, int(replicate))]))
                    for scenario in scenarios
                ]
                replicate_scores.append(float(np.mean(scenario_scores)))
            ci_low, ci_high = _bootstrap_mean_ci(
                replicate_scores,
                seed=scenario_optimizer_seed(int(replicates[0]), scenarios[0]),
            )
            row = {
                "policy_mode": mode,
                "candidate_id": candidate_id,
                "candidate_family": CANDIDATES_BY_ID[candidate_id].family,
                "independent_training_replicates": len(replicate_scores),
                "scenario_count": len(scenarios),
                "tuning_utility_mean": float(np.mean(replicate_scores)),
                "tuning_utility_std_across_replicates": (
                    float(np.std(replicate_scores, ddof=1))
                    if len(replicate_scores) > 1 else 0.0
                ),
                "tuning_utility_ci95_low": ci_low,
                "tuning_utility_ci95_high": ci_high,
                "robust_selection_score": ci_low,
            }
            matching_summaries = [
                summary for summary in cell_summaries
                if str(summary["policy_mode"]) == mode
                and str(summary["candidate_id"]) == candidate_id
            ]
            selected_iterations = [
                int(summary.get("selected_checkpoint_iteration", -1))
                for summary in matching_summaries
            ]
            learning_gains = [
                float(summary.get("validation_learning_gain", 0.0))
                for summary in matching_summaries
            ]
            trained_fraction = float(np.mean([
                iteration >= 0 for iteration in selected_iterations
            ])) if selected_iterations else 0.0
            learning_gain_mean = float(np.mean(learning_gains)) if learning_gains else 0.0
            row["trained_checkpoint_fraction"] = trained_fraction
            row["validation_learning_gain_mean"] = learning_gain_mean
            row["learning_gate_status"] = (
                "eligible"
                if trained_fraction >= 0.80 and learning_gain_mean > 0.0
                else "ineligible"
            )
            mode_rows.append(row)
        mode_rows.sort(
            key=lambda row: (
                0 if row["learning_gate_status"] == "eligible" else 1,
                -float(row["robust_selection_score"]),
                -float(row["tuning_utility_mean"]),
                str(row["candidate_id"]),
            )
        )
        for rank, row in enumerate(mode_rows, start=1):
            row["rank"] = rank
        leaderboard.extend(mode_rows)
        eligible_rows = [
            row for row in mode_rows if row["learning_gate_status"] == "eligible"
        ]
        selection_pool = eligible_rows if eligible_rows else mode_rows[:1]
        winners = selection_pool[: max(1, min(int(top_k), len(selection_pool)))]
        top_candidates[mode] = [str(row["candidate_id"]) for row in winners]
        winner_id = str(winners[0]["candidate_id"])
        selected[mode] = {
            "candidate_id": winner_id,
            "candidate_family": CANDIDATES_BY_ID[winner_id].family,
            "parameters": dict(CANDIDATES_BY_ID[winner_id].parameters),
            "robust_selection_score": float(winners[0]["robust_selection_score"]),
            "tuning_utility_mean": float(winners[0]["tuning_utility_mean"]),
            "learning_gate_status": str(winners[0]["learning_gate_status"]),
            "trained_checkpoint_fraction": float(
                winners[0]["trained_checkpoint_fraction"]
            ),
            "validation_learning_gain_mean": float(
                winners[0]["validation_learning_gain_mean"]
            ),
        }

    candidate_counts = {
        mode: len([row for row in leaderboard if row["policy_mode"] == mode])
        for mode in policy_modes
    }
    all_modes_eligible = all(
        row["learning_gate_status"] == "eligible" for row in selected.values()
    )
    final_design_complete = (
        set(policy_modes) == set(ALL_POLICY_MODES)
        and set(scenarios) == set(DEFAULT_SCENARIOS)
        and len(replicates) >= 5
        and candidate_counts
        and min(candidate_counts.values()) >= 2
    )
    if str(stage) == "final" and all_modes_eligible and final_design_complete:
        freeze_status = "frozen_from_validation_only"
    elif not all_modes_eligible:
        freeze_status = "not_freezable_learning_gate"
    elif str(stage) == "final":
        freeze_status = "not_freezable_incomplete_final_design"
    else:
        freeze_status = "provisional_validation_only"
    freeze = {
        "status": freeze_status,
        "stage": str(stage),
        "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
        "learned_baseline_implementation_version": (
            LEARNED_BASELINE_IMPLEMENTATION_VERSION
        ),
        "heldout_test_access_status": "not_loaded",
        "checkpoint_validation_seeds": sorted({
            int(seed)
            for summary in cell_summaries
            for seed in summary["checkpoint_validation_seeds"]
        }),
        "tuning_validation_seeds": sorted({
            int(seed)
            for summary in cell_summaries
            for seed in summary["tuning_validation_seeds"]
        }),
        "heldout_test_seeds": [],
        "scenarios": scenarios,
        "training_replicate_seeds": [int(seed) for seed in replicates],
        "search_budget_candidates_per_policy": candidate_counts,
        "final_design_complete": bool(final_design_complete),
        "selected": selected,
        "top_candidates": top_candidates,
    }
    return {
        "cell_summaries": cell_summaries,
        "tuning_rows": tuning_rows,
        "leaderboard": leaderboard,
        "frozen_config": freeze,
        "summary": {
            "stage": str(stage),
            "cell_count": len(cell_summaries),
            "expected_cell_count": len(expected_cells),
            "matrix_coverage_status": coverage_status,
            "policy_mode_count": len(policy_modes),
            "scenario_count": len(scenarios),
            "training_replicate_count": len(replicates),
            "heldout_test_access_count": 0,
            "tuning_protocol_status": "valid",
            "learning_gate_status": (
                "supported" if all_modes_eligible else "not_supported"
            ),
            "final_design_status": (
                "complete" if final_design_complete else "incomplete"
            ),
        },
    }


def write_hpo_merge(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "leaderboard.csv", payload["leaderboard"])
    _write_csv(output_dir / "cell_summaries.csv", payload["cell_summaries"])
    with (output_dir / "frozen_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["frozen_config"], handle, indent=2, sort_keys=True)
    serializable = {
        key: value for key, value in payload.items() if key != "tuning_rows"
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)
    lines = [
        "# Nested-Validation Hyperparameter Pilot",
        "",
        f"- protocol: `{TUNING_PROTOCOL_VERSION}`",
        f"- coverage: `{payload['summary']['matrix_coverage_status']}`",
        f"- held-out test accesses: `{payload['summary']['heldout_test_access_count']}`",
        "",
        "| policy | candidate | rank | robust score | mean utility |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["leaderboard"]:
        lines.append(
            f"| {row['policy_mode']} | {row['candidate_id']} | {row['rank']} "
            f"| {float(row['robust_selection_score']):+.8f} "
            f"| {float(row['tuning_utility_mean']):+.8f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", choices=sorted(CANDIDATES_BY_ID))
    parser.add_argument("--policy-mode", choices=ALL_POLICY_MODES)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--training-replicate-seed", type=int)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(DEFAULT_ROLLOUT_SEED_ROOTS))
    parser.add_argument(
        "--checkpoint-validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    parser.add_argument(
        "--tuning-validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TUNING_SEEDS),
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge-inputs", type=Path, nargs="*")
    parser.add_argument("--expected-policy-modes", nargs="*", choices=ALL_POLICY_MODES)
    parser.add_argument("--expected-candidate-ids", nargs="*", choices=sorted(CANDIDATES_BY_ID))
    parser.add_argument("--expected-scenarios", nargs="*", choices=SCENARIOS)
    parser.add_argument("--expected-replicate-seeds", type=int, nargs="*")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.merge_inputs:
        payload = merge_hpo_cells(
            list(args.merge_inputs),
            expected_policy_modes=list(args.expected_policy_modes or []),
            expected_candidate_ids=list(args.expected_candidate_ids or []),
            expected_scenarios=list(args.expected_scenarios or []),
            expected_replicate_seeds=list(args.expected_replicate_seeds or []),
            top_k=int(args.top_k),
            stage=str(args.stage),
        )
        write_hpo_merge(args.output_dir, payload)
        print(
            f"hpo_merge status={payload['summary']['tuning_protocol_status']} "
            f"cells={payload['summary']['cell_count']} output={args.output_dir}"
        )
        return
    required = {
        "candidate_id": args.candidate_id,
        "policy_mode": args.policy_mode,
        "scenario": args.scenario,
        "training_replicate_seed": args.training_replicate_seed,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        parser.error("cell mode requires " + ", ".join(f"--{key.replace('_', '-')}" for key in missing))
    payload = run_hpo_cell(
        candidate_id=str(args.candidate_id),
        policy_mode=str(args.policy_mode),
        scenario=str(args.scenario),
        training_replicate_seed=int(args.training_replicate_seed),
        train_seeds=list(args.train_seeds),
        checkpoint_validation_seeds=list(args.checkpoint_validation_seeds),
        tuning_validation_seeds=list(args.tuning_validation_seeds),
        steps=int(args.steps),
        assets=int(args.assets),
        iterations=int(args.iterations),
    )
    write_hpo_cell(args.output_dir, payload)
    print(
        f"hpo_cell status=valid candidate={args.candidate_id} "
        f"policy={args.policy_mode} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
