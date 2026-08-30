"""Domain-agnostic selection for guarded restoration transactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RestorationPortfolioDecision:
    """Frozen selection result for one design-role portfolio."""

    selected_index: int | None
    eligible_indices: tuple[int, ...]
    design_eligibility: tuple[bool, ...]
    fold_eligibility: tuple[tuple[bool, ...], ...]
    trace_invariance_eligibility: tuple[bool, ...]


def restoration_snapshot_eligible(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    minimum_reduction: float,
    funnel_multiplier: float,
) -> bool:
    """Apply the reward floor, aggregate merit, and worst-case funnel gate."""

    reduction = float(minimum_reduction)
    funnel = float(funnel_multiplier)
    if not 0.0 <= reduction < 1.0:
        raise ValueError("restoration minimum reduction must be in [0, 1)")
    if not np.isfinite(funnel) or funnel < 1.0:
        raise ValueError("restoration funnel multiplier must be at least one")
    values = {
        "candidate_merit": float(candidate["frequency_violation_merit"]),
        "baseline_merit": float(baseline["frequency_violation_merit"]),
        "candidate_worst": float(candidate["worst_frequency_violation"]),
        "baseline_worst": float(baseline["worst_frequency_violation"]),
    }
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("restoration diagnostics must be finite and non-negative")
    return bool(
        int(candidate["reward_violation_count"]) == 0
        and values["candidate_merit"]
        <= values["baseline_merit"] * (1.0 - reduction)
        and values["candidate_worst"]
        <= values["baseline_worst"] * funnel
    )


def fold_guarded_restoration_eligibility(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
    fold_snapshots: Sequence[dict[str, Any]],
    fold_baselines: Sequence[dict[str, Any]],
    *,
    minimum_reduction: float,
    funnel_multiplier: float,
) -> tuple[bool, list[bool]]:
    """Require the pooled design and every predeclared fold to pass."""

    if len(fold_snapshots) != len(fold_baselines) or not fold_snapshots:
        raise ValueError("design fold snapshots and baselines must align")
    fold_flags = [
        restoration_snapshot_eligible(
            fold_snapshot,
            fold_baseline,
            minimum_reduction=minimum_reduction,
            funnel_multiplier=funnel_multiplier,
        )
        for fold_snapshot, fold_baseline in zip(
            fold_snapshots, fold_baselines, strict=True
        )
    ]
    pooled = restoration_snapshot_eligible(
        snapshot,
        baseline,
        minimum_reduction=minimum_reduction,
        funnel_multiplier=funnel_multiplier,
    )
    return bool(pooled and all(fold_flags)), fold_flags


def paired_trace_invariance_diagnostics(
    candidate_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    *,
    identity_fields: tuple[str, ...] = ("disturbance_mode", "seed"),
) -> dict[str, Any]:
    """Check exact paired behavior traces for a function-preserving transaction."""

    if not candidate_rows or not baseline_rows or not identity_fields:
        raise ValueError("paired trace diagnostics require rows and identity fields")

    def index(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
        indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            key = tuple(row[field] for field in identity_fields)
            if key in indexed:
                raise ValueError("paired trace rows must have unique path identities")
            indexed[key] = row
        return indexed

    candidate_index = index(candidate_rows)
    baseline_index = index(baseline_rows)
    if candidate_index.keys() != baseline_index.keys():
        raise ValueError("paired trace rows must use identical paths")
    trace_fields = (
        "ExecutedActionTraceSHA256",
        "RewardTraceSHA256",
        "LatentPolicyTraceSHA256",
    )
    match_counts = {field: 0 for field in trace_fields}
    reward_mean_deltas: list[float] = []
    episode_return_deltas: list[float] = []
    for key, candidate in candidate_index.items():
        baseline = baseline_index[key]
        for field in trace_fields:
            candidate_trace = str(candidate.get(field, ""))
            baseline_trace = str(baseline.get(field, ""))
            if not candidate_trace or not baseline_trace:
                raise ValueError(f"paired trace rows omit {field}")
            match_counts[field] += int(candidate_trace == baseline_trace)
        reward_mean_deltas.append(abs(
            float(candidate["reward_mean"]) - float(baseline["reward_mean"])
        ))
        episode_return_deltas.append(abs(
            float(candidate["episode_return"])
            - float(baseline["episode_return"])
        ))
    count = len(candidate_index)
    all_traces_invariant = all(
        matches == count for matches in match_counts.values()
    )
    return {
        "contract": "paired_exact_action_reward_and_latent_trace_invariance_v1",
        "path_count": count,
        "executed_action_trace_match_count": match_counts[
            "ExecutedActionTraceSHA256"
        ],
        "reward_trace_match_count": match_counts["RewardTraceSHA256"],
        "latent_policy_trace_match_count": match_counts[
            "LatentPolicyTraceSHA256"
        ],
        "maximum_reward_mean_absolute_delta": float(max(reward_mean_deltas)),
        "maximum_episode_return_absolute_delta": float(
            max(episode_return_deltas)
        ),
        "all_traces_invariant": bool(all_traces_invariant),
    }


def select_guarded_restoration_portfolio(
    candidates: Sequence[dict[str, Any]],
    *,
    baseline: dict[str, Any],
    fold_baselines: Sequence[dict[str, Any]],
    minimum_reduction: float,
    funnel_multiplier: float,
) -> RestorationPortfolioDecision:
    """Select the best eligible transaction using one domain-neutral contract."""

    if not candidates:
        return RestorationPortfolioDecision(None, (), (), (), ())
    design_flags: list[bool] = []
    fold_flags: list[tuple[bool, ...]] = []
    trace_flags: list[bool] = []
    for candidate in candidates:
        eligible, candidate_fold_flags = fold_guarded_restoration_eligibility(
            candidate["snapshot"],
            baseline,
            candidate["fold_snapshots"],
            fold_baselines,
            minimum_reduction=minimum_reduction,
            funnel_multiplier=funnel_multiplier,
        )
        requires_invariance = bool(candidate.get(
            "requires_trace_invariance", False
        ))
        diagnostics = candidate.get("trace_invariance")
        trace_eligible = bool(
            not requires_invariance
            or (
                isinstance(diagnostics, dict)
                and diagnostics.get("all_traces_invariant") is True
            )
        )
        design_flags.append(bool(eligible and trace_eligible))
        fold_flags.append(tuple(map(bool, candidate_fold_flags)))
        trace_flags.append(trace_eligible)

    eligible_indices = tuple(
        index for index, eligible in enumerate(design_flags) if eligible
    )

    def rank(index: int) -> tuple[float, ...]:
        candidate = candidates[index]
        snapshot = candidate["snapshot"]
        priority = tuple(float(value) for value in candidate.get(
            "selection_priority", ()
        ))
        if any(not np.isfinite(value) for value in priority):
            raise ValueError("restoration selection priority must be finite")
        return (
            float(snapshot["reward_violation_count"]),
            float(snapshot["frequency_violation_merit"]),
            float(snapshot["worst_frequency_violation"]),
            *priority,
            float(index),
        )

    selected_index = min(eligible_indices, key=rank) if eligible_indices else None
    return RestorationPortfolioDecision(
        selected_index=selected_index,
        eligible_indices=eligible_indices,
        design_eligibility=tuple(design_flags),
        fold_eligibility=tuple(fold_flags),
        trace_invariance_eligibility=tuple(trace_flags),
    )
