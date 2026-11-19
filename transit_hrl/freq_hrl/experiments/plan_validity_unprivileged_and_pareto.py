"""Unprivileged renewal-query acquisition and multi-budget Pareto evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract

from .compute_budget import active_ledger, count_work
from .plan_validity_transfer import (
    PlanValidityTaskSpec,
    jitter_schedule,
    run_episode,
    threshold_score,
)
from .pointmaze_goal_validation import squash_box_action
from .pointmaze_plan_validity_branching import (
    PlanRenewalOpportunity,
    _causal_plan_features,
    _evenly_spaced,
)
from .pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder
from .pointmaze_timing_baselines import TimingRule


PARETO_BUDGETS: tuple[tuple[int, int, int, int], ...] = (
    # (calls_per_episode, bin_steps, max_offset_steps, check_stride_steps)
    (12, 100, 50, 10),
    (15, 80, 40, 8),
    (20, 60, 30, 6),
    (24, 50, 25, 5),
    (30, 40, 20, 4),
    (40, 30, 15, 3),
)
QUANTILE_CALIBRATION_GRID: tuple[float, ...] = (
    0.25,
    0.40,
    0.55,
    0.70,
    0.75,
    0.85,
    0.92,
)


@dataclass(frozen=True)
class BudgetScheduleSpec:
    calls_per_episode: int
    bin_steps: int
    max_offset_steps: int
    check_stride_steps: int

    def validate(self, horizon: int) -> None:
        if (
            int(horizon) < self.bin_steps
            or int(horizon) % self.bin_steps != 0
            or int(horizon) // self.bin_steps != self.calls_per_episode
            or not 0 < self.max_offset_steps < self.bin_steps
            or self.check_stride_steps < 1
            or self.max_offset_steps % self.check_stride_steps != 0
        ):
            raise ValueError(f"invalid budget schedule spec: {self}")


def default_pareto_specs(horizon: int = 1200) -> tuple[BudgetScheduleSpec, ...]:
    specs = tuple(BudgetScheduleSpec(*row) for row in PARETO_BUDGETS)
    for spec in specs:
        spec.validate(horizon)
    return specs


def eligible_check_steps(
    *,
    horizon: int,
    schedule: Iterable[int],
    period_steps: int = 50,
    max_offset_steps: int = 25,
    check_stride_steps: int = 5,
    branch_window_steps: int = 50,
) -> tuple[int, ...]:
    """Return valid non-colliding check steps without touching simulator events."""
    blocked = set(map(int, schedule))
    latest_exclusive = int(horizon) - int(branch_window_steps)
    steps = [
        step
        for step in range(int(period_steps), latest_exclusive)
        if (step % int(period_steps)) <= int(max_offset_steps)
        and (step % int(period_steps)) % int(check_stride_steps) == 0
        and step not in blocked
    ]
    if not steps:
        raise ValueError("no eligible non-colliding check steps available")
    return tuple(steps)


def unprivileged_uniform_opportunities(
    *,
    seed: int,
    horizon: int,
    schedule: Iterable[int],
    count: int = 24,
    period_steps: int = 50,
    max_offset_steps: int = 25,
    check_stride_steps: int = 5,
    branch_window_steps: int = 50,
) -> tuple[PlanRenewalOpportunity, ...]:
    """Stratified clock-only query acquisition with zero state or event access."""
    if int(count) < 1:
        raise ValueError("query count must be positive")
    candidates = eligible_check_steps(
        horizon=horizon,
        schedule=schedule,
        period_steps=period_steps,
        max_offset_steps=max_offset_steps,
        check_stride_steps=check_stride_steps,
        branch_window_steps=branch_window_steps,
    )
    if len(candidates) < int(count):
        raise ValueError("fewer eligible check steps than requested queries")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9_101_001]))
    buckets = np.array_split(np.asarray(candidates, dtype=np.int64), int(count))
    chosen = sorted(int(bucket[int(rng.integers(0, len(bucket)))]) for bucket in buckets)
    if len(set(chosen)) != int(count):
        raise RuntimeError("uniform unprivileged opportunities must be unique")
    return tuple(
        PlanRenewalOpportunity(
            category="unprivileged_uniform",
            step=step,
            source_step=None,
            lag_steps=None,
        )
        for step in chosen
    )


def unprivileged_observable_opportunities(
    controller: Any,
    *,
    task_spec: PlanValidityTaskSpec,
    seed: int,
    args: Any,
    time_scale: PhysicalTimeScaleContract,
    schedule: Iterable[int],
    count: int = 24,
) -> tuple[PlanRenewalOpportunity, ...]:
    """Select renewal queries using causal observable history only (no hidden labels)."""
    if int(count) < 3 or int(count) % 3 != 0:
        raise ValueError("observable opportunity count must be a positive multiple of 3")
    candidates = eligible_check_steps(
        horizon=args.horizon,
        schedule=schedule,
        period_steps=time_scale.upper_period_steps,
        max_offset_steps=args.max_offset_steps,
        check_stride_steps=args.check_stride_steps,
        branch_window_steps=round(args.branch_window_seconds / time_scale.dt_seconds),
    )
    if len(candidates) < int(count):
        raise ValueError("fewer eligible check steps than requested observable queries")
    _, trajectory = run_episode(
        controller,
        task_spec=task_spec,
        seed=int(seed),
        args=args,
        time_scale=time_scale,
        schedule=tuple(schedule),
        feature_steps=candidates,
    )
    rows = trajectory["feature_rows"]
    if len(rows) != len(candidates):
        raise RuntimeError("observable query scan did not record all candidate steps")
    for row in rows:
        if row.get("candidate_feature_has_future_access") or row.get("candidate_feature_has_regime_label"):
            raise RuntimeError("observable query scan leaked privileged or future data")

    names = tuple(rows[0]["feature_names"])
    vel_idx = names.index("target_velocity_change_norm")
    force_idx = names.index("force_current_norm")
    steps = np.asarray([int(r["opportunity_step"]) for r in rows], dtype=np.int64)
    vel_change = np.asarray([float(r["causal_features"][vel_idx]) for r in rows], dtype=np.float64)
    force_norm = np.asarray([float(r["causal_features"][force_idx]) for r in rows], dtype=np.float64)

    per_group = int(count) // 3
    used: set[int] = set()
    selected: list[PlanRenewalOpportunity] = []

    def take_top_spaced(metric: np.ndarray, category: str, pool_fraction: float = 0.35) -> None:
        available = [i for i, s in enumerate(steps) if int(s) not in used]
        pool_size = max(per_group, int(np.ceil(len(available) * pool_fraction)))
        ranked = sorted(available, key=lambda i: (-metric[i], int(steps[i])))[:pool_size]
        chosen_steps = _evenly_spaced([int(steps[i]) for i in ranked], count=per_group)
        for step in chosen_steps:
            used.add(int(step))
            selected.append(
                PlanRenewalOpportunity(
                    category=category,
                    step=int(step),
                    source_step=None,
                    lag_steps=None,
                )
            )

    take_top_spaced(vel_change, "observable_motion_change", pool_fraction=0.35)
    take_top_spaced(force_norm, "observable_force_activity", pool_fraction=0.35)
    calm_score = -(
        vel_change / max(float(np.std(vel_change)), 1e-8)
        + force_norm / max(float(np.std(force_norm)), 1e-8)
    )
    take_top_spaced(calm_score, "observable_calm_window", pool_fraction=0.50)

    if len(selected) != int(count) or len({q.step for q in selected}) != int(count):
        raise RuntimeError("observable query selection did not produce exact unique count")
    return tuple(sorted(selected, key=lambda item: (item.step, item.category)))


def collect_unprivileged_renewal_labels(
    controller: Any,
    *,
    acquisition: str,
    task_spec: PlanValidityTaskSpec,
    args: Any,
    time_scale: PhysicalTimeScaleContract,
    queries_per_path: int = 24,
) -> list[dict[str, Any]]:
    """Collect paired keep/renew branch labels without reading hidden event clocks."""
    if acquisition not in ("uniform", "observable"):
        raise ValueError(f"unknown unprivileged acquisition mode: {acquisition}")
    rows: list[dict[str, Any]] = []
    for seed in args.branch_fit_seeds:
        schedule = jitter_schedule(args, seed)
        if acquisition == "uniform":
            opportunities = unprivileged_uniform_opportunities(
                seed=seed,
                horizon=args.horizon,
                schedule=schedule,
                count=queries_per_path,
                period_steps=time_scale.upper_period_steps,
                max_offset_steps=args.max_offset_steps,
                check_stride_steps=args.check_stride_steps,
                branch_window_steps=round(args.branch_window_seconds / time_scale.dt_seconds),
            )
        else:
            opportunities = unprivileged_observable_opportunities(
                controller,
                task_spec=task_spec,
                seed=seed,
                args=args,
                time_scale=time_scale,
                schedule=schedule,
                count=queries_per_path,
            )
        for query in opportunities:
            arms = [
                run_episode(
                    controller,
                    task_spec=task_spec,
                    seed=seed,
                    args=args,
                    time_scale=time_scale,
                    schedule=schedule,
                    branch_step=query.step,
                    renew=renew,
                )[1]
                for renew in (False, True)
            ]
            keep, renew_arm = arms
            if (
                not np.array_equal(keep["prefix"], renew_arm["prefix"])
                or not np.array_equal(keep["features"], renew_arm["features"])
            ):
                raise RuntimeError("unprivileged keep/renew branches diverged on prefix")
            rows.append({
                "seed": int(seed),
                "category": query.category,
                "opportunity_step": int(query.step),
                "source_step": None,
                "lag_steps": None,
                "feature_names": list(keep["feature_names"]),
                "causal_features": keep["features"].tolist(),
                "renew_ise_advantage": float(keep["branch_ise"] - renew_arm["branch_ise"]),
                "keep_tracking_squared_error_integral": float(keep["branch_ise"]),
                "renew_tracking_squared_error_integral": float(renew_arm["branch_ise"]),
                "renew_waypoint_delta": float(
                    np.linalg.norm(renew_arm["branch_subgoal"] - keep["old_subgoal"])
                ),
                "keep_primitive_steps_replayed": int(keep["episode_length"]),
                "renew_primitive_steps_replayed": int(renew_arm["episode_length"]),
                "branch_window_steps": 50,
                "downstream_upper_call_count_per_branch": 0,
                "prefix_max_abs_difference": 0.0,
                "candidate_feature_has_future_access": False,
                "candidate_feature_has_regime_label": False,
                "query_selection_uses_privileged_events": False,
            })
            count_work("renewal_label_pairs")
    return rows


def clock_schedule_for_budget(horizon: int, spec: BudgetScheduleSpec, offset_steps: int) -> tuple[int, ...]:
    spec.validate(horizon)
    if offset_steps < 0 or offset_steps > spec.max_offset_steps or offset_steps % spec.check_stride_steps != 0:
        raise ValueError("clock offset is outside the budget check grid")
    return tuple(
        0 if start == 0 else start + int(offset_steps)
        for start in range(0, int(horizon), spec.bin_steps)
    )


def run_multi_budget_episode(
    controller: Any,
    *,
    task_spec: PlanValidityTaskSpec,
    seed: int,
    args: Any,
    feature_time_scale: PhysicalTimeScaleContract,
    budget_spec: BudgetScheduleSpec,
    rule: TimingRule,
    latency_recorder: Any = None,
    risk_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one episode at an arbitrary exact bin budget while preserving feature units."""
    budget_spec.validate(args.horizon)
    if not np.isclose(feature_time_scale.dt_seconds, 0.01):
        raise ValueError("multi-budget episode requires the 10ms primitive clock")

    bin_steps = budget_spec.bin_steps
    max_offset = budget_spec.max_offset_steps
    stride = budget_spec.check_stride_steps
    grid_offsets = tuple(range(0, max_offset + 1, stride))

    schedule: tuple[int, ...] | None = None
    active_rule: TimingRule | None = rule
    if rule.kind == "clock":
        offset = next(
            (val for val in grid_offsets if val / max_offset >= rule.threshold - 1e-12),
            max_offset,
        )
        schedule = clock_schedule_for_budget(args.horizon, budget_spec, offset)
        active_rule = None

    task = task_spec.open(seed=int(seed), horizon=args.horizon)
    ledger = active_ledger()
    try:
        observation = task.reset()
        count_work("environment_resets")
        if not np.isclose(task.dt_seconds, feature_time_scale.dt_seconds):
            raise ValueError("plant sampling interval differs from feature time contract")
        adapter = task_spec.adapter(task, maximum_subgoal_delta=args.maximum_subgoal_delta)
        history = PointMazeRegimeFeatureBuilder(
            time_scale=feature_time_scale,
            task_dim=int(observation.task_measurement.size),
        )
        if latency_recorder is not None:
            reset_wall, reset_cpu = time.perf_counter(), time.process_time()
        history.reset(observation)
        controller.reset_recurrent_inference()
        if latency_recorder is not None:
            pending_wall = time.perf_counter() - reset_wall
            pending_cpu = time.process_time() - reset_cpu

        decision_set = set(schedule or ())
        subgoal = observation.achieved_goal.copy()
        decisions: list[int] = []
        errors: list[float] = []
        rewards: list[float] = []
        queries: list[dict[str, Any]] = []
        risk_distances: list[float] = []
        clipped_steps: list[bool] = []
        last_plan_step = -1
        planner_seconds = lower_seconds = score_seconds = 0.0

        for step in range(int(args.horizon)):
            if latency_recorder is not None:
                control_wall, control_cpu = time.perf_counter(), time.process_time()
            bin_index, offset = divmod(step, bin_steps)
            plan_now = step in decision_set if schedule is not None else step == 0
            check = (
                active_rule is not None
                and step > 0
                and decisions[-1] // bin_steps != bin_index
                and offset <= max_offset
                and offset % stride == 0
            )
            if check:
                started = time.perf_counter()
                if active_rule.kind == "threshold":
                    score = threshold_score(
                        active_rule.feature,
                        observation,
                        history,
                        step - last_plan_step,
                        feature_time_scale,
                    )
                else:
                    names, values, _ = _causal_plan_features(
                        observation=observation,
                        feature_builder=history,
                        subgoal=subgoal,
                        plan_age_steps=step - last_plan_step,
                        time_scale=feature_time_scale,
                    )
                    count_work("feature_vector_builds")
                    score = active_rule.score(
                        names,
                        values,
                        offset / max_offset,
                        (args.horizon - step) / args.horizon,
                    )
                count_work("trigger_score_checks")
                queries.append({
                    "step": step,
                    "offset": offset,
                    "score": score,
                    "stop": score >= active_rule.threshold,
                })
                plan_now = score >= active_rule.threshold or offset == max_offset
                score_seconds += time.perf_counter() - started

            if plan_now:
                upper_state = history.upper_state(observation, oracle_context=None)
                started = time.perf_counter()
                output = controller.plan_goal(upper_state, sample=False)
                count_work("planner_calls")
                planner_seconds += time.perf_counter() - started
                raw_goal = np.asarray(output["action"], dtype=np.float32)
                subgoal = adapter.decode(raw_goal, observation.achieved_goal)
                decisions.append(step)
                last_plan_step = step

            lower_state = history.lower_state(observation, subgoal=subgoal)
            started = time.perf_counter()
            output = controller.act_conditioned(lower_state, sample=False)
            count_work("lower_calls")
            lower_seconds += time.perf_counter() - started
            raw_action = np.asarray(output["action"], dtype=np.float32)
            requested = squash_box_action(raw_action, task.action_low, task.action_high)
            if latency_recorder is not None:
                latency_recorder.add(
                    time.perf_counter() - control_wall + pending_wall,
                    time.process_time() - control_cpu + pending_cpu,
                    first_step=step == 0,
                )
            if ledger is not None:
                ledger.count("environment_step_attempts")
            next_observation, reward, terminated, truncated, info = task.step(requested)
            if ledger is not None:
                ledger.count("environment_steps")
            done = bool(terminated or truncated)
            if done and step + 1 != int(args.horizon):
                raise RuntimeError("multi-budget episode ended before fixed horizon")
            errors.append(float(info["tracking_distance"]) ** 2 * feature_time_scale.dt_seconds)
            if risk_spec is not None:
                risk_distances.append(float(info["tracking_distance"]))
                disturbed = requested + observation.force
                clipped_steps.append(
                    bool(np.any((disturbed < task.action_low) | (disturbed > task.action_high)))
                )
            rewards.append(float(reward))
            observation = next_observation
            if latency_recorder is not None:
                update_wall, update_cpu = time.perf_counter(), time.process_time()
            history.update(observation)
            if latency_recorder is not None:
                pending_wall = time.perf_counter() - update_wall
                pending_cpu = time.process_time() - update_cpu
            count_work("history_updates")

        durations = np.diff([*decisions, int(args.horizon)])
        if (
            len(decisions) != budget_spec.calls_per_episode
            or [s // bin_steps for s in decisions] != list(range(budget_spec.calls_per_episode))
            or int(min(durations)) < bin_steps - max_offset
            or int(max(durations)) > bin_steps + max_offset
        ):
            raise RuntimeError("multi-budget execution violated its exact call contract")

        early_calls = sum(
            1 for s in decisions[1:] if (s % bin_steps) < max_offset
        )
        row: dict[str, Any] = {
            "seed": int(seed),
            "task": task_spec.name,
            "episode_length": int(args.horizon),
            "mode": rule.name,
            "calls_per_episode": budget_spec.calls_per_episode,
            "bin_steps": bin_steps,
            "max_offset_steps": max_offset,
            "check_stride_steps": stride,
            "episode_return": float(sum(rewards)),
            "tracking_squared_error_integral": float(sum(errors)),
            "decision_steps": decisions,
            "upper_decision_count": len(decisions),
            "trigger_score_checks": len(queries),
            "trigger_early_calls": int(early_calls),
            "option_duration_steps_min": int(min(durations)),
            "option_duration_steps_max": int(max(durations)),
            "planner_wall_seconds": planner_seconds,
            "lower_wall_seconds": lower_seconds,
            "trigger_score_wall_seconds": score_seconds,
            "protocol_valid": True,
        }
        if risk_spec is not None:
            from .plan_validity_reliability import episode_risk

            row["reliability"] = episode_risk(
                errors,
                risk_distances,
                clipped_steps,
                dt=feature_time_scale.dt_seconds,
                period=bin_steps,
                **risk_spec,
            )
        return row
    finally:
        task.environment.close()


def out_of_fold_legacy_scores(fit_rows: list[dict[str, Any]], fitted_model: dict[str, Any]) -> np.ndarray:
    """Recompute leave-one-fit-path-out scores for quantile threshold calibration."""
    from .pointmaze_compact_plan_validity import _base_matrix, _causal_validity_interactions
    from .pointmaze_plan_validity_branching import _ridge_fit_predict

    schema, matrix = _base_matrix(fit_rows)
    x, _ = _causal_validity_interactions(schema, matrix)
    target = np.asarray([row["renew_ise_advantage"] for row in fit_rows], dtype=np.float64)
    groups = np.asarray([row["seed"] for row in fit_rows], dtype=np.int64)
    alpha = float(fitted_model["model"]["alpha"])
    oof = np.empty(len(fit_rows), dtype=np.float64)
    for held_out in np.unique(groups):
        train = groups != held_out
        oof[~train], _ = _ridge_fit_predict(x[train], target[train], x[~train], alpha=alpha)
    return oof


def quantile_rule_candidates(
    base_rule: TimingRule,
    oof_scores: np.ndarray,
    *,
    prefix: str,
    quantiles: Iterable[float] = QUANTILE_CALIBRATION_GRID,
) -> list[TimingRule]:
    """Construct scalar-threshold candidates over a frozen predictor's OOF distribution."""
    if base_rule.kind != "legacy":
        raise ValueError("quantile calibration expects a legacy ridge predictor")
    rules = []
    for q in quantiles:
        if not 0.0 < float(q) < 1.0:
            raise ValueError("quantile must lie in (0, 1)")
        cut = float(np.quantile(oof_scores, float(q)))
        rules.append(
            TimingRule(
                name=f"{prefix}_q{int(round(float(q) * 100)):02d}",
                kind=base_rule.kind,
                view=base_rule.view,
                threshold=cut,
                feature=base_rule.feature,
                model=base_rule.model,
            )
        )
    return rules
