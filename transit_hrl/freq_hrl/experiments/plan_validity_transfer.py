"""Shared plan-renewal execution and label acquisition for two physical plants."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any
import time

import numpy as np
import torch

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from freq_hrl.domains.mujoco.reacher_tracking import ReacherSubgoalAdapter, ReacherTrackingTask
from freq_hrl.rl import (
    GoalConditionedActorCriticPPO, GoalConditionedPPOConfig, HierarchicalRolloutBuilder,
    train_frequency_separated_ppo,
)
from .pointmaze_budgeted_trigger import balanced_jitter_schedule
from .compute_budget import active_ledger, count_work, measured_phase
from .pointmaze_goal_validation import POINTMAZE_LOWER_ACTION_COST, _training_seed, pointmaze_goal_bounds, squash_box_action
from .pointmaze_plan_validity_branching import _causal_plan_features, plan_renewal_opportunities
from .pointmaze_plan_value_qualification import (
    PointMazeRegimeFeatureBuilder, _make_task, pointmaze_plan_value_checkpoint_rank,
)


PROTOCOL_VERSION = "same_algorithm_plan_validity_second_task_v1_prepared"
TASKS = ("pointmaze", "reacher_tracking")


@dataclass(frozen=True)
class PlanValidityTaskSpec:
    name: str
    options: dict
    pointmaze_env_id: str

    def open(self, *, seed, horizon):
        count_work("environment_opens")
        if self.name == "pointmaze":
            return _make_task(env_id=self.pointmaze_env_id, seed=seed, horizon=horizon, **self.options)
        if self.name == "reacher_tracking":
            return ReacherTrackingTask(seed=seed, horizon=horizon, **self.options)
        raise ValueError(f"unknown plan-validity task: {self.name}")

    def adapter(self, task, *, maximum_subgoal_delta):
        delta = np.full(2, maximum_subgoal_delta, dtype=np.float32)
        if self.name == "reacher_tracking":
            return ReacherSubgoalAdapter(maximum_delta=delta, action_cost=POINTMAZE_LOWER_ACTION_COST)
        low, high = pointmaze_goal_bounds(task.environment)
        return RelativeSubgoalAdapter(maximum_delta=delta, world_low=low, world_high=high,
                                       action_cost=POINTMAZE_LOWER_ACTION_COST)


def time_contract(args):
    return PhysicalTimeScaleContract(dt_seconds=0.01, upper_period_seconds=args.upper_period_seconds,
                                     history_seconds=args.history_seconds, fast_period_seconds=args.fast_period_seconds)


def jitter_schedule(args, seed):
    return balanced_jitter_schedule(seed=seed, horizon=args.horizon, period_steps=50, max_offset_steps=25)


def run_episode(controller: Any, *, task_spec, seed, args, time_scale, schedule=None,
                rule=None, sample=False, branch_step=None, renew=False,
                intervention_step=None, arm=None, feature_steps=(), latency_recorder=None):
    """One execution loop: scheduled training, adaptive timing, or paired keep/renew."""
    period = time_scale.upper_period_steps
    mode_name = rule.name if rule is not None else "scheduled"
    if rule is not None and rule.kind == "jitter":
        schedule, rule = jitter_schedule(args, seed), None
    elif rule is not None and rule.kind == "clock" and intervention_step is None:
        offset = next((value for value in range(0, 26, 5) if value / 25 >= rule.threshold), 25)
        schedule = tuple(0 if start == 0 else start + offset for start in range(0, args.horizon, 50))
        rule = None
    if (args.horizon % period or period != 50 or args.max_offset_steps != 25
        or args.check_stride_steps != 5 or not np.isclose(time_scale.dt_seconds, 0.01)):
        raise ValueError("shared renewal execution requires the 10ms/50/25/5 contract")
    if (schedule is None) == (rule is None) or (sample and schedule is None):
        raise ValueError("choose one scheduled or adaptive execution mode")
    if schedule is not None and (len(schedule) != args.horizon // period
        or [s // period for s in schedule] != list(range(args.horizon // period))
        or schedule[0] != 0 or any(s % period > args.max_offset_steps for s in schedule)):
        raise ValueError("schedule must preserve one call per bin")
    if branch_step is not None and (schedule is None or sample or branch_step in schedule
                                   or not 0 < branch_step < args.horizon - 50):
        raise ValueError("keep/renew query needs a noncolliding prefix and complete window")
    if intervention_step is not None and (rule is None or arm not in ("now", "wait_one_check")
        or not period <= intervention_step < args.horizon - period
        or intervention_step % period >= 25 or intervention_step % 5):
        raise ValueError("timing intervention must be a nondeadline check")
    task = task_spec.open(seed=seed, horizon=args.horizon)
    ledger = active_ledger()
    try:
        observation = task.reset()
        count_work("environment_resets")
        if not np.isclose(task.dt_seconds, time_scale.dt_seconds):
            raise ValueError("plant sampling interval differs from the renewal contract")
        adapter = task_spec.adapter(task, maximum_subgoal_delta=args.maximum_subgoal_delta)
        history = PointMazeRegimeFeatureBuilder(time_scale=time_scale, task_dim=observation.task_measurement.size)
        if latency_recorder is not None:
            reset_wall, reset_cpu = time.perf_counter(), time.process_time()
        history.reset(observation)
        controller.reset_recurrent_inference()
        if latency_recorder is not None:
            pending_wall, pending_cpu = time.perf_counter() - reset_wall, time.process_time() - reset_cpu
        builder = HierarchicalRolloutBuilder(gamma=float(controller.config.gamma)) if sample else None
        decision_set = set(schedule or ())
        subgoal = observation.achieved_goal.copy()
        decisions, errors, rewards, queries = [], [], [], []
        feature_rows = []
        feature_steps = set(feature_steps)
        captured = None
        last_plan_step = -1
        planner_seconds = lower_seconds = score_seconds = 0.0
        end_step = args.horizon if branch_step is None else branch_step + 50
        for step in range(end_step):
            if latency_recorder is not None:
                control_wall, control_cpu = time.perf_counter(), time.process_time()
            bin_index, offset = divmod(step, period)
            plan_now = step in decision_set if schedule is not None else step == 0
            names = values = None
            query = step == branch_step or step == intervention_step
            check = (rule is not None and step > 0 and decisions[-1] // period != bin_index
                     and offset <= 25 and offset % 5 == 0)
            if query or check or step in feature_steps:
                started = time.perf_counter()
                if query or step in feature_steps or rule.kind != "threshold":
                    names, values, _ = _causal_plan_features(
                        observation=observation, feature_builder=history, subgoal=subgoal,
                        plan_age_steps=step - last_plan_step, time_scale=time_scale)
                    count_work("feature_vector_builds")
                if check:
                    score = (threshold_score(rule.feature, observation, history, step - last_plan_step, time_scale)
                             if rule.kind == "threshold" else
                             rule.score(names, values, offset / 25, (args.horizon - step) / args.horizon))
                    count_work("trigger_score_checks")
                    queries.append({"step": step, "offset": offset, "score": score,
                                    "stop": score >= rule.threshold})
                    plan_now = score >= rule.threshold or offset == 25
                score_seconds += time.perf_counter() - started
                if query:
                    captured = {"feature_names": names, "features": values.copy(),
                                "prefix": np.concatenate((observation.physical, observation.achieved_goal,
                                    observation.target, observation.task_measurement, history.history,
                                    subgoal, [last_plan_step])).astype(np.float64),
                                "old_subgoal": subgoal.copy()}
                if step in feature_steps:
                    feature_rows.append({"seed": seed, "opportunity_step": step,
                        "feature_names": list(names), "causal_features": values.tolist(),
                        "candidate_feature_has_future_access": False,
                        "candidate_feature_has_regime_label": False})
            if branch_step is not None and step >= branch_step:
                plan_now = step == branch_step and renew
            if step == intervention_step:
                if not check:
                    raise RuntimeError("intervention is not visited under the factual prefix")
                plan_now = arm == "now"
            if plan_now:
                upper_state = history.upper_state(observation, oracle_context=None)
                started = time.perf_counter()
                output = controller.plan_goal(upper_state, sample=sample)
                count_work("planner_calls")
                planner_seconds += time.perf_counter() - started
                raw_goal = np.asarray(output["action"], dtype=np.float32)
                subgoal = adapter.decode(raw_goal, observation.achieved_goal)
                decisions.append(step)
                last_plan_step = step
                if builder is not None:
                    builder.begin_upper(state=upper_state, action=raw_goal,
                                         logp=float(output["logp"]), value=float(output["value"]))
            if step == branch_step:
                captured["branch_subgoal"] = subgoal.copy()
            lower_state = history.lower_state(observation, subgoal=subgoal)
            started = time.perf_counter()
            output = controller.act_conditioned(lower_state, sample=sample)
            count_work("lower_calls")
            lower_seconds += time.perf_counter() - started
            raw_action = np.asarray(output["action"], dtype=np.float32)
            requested = squash_box_action(raw_action, task.action_low, task.action_high)
            if latency_recorder is not None:
                latency_recorder.add(time.perf_counter() - control_wall + pending_wall,
                    time.process_time() - control_cpu + pending_cpu, first_step=step == 0)
            if ledger is not None:
                ledger.count("environment_step_attempts")
            next_observation, reward, terminated, truncated, info = task.step(requested)
            if ledger is not None:
                ledger.count("environment_steps")
            done = bool(terminated or truncated)
            if done and step + 1 != args.horizon:
                raise RuntimeError("plan-validity task terminated before the fixed horizon")
            if builder is not None:
                intrinsic = adapter.intrinsic_reward(achieved_before=observation.achieved_goal,
                    achieved_after=next_observation.achieved_goal, subgoal=subgoal, action=requested)
                builder.add_lower(state=lower_state, action=raw_action, logp=float(output["logp"]),
                    value=float(output["value"]), reward=float(intrinsic), upper_reward=float(reward),
                    done=done, lower_done=bool(done or step + 1 in decision_set), cost=0.0)
            errors.append(float(info["tracking_distance"]) ** 2 * time_scale.dt_seconds)
            rewards.append(float(reward))
            observation = next_observation
            if latency_recorder is not None:
                update_wall, update_cpu = time.perf_counter(), time.process_time()
            history.update(observation)
            if latency_recorder is not None:
                pending_wall, pending_cpu = time.perf_counter() - update_wall, time.process_time() - update_cpu
            count_work("history_updates")
        if (branch_step is not None or intervention_step is not None) and captured is None:
            raise RuntimeError("paired execution did not reach its causal query")
        durations = np.diff([*decisions, end_step])
        if branch_step is None and ([s // period for s in decisions] != list(range(args.horizon // period))
                                   or min(durations) < 25 or max(durations) > 75):
            raise RuntimeError("adaptive execution changed the fixed call budget")
        batch = None
        if builder is not None:
            builder.finish(terminal=True)
            batch = builder.build()
        row = {"seed": int(seed), "task": task_spec.name, "episode_length": end_step,
               "mode": mode_name,
               "episode_return": float(sum(rewards)), "tracking_squared_error_integral": float(sum(errors)),
               "decision_steps": decisions, "upper_decision_count": len(decisions), "queries": queries,
               "trigger_score_checks": len(queries), "option_duration_steps_min": int(min(durations)),
               "option_duration_steps_max": int(max(durations)),
               "planner_wall_seconds": planner_seconds, "lower_wall_seconds": lower_seconds,
               "trigger_score_wall_seconds": score_seconds, "protocol_valid": True,
               **(captured or {})}
        if branch_step is not None:
            row["branch_ise"] = float(sum(errors[branch_step:]))
        if feature_steps:
            row["feature_rows"] = feature_rows
        return batch, row
    finally:
        task.environment.close()


def threshold_score(feature, observation, history, age_steps, time_scale):
    """Compute only the scalar a simple threshold actually needs."""
    if feature == "plan_age_fraction":
        return age_steps / time_scale.upper_period_steps
    if feature == "tracking_distance":
        return float(np.linalg.norm(observation.target_error))
    if feature == "force_current_norm":
        return float(np.linalg.norm(np.asarray(observation.force, dtype=np.float64)))
    if feature == "target_velocity_change_norm":
        target = history.history.reshape(time_scale.history_steps, -1).astype(np.float64)[:, :2]
        return float(np.linalg.norm((target[-1] - target[-2]) / time_scale.dt_seconds
            - (target[-1] - target[-26]) / (25 * time_scale.dt_seconds)))
    raise ValueError(f"unknown threshold feature: {feature}")


def renewal_queries(task_spec, seed, args, schedule):
    task = task_spec.open(seed=seed, horizon=args.horizon)
    try:
        return plan_renewal_opportunities(horizon=args.horizon, period_steps=50, branch_window_steps=50,
            regime_change_steps=task.driver.regime_change_steps, force_pulse_steps=task.driver.pulse_start_steps,
            distractor_change_steps=task.driver.distractor_change_steps, seed=seed,
            max_events_per_class=args.max_events_per_class, blocked_steps=schedule)
    finally:
        task.environment.close()


def collect_renewal_features(controller, *, task_spec, args, time_scale):
    """Thresholds acquire the same causal query features without unused branch labels."""
    rows = []
    for seed in args.branch_fit_seeds:
        schedule = jitter_schedule(args, seed)
        queries = renewal_queries(task_spec, seed, args, schedule)
        _, trajectory = run_episode(controller, task_spec=task_spec, seed=seed, args=args,
            time_scale=time_scale, schedule=schedule, feature_steps=[query.step for query in queries])
        rows.extend(trajectory["feature_rows"])
    return rows


def collect_renewal_labels(controller, *, task_spec, args, time_scale):
    rows = []
    for seed in args.branch_fit_seeds:
        schedule = jitter_schedule(args, seed)
        opportunities = renewal_queries(task_spec, seed, args, schedule)
        for query in opportunities:
            arms = [run_episode(controller, task_spec=task_spec, seed=seed, args=args,
                                 time_scale=time_scale, schedule=schedule,
                                 branch_step=query.step, renew=renew)[1] for renew in (False, True)]
            keep, renew = arms
            if (not np.array_equal(keep["prefix"], renew["prefix"])
                or not np.array_equal(keep["features"], renew["features"])):
                raise RuntimeError("keep/renew labels do not share an identical causal prefix")
            rows.append({"seed": seed, "category": query.category, "opportunity_step": query.step,
                         "source_step": query.source_step, "lag_steps": query.lag_steps,
                         "feature_names": list(keep["feature_names"]), "causal_features": keep["features"].tolist(),
                         "renew_ise_advantage": keep["branch_ise"] - renew["branch_ise"],
                         "keep_tracking_squared_error_integral": keep["branch_ise"],
                         "renew_tracking_squared_error_integral": renew["branch_ise"],
                         "renew_waypoint_delta": float(np.linalg.norm(renew["branch_subgoal"] - keep["old_subgoal"])),
                         "keep_primitive_steps_replayed": keep["episode_length"],
                         "renew_primitive_steps_replayed": renew["episode_length"],
                         "branch_window_steps": 50, "downstream_upper_call_count_per_branch": 0,
                         "prefix_max_abs_difference": 0.0,
                         "candidate_feature_has_future_access": False,
                         "candidate_feature_has_regime_label": False})
            count_work("renewal_label_pairs")
    return rows


def train_shared_controller(task_spec, args, time_scale):
    task = task_spec.open(seed=0, horizon=args.horizon)
    try:
        observation = task.reset()
        count_work("environment_resets")
        if (observation.physical.shape != (4,) or observation.achieved_goal.shape != (2,)
            or observation.task_measurement.shape != (6,) or task.action_low.shape != (2,)):
            raise ValueError("two-task controller expects four physical and six exogenous channels")
        task_metadata = task.diagnostics()
    finally:
        task.environment.close()
    shared_dim = 4 + 2 + 6 * time_scale.history_steps
    torch.manual_seed(args.optimizer_seed)
    np.random.seed(args.optimizer_seed % (2 ** 32 - 1))
    model = GoalConditionedActorCriticPPO(GoalConditionedPPOConfig(
        upper_state_dim=shared_dim, lower_state_dim=shared_dim, goal_dim=2, action_dim=2,
        hidden_dim=args.reference_hidden_dim, learning_rate=args.learning_rate,
        epochs=4, minibatch_size=1024, init_log_std=-0.7))

    def rollout(controller, seed, sample):
        role = "train_rollout" if sample else (
            "selection_rollout" if seed in args.selection_seeds else "diagnostic_rollout")
        with measured_phase(role):
            return run_episode(controller, task_spec=task_spec, seed=seed, args=args, time_scale=time_scale,
                               schedule=jitter_schedule(args, seed), sample=sample)

    payload, _, trained = train_frequency_separated_ppo(
        model=model, train_seeds=args.train_seeds, selection_seeds=args.selection_seeds,
        eval_seeds=args.timing_selection_seeds, iterations=args.iterations, rollout_fn=rollout,
        objective_fn=lambda row: row["episode_return"], policy="shared_goal_conditioned_ppo",
        domain=task_spec.name,
        training_seed_fn=lambda root, iteration: _training_seed(
            optimizer_seed=args.optimizer_seed, rollout_root=root, iteration=iteration),
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        checkpoint_minimum_iteration=0, checkpoint_rank_fn=pointmaze_plan_value_checkpoint_rank,
        checkpoint_rank_names=("negative_mean_tracking_squared_error_integral", "mean_dense_return"),
        checkpoint_rank_contract="shared_lexicographic_negative_ise_then_return",
        metadata={"protocol_version": PROTOCOL_VERSION, "task_spec": task_metadata,
                  "capacity": {"actual_parameter_count": model.trainable_parameter_count},
                  "runtime_versions": {name: version(name) for name in ("gymnasium", "mujoco", "torch", "numpy")}})
    count_work("controller_optimizer_updates", payload["gradient_updates_train"])
    return payload, trained
