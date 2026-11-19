"""Route B components: generic neural validity predictor, external timing baselines, native periodic HRL, and CEM planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.rl import (
    GoalConditionedActorCriticPPO,
    GoalConditionedPPOConfig,
    train_frequency_separated_ppo,
)

from .compute_budget import count_work, measured_phase
from .plan_validity_transfer import (
    PROTOCOL_VERSION,
    PlanValidityTaskSpec,
    run_episode,
)
from .pointmaze_compact_plan_validity import _base_matrix, _select_named_features
from .pointmaze_goal_validation import _training_seed
from .pointmaze_plan_value_qualification import pointmaze_plan_value_checkpoint_rank
from .pointmaze_timing_baselines import TimingRule, select_rule


GENERIC_RAW_FEATURE_NAMES: tuple[str, ...] = (
    "physical_0",
    "physical_1",
    "physical_2",
    "physical_3",
    "target_error_0",
    "target_error_1",
    "waypoint_error_0",
    "waypoint_error_1",
    "plan_age_fraction",
    "target_current_0",
    "target_current_1",
    "force_current_0",
    "force_current_1",
    "target_velocity_001_0",
    "target_velocity_001_1",
    "target_velocity_010_0",
    "target_velocity_010_1",
    "target_velocity_025_0",
    "target_velocity_025_1",
    "target_velocity_050_0",
    "target_velocity_050_1",
    "force_rms_010_0",
    "force_rms_010_1",
    "force_rms_025_0",
    "force_rms_025_1",
)

STATE_ONLY_RENEWAL_NAMES: tuple[str, ...] = (
    "physical_0",
    "physical_1",
    "physical_2",
    "physical_3",
    "target_error_0",
    "target_error_1",
    "target_current_0",
    "target_current_1",
    "force_current_0",
    "force_current_1",
    "target_velocity_001_0",
    "target_velocity_001_1",
    "target_velocity_010_0",
    "target_velocity_010_1",
    "target_velocity_025_0",
    "target_velocity_025_1",
    "force_rms_010_0",
    "force_rms_010_1",
)


@dataclass
class RouteBTimingRule(TimingRule):
    """TimingRule extension supporting generic raw-channel MLPs and critic heads."""

    def score(
        self,
        names: tuple,
        values: np.ndarray,
        offset_fraction: float,
        remaining_fraction: float,
    ) -> float:
        if self.kind in ("raw_mlp", "gate_mlp"):
            names_in = tuple(self.model["input_names"])
            base = _select_named_features(
                tuple(names), np.asarray(values, dtype=np.float64)[None, :], names_in
            )[0]
            if self.kind == "gate_mlp":
                base = np.r_[base, float(offset_fraction), float(remaining_fraction)]
            mean = np.asarray(self.model["feature_mean"], dtype=np.float64)
            scale = np.asarray(self.model["feature_scale"], dtype=np.float64)
            z = (base - mean) / scale
            for layer in self.model["layers"][:-1]:
                z = np.tanh(z @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"]))
            last = self.model["layers"][-1]
            return float((z @ np.asarray(last["weight"]).T + np.asarray(last["bias"]))[0])
        if self.kind == "oc_advantage":
            q_names = tuple(self.model["q_input_names"])
            v_names = tuple(self.model["v_input_names"])
            arr = np.asarray(values, dtype=np.float64)[None, :]
            xq = _select_named_features(tuple(names), arr, q_names)[0]
            xv = _select_named_features(tuple(names), arr, v_names)[0]
            zq = (xq - np.asarray(self.model["q_mean"])) / np.asarray(self.model["q_scale"])
            zv = (xv - np.asarray(self.model["v_mean"])) / np.asarray(self.model["v_scale"])
            for layer in self.model["q_layers"][:-1]:
                zq = np.tanh(zq @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"]))
            q_last = self.model["q_layers"][-1]
            q_cost = float((zq @ np.asarray(q_last["weight"]).T + np.asarray(q_last["bias"]))[0])
            for layer in self.model["v_layers"][:-1]:
                zv = np.tanh(zv @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"]))
            v_last = self.model["v_layers"][-1]
            v_cost = float((zv @ np.asarray(v_last["weight"]).T + np.asarray(v_last["bias"]))[0])
            # Positive advantage means keeping current option has higher predicted 50-step ISE than renewing
            return float(q_cost - v_cost)
        return super().score(names, values, offset_fraction, remaining_fraction)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteBTimingRule":
        return cls(**payload)


def _build_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )


def _serialize_mlp(net: nn.Sequential) -> list[dict[str, Any]]:
    return [
        {
            "weight": layer.weight.detach().cpu().numpy().tolist(),
            "bias": layer.bias.detach().cpu().numpy().tolist(),
        }
        for layer in net
        if isinstance(layer, nn.Linear)
    ]


def _train_regression_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    seed: int,
    hidden_dim: int = 32,
    epochs: int = 220,
    lr: float = 3e-3,
    weight_decay: float = 1e-2,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = train_x.mean(axis=0)
    scale = np.where(train_x.std(axis=0) > 1e-6, train_x.std(axis=0), 1.0)
    zx = torch.as_tensor((train_x - mean) / scale, dtype=torch.float32)
    zy = torch.as_tensor(train_y, dtype=torch.float32).view(-1, 1)
    ze = torch.as_tensor((eval_x - mean) / scale, dtype=torch.float32)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        net = _build_mlp(train_x.shape[1], hidden_dim)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.SmoothL1Loss(beta=0.05)
        for _ in range(int(epochs)):
            pred = net(zx)
            loss = loss_fn(pred, zy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            count_work("generic_mlp_optimizer_updates")
        with torch.no_grad():
            eval_pred = net(ze).squeeze(-1).cpu().numpy().astype(np.float64)
        layers = _serialize_mlp(net)
        param_count = sum(p.numel() for p in net.parameters())
    return eval_pred, {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "layers": layers,
        "parameter_count": int(param_count),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
    }


def fit_generic_mlp_validity_predictor(
    fit_rows: list[dict[str, Any]],
    *,
    seed: int,
    hidden_dim: int = 32,
    epochs: int = 220,
    weight_decay_grid: Iterable[float] = (1e-3, 1e-2, 5e-2),
    threshold_quantile: float = 0.75,
) -> tuple[RouteBTimingRule, np.ndarray]:
    """Fit a 2-layer MLP on 25 raw state/history channels without hand-crafted dot-products."""
    if not 0.0 < float(threshold_quantile) < 1.0:
        raise ValueError("threshold_quantile must lie in (0, 1)")
    schema, matrix = _base_matrix(fit_rows)
    x = _select_named_features(schema, matrix, GENERIC_RAW_FEATURE_NAMES)
    y = np.asarray([r["renew_ise_advantage"] for r in fit_rows], dtype=np.float64)
    groups = np.asarray([r["seed"] for r in fit_rows], dtype=np.int64)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("generic MLP cross-validation requires at least two fit paths")

    wd_candidates = tuple(sorted(set(map(float, weight_decay_grid))))
    cv_mse: dict[str, float] = {}
    oof_by_wd: dict[float, np.ndarray] = {}
    for wd_idx, wd in enumerate(wd_candidates):
        oof = np.empty(len(fit_rows), dtype=np.float64)
        for fold_idx, held_out in enumerate(unique_groups):
            train_mask = groups != held_out
            oof[~train_mask], _ = _train_regression_mlp(
                x[train_mask],
                y[train_mask],
                x[~train_mask],
                seed=int(seed) + wd_idx * 100 + fold_idx,
                hidden_dim=hidden_dim,
                epochs=epochs,
                weight_decay=wd,
            )
        oof_by_wd[wd] = oof
        cv_mse[str(wd)] = float(np.mean(np.square(y - oof)))

    best_wd = min(wd_candidates, key=lambda wd: (cv_mse[str(wd)], wd))
    best_oof = oof_by_wd[best_wd]
    _, model = _train_regression_mlp(
        x,
        y,
        x,
        seed=int(seed) + 999,
        hidden_dim=hidden_dim,
        epochs=epochs,
        weight_decay=best_wd,
    )
    threshold = float(np.quantile(best_oof, float(threshold_quantile)))
    model.update({
        "input_names": list(GENERIC_RAW_FEATURE_NAMES),
        "cv_mse": cv_mse,
        "selected_weight_decay": float(best_wd),
        "oof_mse": float(cv_mse[str(best_wd)]),
        "threshold_quantile": float(threshold_quantile),
        "threshold_source": "branch_fit_leave_one_path_out_predictions",
    })
    rule = RouteBTimingRule(
        name="generic_mlp_validity",
        kind="raw_mlp",
        view="generic_raw_25",
        threshold=threshold,
        model=model,
    )
    return rule, best_oof


def train_native_periodic_controller(
    task_spec: Any,
    args: Any,
    time_scale: PhysicalTimeScaleContract,
) -> tuple[dict[str, Any], GoalConditionedActorCriticPPO]:
    """Train a dedicated periodic (offset=0) goal-conditioned HRL controller from scratch."""
    task = task_spec.open(seed=0, horizon=args.horizon)
    try:
        observation = task.reset()
        count_work("environment_resets")
        if (
            observation.physical.shape != (4,)
            or observation.achieved_goal.shape != (2,)
            or observation.task_measurement.shape != (6,)
            or task.action_low.shape != (2,)
        ):
            raise ValueError("periodic controller expects 4 physical and 6 exogenous channels")
        task_metadata = task.diagnostics()
    finally:
        task.environment.close()

    shared_dim = 4 + 2 + 6 * time_scale.history_steps
    torch.manual_seed(int(args.optimizer_seed))
    np.random.seed(int(args.optimizer_seed) % (2 ** 32 - 1))
    model = GoalConditionedActorCriticPPO(
        GoalConditionedPPOConfig(
            upper_state_dim=shared_dim,
            lower_state_dim=shared_dim,
            goal_dim=2,
            action_dim=2,
            hidden_dim=args.reference_hidden_dim,
            learning_rate=args.learning_rate,
            epochs=4,
            minibatch_size=1024,
            init_log_std=-0.7,
        )
    )
    periodic_schedule = tuple(range(0, int(args.horizon), 50))

    def rollout(controller, seed, sample):
        role = (
            "train_rollout"
            if sample
            else ("selection_rollout" if seed in args.selection_seeds else "diagnostic_rollout")
        )
        with measured_phase(role):
            return run_episode(
                controller,
                task_spec=task_spec,
                seed=seed,
                args=args,
                time_scale=time_scale,
                schedule=periodic_schedule,
                sample=sample,
            )

    payload, _, trained = train_frequency_separated_ppo(
        model=model,
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        eval_seeds=args.timing_selection_seeds,
        iterations=args.iterations,
        rollout_fn=rollout,
        objective_fn=lambda row: row["episode_return"],
        policy="native_periodic_goal_conditioned_ppo",
        domain=task_spec.name,
        training_seed_fn=lambda root, iteration: _training_seed(
            optimizer_seed=args.optimizer_seed, rollout_root=root, iteration=iteration
        ),
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        checkpoint_minimum_iteration=0,
        checkpoint_rank_fn=pointmaze_plan_value_checkpoint_rank,
        checkpoint_rank_names=(
            "negative_mean_tracking_squared_error_integral",
            "mean_dense_return",
        ),
        checkpoint_rank_contract="shared_lexicographic_negative_ise_then_return",
        metadata={
            "protocol_version": PROTOCOL_VERSION,
            "schedule_contract": "strict_periodic_offset_0",
            "task_spec": task_metadata,
            "capacity": {"actual_parameter_count": model.trainable_parameter_count},
            "runtime_versions": {
                name: version(name) for name in ("gymnasium", "mujoco", "torch", "numpy")
            },
        },
    )
    count_work("controller_optimizer_updates", payload["gradient_updates_train"])
    return payload, trained


def _rollout_stochastic_gate_episode(
    controller: Any,
    gate_net: nn.Module,
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    task_spec: Any,
    seed: int,
    args: Any,
    time_scale: PhysicalTimeScaleContract,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run one online episode under a stochastic binary temporal gate (TEMPLE style)."""
    from .pointmaze_goal_validation import squash_box_action
    from .pointmaze_plan_validity_branching import _causal_plan_features
    from .pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder

    period = time_scale.upper_period_steps
    max_offset = int(args.max_offset_steps)
    stride = int(args.check_stride_steps)
    task = task_spec.open(seed=int(seed), horizon=args.horizon)
    try:
        obs = task.reset()
        count_work("environment_resets")
        adapter = task_spec.adapter(task, maximum_subgoal_delta=args.maximum_subgoal_delta)
        history = PointMazeRegimeFeatureBuilder(
            time_scale=time_scale, task_dim=int(obs.task_measurement.size)
        )
        history.reset(obs)
        controller.reset_recurrent_inference()
        subgoal = obs.achieved_goal.copy()
        last_plan_step = -1
        decisions: list[int] = []
        step_ises: list[float] = []
        gate_events: list[dict[str, Any]] = []

        for step in range(int(args.horizon)):
            bin_index, offset = divmod(step, period)
            plan_now = step == 0
            if (
                step > 0
                and decisions[-1] // period != bin_index
                and offset <= max_offset
                and offset % stride == 0
            ):
                names, values, _ = _causal_plan_features(
                    observation=obs,
                    feature_builder=history,
                    subgoal=subgoal,
                    plan_age_steps=step - last_plan_step,
                    time_scale=time_scale,
                )
                count_work("feature_vector_builds")
                base = _select_named_features(
                    tuple(names),
                    np.asarray(values, dtype=np.float64)[None, :],
                    GENERIC_RAW_FEATURE_NAMES,
                )[0]
                feat = np.r_[base, offset / max_offset, (args.horizon - step) / args.horizon]
                if offset == max_offset:
                    plan_now = True
                else:
                    z = torch.as_tensor(
                        (feat - feature_mean) / feature_scale, dtype=torch.float32
                    ).unsqueeze(0)
                    with torch.no_grad():
                        logit = float(gate_net(z).item())
                    prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0))))
                    action_call = bool(rng.random() < prob)
                    gate_events.append({
                        "step": step,
                        "feat": feat,
                        "action": 1.0 if action_call else 0.0,
                        "prob": prob,
                    })
                    plan_now = action_call

            if plan_now:
                u_state = history.upper_state(obs, oracle_context=None)
                out = controller.plan_goal(u_state, sample=False)
                count_work("planner_calls")
                subgoal = adapter.decode(
                    np.asarray(out["action"], dtype=np.float32), obs.achieved_goal
                )
                decisions.append(step)
                last_plan_step = step

            l_state = history.lower_state(obs, subgoal=subgoal)
            l_out = controller.act_conditioned(l_state, sample=False)
            count_work("lower_calls")
            req = squash_box_action(
                np.asarray(l_out["action"], dtype=np.float32),
                task.action_low,
                task.action_high,
            )
            nxt, _, terminated, truncated, info = task.step(req)
            count_work("environment_steps")
            if (terminated or truncated) and step + 1 != int(args.horizon):
                raise RuntimeError("TEMPLE gate episode terminated early")
            step_ises.append(float(info["tracking_distance"]) ** 2 * time_scale.dt_seconds)
            obs = nxt
            history.update(obs)
            count_work("history_updates")

        # Assign discounted negative tracking-loss-to-go over next 100 steps to each gate decision
        ises = np.asarray(step_ises, dtype=np.float64)
        for ev in gate_events:
            s = ev["step"]
            window = ises[s : min(len(ises), s + 100)]
            discounts = 0.99 ** np.arange(len(window), dtype=np.float64)
            ev["return_to_go"] = float(-np.sum(window * discounts))

        return {
            "episode_ise": float(ises.sum()),
            "gate_events": gate_events,
        }
    finally:
        task.environment.close()


def fit_temple_gate_rule(
    controller: Any,
    *,
    task_spec: Any,
    args: Any,
    time_scale: PhysicalTimeScaleContract,
    seed: int,
    rounds: int = 20,
    hidden_dim: int = 32,
) -> tuple[RouteBTimingRule, dict[str, Any]]:
    """Train a budget-matched TEMPLE temporal gate via online PPO/actor-critic updates."""
    from .plan_validity_transfer import jitter_schedule

    # Warmup normalization statistics from one scheduled rollout per fit path (8 episodes)
    warmup_feats: list[np.ndarray] = []
    for fseed in args.branch_fit_seeds:
        sched = jitter_schedule(args, fseed)
        cands = [
            step
            for step in range(50, args.horizon - 50)
            if (step % 50) <= 25 and (step % 50) % 5 == 0 and step not in sched
        ]
        _, traj = run_episode(
            controller,
            task_spec=task_spec,
            seed=fseed,
            args=args,
            time_scale=time_scale,
            schedule=sched,
            feature_steps=cands,
        )
        for r in traj["feature_rows"]:
            s_idx = int(r["opportunity_step"])
            base = _select_named_features(
                tuple(r["feature_names"]),
                np.asarray(r["causal_features"], dtype=np.float64)[None, :],
                GENERIC_RAW_FEATURE_NAMES,
            )[0]
            warmup_feats.append(
                np.r_[base, (s_idx % 50) / 25.0, (args.horizon - s_idx) / args.horizon]
            )
    wmat = np.vstack(warmup_feats)
    f_mean = wmat.mean(axis=0)
    f_scale = np.where(wmat.std(axis=0) > 1e-6, wmat.std(axis=0), 1.0)

    in_dim = len(GENERIC_RAW_FEATURE_NAMES) + 2
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7_701]))
    candidates: list[RouteBTimingRule] = []

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 301)
        gate_net = _build_mlp(in_dim, hidden_dim)
        # Initialize bias slightly negative so early-check probability starts around ~0.30
        with torch.no_grad():
            gate_net[-1].bias.fill_(-0.85)
        value_net = _build_mlp(in_dim, hidden_dim)
        opt_actor = torch.optim.Adam(gate_net.parameters(), lr=2.5e-3)
        opt_critic = torch.optim.Adam(value_net.parameters(), lr=3e-3)

        for r_idx in range(int(rounds)):
            batch_events: list[dict[str, Any]] = []
            for fseed in args.branch_fit_seeds:
                roll_seed = _training_seed(
                    optimizer_seed=int(seed), rollout_root=int(fseed), iteration=r_idx
                )
                ep = _rollout_stochastic_gate_episode(
                    controller,
                    gate_net,
                    feature_mean=f_mean,
                    feature_scale=f_scale,
                    task_spec=task_spec,
                    seed=roll_seed,
                    args=args,
                    time_scale=time_scale,
                    rng=rng,
                )
                batch_events.extend(ep["gate_events"])

            if batch_events:
                feats = np.vstack([ev["feat"] for ev in batch_events])
                z = torch.as_tensor((feats - f_mean) / f_scale, dtype=torch.float32)
                acts = torch.as_tensor(
                    [ev["action"] for ev in batch_events], dtype=torch.float32
                ).view(-1, 1)
                old_p = torch.as_tensor(
                    np.clip([ev["prob"] for ev in batch_events], 1e-4, 1.0 - 1e-4),
                    dtype=torch.float32,
                ).view(-1, 1)
                rtg = torch.as_tensor(
                    [ev["return_to_go"] for ev in batch_events], dtype=torch.float32
                ).view(-1, 1)

                for _ in range(4):
                    v_pred = value_net(z)
                    v_loss = nn.functional.mse_loss(v_pred, rtg)
                    opt_critic.zero_grad()
                    v_loss.backward()
                    opt_critic.step()

                with torch.no_grad():
                    adv = rtg - value_net(z)
                    adv = (adv - adv.mean()) / (adv.std() + 1e-6)

                old_logp = acts * torch.log(old_p) + (1.0 - acts) * torch.log(1.0 - old_p)
                for _ in range(4):
                    logits = gate_net(z)
                    probs = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
                    logp = acts * torch.log(probs) + (1.0 - acts) * torch.log(1.0 - probs)
                    ratio = torch.exp(logp - old_logp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
                    ent = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
                    a_loss = -(torch.min(surr1, surr2) + 0.01 * ent).mean()
                    opt_actor.zero_grad()
                    a_loss.backward()
                    opt_actor.step()
                    count_work("temple_gate_optimizer_updates")

            if (r_idx + 1) % 5 == 0 or r_idx == int(rounds) - 1:
                layers = _serialize_mlp(gate_net)
                for cut_name, cut in (("m02", -0.2), ("z00", 0.0), ("p02", 0.2)):
                    candidates.append(
                        RouteBTimingRule(
                            name=f"temple_gate_r{r_idx + 1}_{cut_name}",
                            kind="gate_mlp",
                            view="generic_raw_25_plus_phase",
                            threshold=float(cut),
                            model={
                                "input_names": list(GENERIC_RAW_FEATURE_NAMES),
                                "feature_mean": f_mean.tolist(),
                                "feature_scale": f_scale.tolist(),
                                "layers": layers,
                            },
                        )
                    )

    sel_cache: dict[str, list[dict]] = {}

    def eval_sel(rule: TimingRule) -> list[dict]:
        if rule.name not in sel_cache:
            sel_cache[rule.name] = [
                run_episode(
                    controller,
                    task_spec=task_spec,
                    seed=s,
                    args=args,
                    time_scale=time_scale,
                    rule=rule,
                )[1]
                for s in args.timing_selection_seeds
            ]
        return sel_cache[rule.name]

    selected, table = select_rule(candidates, eval_sel)
    return selected, {
        "candidate_count": len(candidates),
        "selected_name": selected.name,
        "selection_table": table,
    }


def fit_deliberation_option_critic_rule(
    controller: Any,
    *,
    task_spec: Any,
    args: Any,
    time_scale: PhysicalTimeScaleContract,
    seed: int,
    rollout_episodes: int = 160,
    hidden_dim: int = 32,
) -> tuple[RouteBTimingRule, dict[str, Any]]:
    """Fit single-trajectory Option-Critic intra-option Q vs renewal V with deliberation margin."""
    from .pointmaze_goal_validation import squash_box_action
    from .pointmaze_plan_validity_branching import _causal_plan_features
    from .pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 8_801]))
    period = time_scale.upper_period_steps
    v_samples_x: list[np.ndarray] = []
    v_samples_y: list[float] = []
    q_samples_x: list[np.ndarray] = []
    q_samples_y: list[float] = []

    # Collect single-trajectory variable-duration rollouts without paired branching!
    for ep_idx in range(int(rollout_episodes)):
        base_seed = int(args.branch_fit_seeds[ep_idx % len(args.branch_fit_seeds)])
        ep_seed = _training_seed(
            optimizer_seed=int(seed) + 17, rollout_root=base_seed, iteration=ep_idx
        )
        # Randomize within-bin call offsets across {0, 5, 10, 15, 20, 25} to cover option ages
        sched = tuple(
            0 if b == 0 else 50 * b + 5 * int(rng.integers(0, 6))
            for b in range(int(args.horizon) // period)
        )
        sched_set = set(sched)
        task = task_spec.open(seed=ep_seed, horizon=args.horizon)
        try:
            obs = task.reset()
            count_work("environment_resets")
            adapter = task_spec.adapter(task, maximum_subgoal_delta=args.maximum_subgoal_delta)
            history = PointMazeRegimeFeatureBuilder(
                time_scale=time_scale, task_dim=int(obs.task_measurement.size)
            )
            history.reset(obs)
            controller.reset_recurrent_inference()
            subgoal = obs.achieved_goal.copy()
            last_plan_step = -1
            step_ises: list[float] = []
            renewal_records: list[tuple[int, np.ndarray]] = []
            continue_records: list[tuple[int, int, np.ndarray, np.ndarray]] = []

            sorted_sched = sorted(sched)
            next_call_after: dict[int, int] = {}
            ptr = 0
            for s in range(int(args.horizon)):
                while ptr < len(sorted_sched) and sorted_sched[ptr] <= s:
                    ptr += 1
                next_call_after[s] = sorted_sched[ptr] if ptr < len(sorted_sched) else int(args.horizon)

            for step in range(int(args.horizon)):
                offset = step % period
                is_call = step in sched_set
                is_check = (
                    step > 0
                    and not is_call
                    and last_plan_step // period != step // period
                    and offset <= 25
                    and offset % 5 == 0
                )
                if is_call and step > 0 and step <= int(args.horizon) - 50:
                    names, values, _ = _causal_plan_features(
                        observation=obs,
                        feature_builder=history,
                        subgoal=subgoal,
                        plan_age_steps=0,
                        time_scale=time_scale,
                    )
                    count_work("feature_vector_builds")
                    xv = _select_named_features(
                        tuple(names),
                        np.asarray(values, dtype=np.float64)[None, :],
                        STATE_ONLY_RENEWAL_NAMES,
                    )[0]
                    # Only clean uninterrupted 50-step windows or bootstrap later
                    renewal_records.append((step, xv))
                elif is_check and step <= int(args.horizon) - 50:
                    names, values, _ = _causal_plan_features(
                        observation=obs,
                        feature_builder=history,
                        subgoal=subgoal,
                        plan_age_steps=step - last_plan_step,
                        time_scale=time_scale,
                    )
                    count_work("feature_vector_builds")
                    arr = np.asarray(values, dtype=np.float64)[None, :]
                    xq = _select_named_features(tuple(names), arr, GENERIC_RAW_FEATURE_NAMES)[0]
                    xv_next = _select_named_features(tuple(names), arr, STATE_ONLY_RENEWAL_NAMES)[0]
                    continue_records.append((step, next_call_after[step], xq, xv_next))

                if is_call:
                    u_state = history.upper_state(obs, oracle_context=None)
                    out = controller.plan_goal(u_state, sample=False)
                    count_work("planner_calls")
                    subgoal = adapter.decode(
                        np.asarray(out["action"], dtype=np.float32), obs.achieved_goal
                    )
                    last_plan_step = step

                l_state = history.lower_state(obs, subgoal=subgoal)
                l_out = controller.act_conditioned(l_state, sample=False)
                count_work("lower_calls")
                req = squash_box_action(
                    np.asarray(l_out["action"], dtype=np.float32),
                    task.action_low,
                    task.action_high,
                )
                nxt, _, terminated, truncated, info = task.step(req)
                count_work("environment_steps")
                if (terminated or truncated) and step + 1 != int(args.horizon):
                    raise RuntimeError("Option-Critic episode terminated early")
                step_ises.append(float(info["tracking_distance"]) ** 2 * time_scale.dt_seconds)
                obs = nxt
                history.update(obs)
                count_work("history_updates")

            ises = np.asarray(step_ises, dtype=np.float64)
            # Store renewal 50-step costs where next call is >= 45 steps away (on-option segment)
            for s, xv in renewal_records:
                nxt_c = next_call_after[s]
                dur = min(50, nxt_c - s)
                if dur >= 25:
                    # Scale to 50-step rate if truncated between 25..49 steps
                    v_samples_x.append(xv)
                    v_samples_y.append(float(ises[s : s + dur].sum() * (50.0 / dur)))

            for s, nxt_c, xq, _ in continue_records:
                dur = min(50, nxt_c - s)
                if dur >= 10:
                    q_samples_x.append(xq)
                    q_samples_y.append(float(ises[s : s + dur].sum() * (50.0 / dur)))
        finally:
            task.environment.close()

    vx = np.vstack(v_samples_x)
    vy = np.asarray(v_samples_y, dtype=np.float64)
    qx = np.vstack(q_samples_x)
    qy = np.asarray(q_samples_y, dtype=np.float64)

    _, v_model = _train_regression_mlp(
        vx, vy, vx, seed=int(seed) + 501, hidden_dim=hidden_dim, epochs=200, weight_decay=1e-2
    )
    q_pred_train, q_model = _train_regression_mlp(
        qx, qy, qx, seed=int(seed) + 601, hidden_dim=hidden_dim, epochs=200, weight_decay=1e-2
    )

    # Evaluate predicted OC advantages on training checks to form deliberation threshold grid
    zv = (
        _select_named_features(GENERIC_RAW_FEATURE_NAMES, qx, STATE_ONLY_RENEWAL_NAMES)
        - np.asarray(v_model["feature_mean"])
    ) / np.asarray(v_model["feature_scale"])
    for layer in v_model["layers"][:-1]:
        zv = np.tanh(zv @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"]))
    v_last = v_model["layers"][-1]
    v_pred_on_q = (zv @ np.asarray(v_last["weight"]).T + np.asarray(v_last["bias"])).reshape(-1)
    adv_dist = q_pred_train - v_pred_on_q

    delib_cuts = [float(np.quantile(adv_dist, q)) for q in (0.35, 0.50, 0.65, 0.75, 0.85, 0.92)]
    base_model = {
        "q_input_names": list(GENERIC_RAW_FEATURE_NAMES),
        "v_input_names": list(STATE_ONLY_RENEWAL_NAMES),
        "q_mean": q_model["feature_mean"],
        "q_scale": q_model["feature_scale"],
        "q_layers": q_model["layers"],
        "v_mean": v_model["feature_mean"],
        "v_scale": v_model["feature_scale"],
        "v_layers": v_model["layers"],
    }
    candidates = [
        RouteBTimingRule(
            name=f"delib_oc_i{idx}",
            kind="oc_advantage",
            view="oc_q_minus_v",
            threshold=cut,
            model=base_model,
        )
        for idx, cut in enumerate(delib_cuts)
    ]

    sel_cache: dict[str, list[dict]] = {}

    def eval_sel(rule: TimingRule) -> list[dict]:
        if rule.name not in sel_cache:
            sel_cache[rule.name] = [
                run_episode(
                    controller,
                    task_spec=task_spec,
                    seed=s,
                    args=args,
                    time_scale=time_scale,
                    rule=rule,
                )[1]
                for s in args.timing_selection_seeds
            ]
        return sel_cache[rule.name]

    selected, table = select_rule(candidates, eval_sel)
    return selected, {
        "v_samples": len(vy),
        "q_samples": len(qy),
        "selected_name": selected.name,
        "selected_deliberation_margin": float(selected.threshold),
        "selection_table": table,
    }


class CEMSubgoalControllerWrapper:
    """Replaces fast MLP upper planner with a multi-rollout CEM subgoal trajectory optimizer."""

    def __init__(
        self,
        base_controller: Any,
        *,
        population_size: int = 64,
        elites: int = 12,
        cem_iters: int = 3,
        horizon_steps: int = 25,
        maximum_subgoal_delta: float = 0.75,
        dt_seconds: float = 0.01,
        seed: int = 20261118,
    ) -> None:
        self.base = base_controller
        self.config = base_controller.config
        self.population_size = int(population_size)
        self.elites = int(elites)
        self.cem_iters = int(cem_iters)
        self.horizon_steps = int(horizon_steps)
        self.max_delta = float(maximum_subgoal_delta)
        self.dt = float(dt_seconds)
        self._rng = np.random.default_rng(int(seed))

    def reset_recurrent_inference(self) -> None:
        self.base.reset_recurrent_inference()

    def act_conditioned(self, lower_state: np.ndarray, sample: bool = False) -> dict[str, Any]:
        return self.base.act_conditioned(lower_state, sample=sample)

    def plan_goal(self, upper_state: np.ndarray, sample: bool = False) -> dict[str, Any]:
        # Causal extraction from upper_state = [physical(4), target_error(2), history(64*6)]
        state = np.asarray(upper_state, dtype=np.float64).reshape(-1)
        phys_vel = state[2:4]
        target_err = state[4:6]
        hist = state[6:].reshape(-1, 6)
        target_now = hist[-1, :2]
        achieved_now = target_now - target_err
        vel_short = (hist[-1, :2] - hist[-11, :2]) / (10.0 * self.dt)

        # Warm-start anchor from the learned upper policy (raw action before tanh decode)
        warm = np.asarray(
            self.base.plan_goal(upper_state, sample=False)["action"], dtype=np.float64
        )
        mean = np.tanh(warm) * self.max_delta
        std = np.full(2, 0.30 * self.max_delta, dtype=np.float64)

        # Precompute causal constant-velocity damped target forecast over H_p steps
        steps = np.arange(1, self.horizon_steps + 1, dtype=np.float64)[:, None]
        decay = (0.96 ** steps)
        pred_targets = target_now[None, :] + vel_short[None, :] * (steps * self.dt) * decay

        for _ in range(self.cem_iters):
            noise = self._rng.standard_normal((self.population_size, 2))
            deltas = np.clip(mean[None, :] + noise * std[None, :], -self.max_delta, self.max_delta)
            deltas[0] = mean  # Always include current mean in candidate pool
            subgoals = achieved_now[None, :] + deltas  # (M, 2)

            # Vectorized 2nd-order damped PD closed-loop tracking surrogate over H_p steps
            pos = np.repeat(achieved_now[None, :], self.population_size, axis=0)
            vel = np.repeat(phys_vel[None, :] * 0.5, self.population_size, axis=0)
            costs = np.zeros(self.population_size, dtype=np.float64)
            kp, kd = 14.0, 4.5
            for k in range(self.horizon_steps):
                acc = np.clip(kp * (subgoals - pos) - kd * vel, -1.0, 1.0)
                vel = 0.92 * vel + acc * (self.dt * 8.0)
                pos = pos + vel * self.dt
                err = pos - pred_targets[k][None, :]
                costs += np.sum(err * err, axis=1) * self.dt + 0.002 * np.sum(acc * acc, axis=1) * self.dt
            # Mild trust-region anchor toward learned upper policy proposal
            costs += 0.08 * np.sum(np.square(deltas - np.tanh(warm)[None, :] * self.max_delta), axis=1)

            elite_idx = np.argsort(costs)[: self.elites]
            elites = deltas[elite_idx]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), 0.03 * self.max_delta)

        ratio = np.clip(mean / max(self.max_delta, 1e-6), -0.999, 0.999)
        raw_action = np.arctanh(ratio).astype(np.float32)
        return {"action": raw_action, "logp": 0.0, "value": 0.0}
