"""Timing-only controls on a shared, frozen PointMaze upper/lower controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .pointmaze_budgeted_trigger import _predict_renewal_advantage
from .pointmaze_compact_plan_validity import (
    _CAUSAL_DYNAMIC_NAMES, _CURRENT_COMPACT_NAMES, _base_matrix,
    _quadratic_features, _select_named_features,
)
from .pointmaze_plan_validity_branching import _ridge_fit_predict


PROTOCOL_VERSION = "pointmaze_strong_timing_baselines_v1_prepared"
VIEWS = {"current": _CURRENT_COMPACT_NAMES, "history": _CAUSAL_DYNAMIC_NAMES}
THRESHOLD_FEATURES = (
    "plan_age_fraction", "tracking_distance",
    "target_velocity_change_norm", "force_current_norm",
)


def feature_view(rows: list[dict], view: str, *, quadratic: bool) -> tuple:
    schema, matrix = _base_matrix(rows)
    names = VIEWS[view]
    x = _select_named_features(schema, matrix, names)
    if quadratic:
        return _quadratic_features(x, names)
    # A termination action depends on the remaining opportunity to spend the call.
    return np.column_stack((x, [r["offset_fraction"] for r in rows],
                            [r["remaining_fraction"] for r in rows])), (
        *names, "offset_fraction", "remaining_fraction",
    )


@dataclass
class TimingRule:
    name: str
    kind: str
    view: str = "history"
    threshold: float = 0.0
    feature: str = "plan_age_fraction"
    model: dict = field(default_factory=dict)

    def score(self, names: tuple, values: np.ndarray, offset_fraction: float,
              remaining_fraction: float) -> float:
        if self.kind == "threshold":
            return float(values[names.index(self.feature)])
        if self.kind == "clock":
            return float(offset_fraction)
        if self.kind == "legacy":
            return _predict_renewal_advantage(
                feature_names=names, features=values, predictor=self.view,
                fitted=self.model,
            )
        row = {"feature_names": names, "causal_features": values,
               "offset_fraction": offset_fraction, "remaining_fraction": remaining_fraction}
        x, _ = feature_view([row], self.view, quadratic=self.kind == "ridge")
        z = (x[0] - self.model["feature_mean"]) / self.model["feature_scale"]
        if self.kind == "ridge":
            return float(self.model["weights"][0] + z @ self.model["weights"][1:])
        if self.kind == "termination":
            for layer in self.model["layers"][:-1]:
                z = np.tanh(z @ np.asarray(layer["weight"]).T + layer["bias"])
            last = self.model["layers"][-1]
            return float((z @ np.asarray(last["weight"]).T + last["bias"])[0])
        raise ValueError(f"unknown timing rule {self.kind}")

    def to_dict(self) -> dict:
        return asdict(self)


def threshold_candidates(fit_rows: list[dict]) -> list[TimingRule]:
    schema, x = _base_matrix(fit_rows)
    rules = [TimingRule(f"clock_{offset}", "clock", threshold=offset / 25)
             for offset in range(0, 26, 5)]
    for feature in THRESHOLD_FEATURES:
        values = x[:, schema.index(feature)]
        # Endpoints allow an honest fixed/deadline optimum, not a forced early call.
        cuts = np.unique(np.r_[np.nextafter(values.min(), -np.inf),
                              np.quantile(values, [0.25, 0.5, 0.75, 0.9]),
                              np.nextafter(values.max(), np.inf)])
        rules.extend(TimingRule(f"threshold_{feature}_{i}", "threshold",
                                threshold=float(cut), feature=feature)
                     for i, cut in enumerate(cuts))
    return rules


def _ridge_at_df(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray,
                 effective_df: float) -> tuple[np.ndarray, dict]:
    scale = x.std(axis=0)
    z = (x - x.mean(axis=0)) / np.where(scale > 1e-8, scale, 1.0)
    eigenvalues = np.linalg.svd(z, compute_uv=False) ** 2
    eigenvalues = eigenvalues[eigenvalues > max(1.0, eigenvalues[0]) * 1e-12]
    if not 0 < effective_df < len(eigenvalues):
        raise ValueError("requested ridge effective DF exceeds the training rank")
    lower, upper = 0.0, float(eigenvalues.sum() / effective_df)
    for _ in range(70):
        alpha = (lower + upper) / 2
        if np.sum(eigenvalues / (eigenvalues + alpha)) > effective_df:
            lower = alpha
        else:
            upper = alpha
    prediction, model = _ridge_fit_predict(x, y, eval_x, alpha=upper)
    model.update(effective_df_slopes=float(np.sum(eigenvalues / (eigenvalues + upper))),
                 coefficient_count=x.shape[1] + 1)
    return prediction, model


def fit_capacity_controls(fit_rows: list[dict], *, effective_df: float = 8.0,
                          threshold_quantile: float = 0.75,
                          views=tuple(VIEWS)) -> list[TimingRule]:
    """Same branch labels, 171 coefficients and fixed effective DF in both views."""
    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold quantile must be inside (0, 1)")
    y = np.asarray([r["renew_ise_advantage"] for r in fit_rows])
    groups = np.asarray([r["seed"] for r in fit_rows])
    if len(np.unique(groups)) < 3:
        raise ValueError("capacity controls require three fitting paths")
    rules = []
    for view in views:
        x, names = feature_view(fit_rows, view, quadratic=True)
        oof = np.empty(len(y))
        for group in np.unique(groups):
            train = groups != group
            oof[~train], _ = _ridge_at_df(x[train], y[train], x[~train], effective_df)
        _, model = _ridge_at_df(x, y, x, effective_df)
        model.update(feature_names=list(names), threshold_source="fit_path_oof",
                     fit_path_count=len(np.unique(groups)),
                     oof_mse=float(np.mean((oof - y) ** 2)))
        rules.append(TimingRule(f"matched_ridge_{view}", "ridge", view=view,
                                threshold=float(np.quantile(oof, threshold_quantile)),
                                model=model))
    return rules


def fit_termination(fit_rows: list[dict], *, view: str, seed: int,
                    epochs: int = 200, hidden_dim: int = 64) -> TimingRule:
    """Cost-sensitive policy improvement from full-episode paired timing credit."""
    import torch
    from freq_hrl.rl.dual_actor_critic import BernoulliActor
    from .compute_budget import count_work

    if epochs < 1 or hidden_dim < 1:
        raise ValueError("termination fit dimensions must be positive")
    x, names = feature_view(fit_rows, view, quadratic=False)
    advantages = np.asarray([r["timing_advantage"] for r in fit_rows])
    if not np.all(np.isfinite(advantages)) or not np.any(advantages):
        raise ValueError("termination fitting requires finite, nonzero timing credit")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.1)
    states = torch.as_tensor((x - mean) / scale, dtype=torch.float32)
    labels = torch.as_tensor(advantages > 0, dtype=torch.float32).view(-1, 1)
    weights = torch.as_tensor(np.abs(advantages) / np.mean(np.abs(advantages)),
                              dtype=torch.float32).view(-1, 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = BernoulliActor(x.shape[1], hidden_dim, init_logit=0.0)
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-3)
        for _ in range(epochs):
            loss = (weights * torch.nn.functional.binary_cross_entropy_with_logits(
                actor.net(states), labels, reduction="none")).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count_work("termination_optimizer_updates")
        layers = [{"weight": layer.weight.detach().numpy().tolist(),
                   "bias": layer.bias.detach().numpy().tolist()}
                  for layer in actor.net if isinstance(layer, torch.nn.Linear)]
        parameter_count = sum(p.numel() for p in actor.parameters())
    return TimingRule(f"termination_{view}", "termination", view=view, model={
        "feature_names": list(names), "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(), "layers": layers,
        "parameter_count": parameter_count, "epochs": epochs,
        "fit_loss": float(loss.detach()), "fit_rows": len(fit_rows),
        "target": "paired_full_episode_ise_wait_one_check_minus_stop_now",
    })


def select_rule(rules: list[TimingRule], evaluate: Any) -> tuple[TimingRule, list[dict]]:
    """Selection uses only a supplied selection-path evaluator, never test rows."""
    rows = []
    for index, rule in enumerate(rules):
        outcomes = evaluate(rule)
        rows.append({"index": index, "name": rule.name,
                     "mean_ise": float(np.mean([r["tracking_squared_error_integral"]
                                                for r in outcomes]))})
    chosen = min(rows, key=lambda row: (row["mean_ise"], row["index"]))
    return rules[chosen["index"]], rows


def termination_readiness(rows: list[dict], clock_rows: list[dict]) -> dict:
    """Development eligibility only; never filters roots or replaces the best checkpoint."""
    by_seed = {r["seed"]: r for r in clock_rows}
    offsets = [step % 50 for row in rows for step in row["decision_steps"][1:]]
    by_bin_offset: dict[tuple[int, int], set] = {}
    for row in rows:
        for query in row["queries"]:
            key = (query["step"] // 50, query["offset"])
            by_bin_offset.setdefault(key, set()).add(query["stop"])
    conditional = any(len(actions) == 2 for actions in by_bin_offset.values())
    improvement = float(np.mean([by_seed[r["seed"]]["tracking_squared_error_integral"]
                                - r["tracking_squared_error_integral"] for r in rows]))
    active = len(set(offsets)) > 1 and any(offset < 25 for offset in offsets)
    return {"behaviorally_noncollapsed": active,
            "state_conditional_at_same_bin_and_offset": conditional,
            "selection_ise_gain_over_best_clock": improvement,
            "eligible_for_fresh_confirmation": bool(active and conditional and improvement > 0),
            "performance_claim_supported": False}


def rollout_rule(controller: Any, rule: TimingRule, *, seed: int, args: Any,
                 time_scale: Any, intervention_step: int | None = None,
                 arm: str | None = None) -> dict:
    from .pointmaze_adaptive_pair_diagnostic import rollout_intervention

    if rule.kind == "jitter":
        if intervention_step is not None:
            raise ValueError("paired stopping credit uses adaptive checks, not jitter schedules")
        from .pointmaze_budgeted_trigger import rollout_budgeted_trigger
        from .pointmaze_plan_validity_branching import _task_options

        row = rollout_budgeted_trigger(controller, seed=seed, mode="balanced_jitter",
            fitted_predictors={}, env_id=args.env_id, horizon=args.horizon, time_scale=time_scale,
            maximum_subgoal_delta=args.maximum_subgoal_delta, max_offset_steps=args.max_offset_steps,
            check_stride_steps=args.check_stride_steps, task_options=_task_options(args))
        row["queries"] = []
        return row

    queries = []

    def score(names, values, offset_fraction, remaining_fraction):
        value = rule.score(tuple(names), np.asarray(values), offset_fraction, remaining_fraction)
        queries.append({"step": int(round(args.horizon * (1 - remaining_fraction))),
                        "offset": int(round(offset_fraction * args.max_offset_steps)),
                        "score": value, "stop": value >= rule.threshold})
        return value

    row = rollout_intervention(
        controller, seed=seed, intervention_step=intervention_step, arm=arm,
        predictor={"threshold": rule.threshold}, args=args, time_scale=time_scale,
        score_fn=score,
    )
    row.update(seed=int(seed), mode=rule.name, queries=queries,
               trigger_score_checks=len(queries),
               upper_decision_count=len(row["decision_steps"]),
               episode_length=args.horizon)
    return row


def collect_timing_pairs(controller: Any, rules: list[TimingRule], *, seeds: list[int],
                         args: Any, time_scale: Any, round_index: int,
                         rollout_fn=None) -> tuple[list, int]:
    """Acquire equal-budget action-aligned labels on a mixture of visited states."""
    rollout_fn = rollout_rule if rollout_fn is None else rollout_fn
    rows, replay_steps = [], 0
    period = time_scale.upper_period_steps
    # The first round's two heads share a clock; replay that identical behavior once.
    unique_rules = []
    for rule in rules:
        if not any(rule.to_dict() == other.to_dict() for other in unique_rules):
            unique_rules.append(rule)
    for rule_index, rule in enumerate(unique_rules):
        for seed in seeds:
            factual = rollout_fn(controller, rule, seed=seed, args=args, time_scale=time_scale)
            replay_steps += args.horizon
            eligible_bins = np.arange(1, args.horizon // period - 1)
            if args.pairs_per_path > len(eligible_bins):
                raise ValueError("timing-pair budget exceeds available bins")
            rng = np.random.default_rng(np.random.SeedSequence(
                [args.optimizer_seed, seed, round_index, rule_index, 305]))
            for bin_index in sorted(rng.choice(eligible_bins, args.pairs_per_path, replace=False)):
                last_offset = min(factual["decision_steps"][bin_index] % period,
                                  args.max_offset_steps - args.check_stride_steps)
                offset = int(rng.choice(np.arange(0, last_offset + 1, args.check_stride_steps)))
                step = int(bin_index * period + offset)
                arms = {arm: rollout_fn(controller, rule, seed=seed, args=args,
                                         time_scale=time_scale, intervention_step=step, arm=arm)
                        for arm in ("now", "wait_one_check")}
                replay_steps += 2 * args.horizon
                now, wait = arms["now"], arms["wait_one_check"]
                factual_arm = now if offset == factual["decision_steps"][bin_index] % period else wait
                if (not np.array_equal(now["prefix"], wait["prefix"])
                    or not np.array_equal(now["features"], wait["features"])
                    or factual_arm["decision_steps"] != factual["decision_steps"]
                    or abs(factual_arm["tracking_squared_error_integral"]
                           - factual["tracking_squared_error_integral"]) > 1e-8):
                    raise RuntimeError("timing credit lacks an identical prefix and factual replay")
                rows.append({
                    "seed": seed, "round": round_index, "behavior": rule.name,
                    "step": step, "offset_fraction": offset / args.max_offset_steps,
                    "remaining_fraction": (args.horizon - step) / args.horizon,
                    "feature_names": list(now["feature_names"]),
                    "causal_features": now["features"].tolist(),
                    "candidate_feature_has_future_access": False,
                    "candidate_feature_has_regime_label": False,
                    "timing_advantage": wait["tracking_squared_error_integral"]
                                        - now["tracking_squared_error_integral"],
                })
    return rows, replay_steps
