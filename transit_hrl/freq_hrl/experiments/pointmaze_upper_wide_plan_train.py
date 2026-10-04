"""Stage115: widen the frozen-upper residual into an explicit Bernstein plan head."""

import copy

import numpy as np
import torch
from torch import nn

from freq_hrl.policies import BernsteinPlanCurve
from freq_hrl.rl.plan_actions import LearnedPlanActionMapper
from . import pointmaze_upper_residual_train as base
from . import pointmaze_upper_local_credit_train as local_credit
from scripts import pointmaze_upper_wide_plan_train_stage115_spec as spec


class WidePlanResidualActor(nn.Module):
    """Keep the donor four coordinates and add four zero-initialized coordinates."""

    def __init__(self, donor):
        super().__init__()
        self.feedback_dim = int(donor.net[0].in_features)
        self.base = copy.deepcopy(donor).requires_grad_(False)
        self.extra_dim = int(donor.log_std.numel())
        self.readout = nn.Linear(self.feedback_dim, self.extra_dim)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def flat_input(self, state):
        return torch.cat((state[..., :self.feedback_dim],
                          torch.zeros_like(state[..., self.feedback_dim:])), -1)

    def distribution(self, state):
        donor = self.base.distribution(self.flat_input(state))
        extra_mean = self.readout(state)
        return torch.distributions.Normal(
            torch.cat((donor.mean, extra_mean), -1),
            torch.cat((donor.stddev, donor.stddev), -1),
        )

    def log_prob_entropy(self, state, action):
        distribution = self.distribution(state)
        return distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1)

    def forward_with_mean(self, state, sample=True):
        distribution = self.distribution(state)
        action = distribution.rsample() if sample else distribution.mean
        return action, distribution.log_prob(action).sum(-1), distribution.mean

    def forward(self, state, sample=True):
        action, logp, _ = self.forward_with_mean(state, sample=sample)
        return action, logp


class WideBernsteinPlan(base.source.native.curves.CalibratedPlan):
    """Use a five-coefficient anchored curve while retaining the native decoder."""

    def __init__(self, predictor, period, scale, envelope):
        super().__init__(predictor, period, scale, 1.0, envelope)
        self.mapper = LearnedPlanActionMapper(
            BernsteinPlanCurve(
                horizon_s=period * base.source.baseline.forecast.spec.DT_SECONDS,
                basis_dim=spec.PLAN_BASIS,
                n_entities=2,
            ),
            coefficient_scale=scale,
            anchor_first_coefficient=True,
        )
        self.basis = np.asarray([
            self.mapper.curve.basis(t * base.source.baseline.forecast.spec.DT_SECONDS)
            for t in range(period + 1)
        ])

    @staticmethod
    def _mapper_action(action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 8:
            raise ValueError(f"expected wide plan action dim 8, got {action.size}")
        return np.concatenate((action[:2], action[4:6], action[2:4], action[6:8]))

    def plan_coefficients(self, action):
        return self.mapper.residual_coefficients(self._mapper_action(action))

    def decode(self, *, action, observation, history, step, world_low, world_high):
        self.proposed_actions.append(np.asarray(action, dtype=np.float64).copy())
        self.bounds = np.asarray(world_low), np.asarray(world_high)
        targets = history.history.reshape(-1, 6)[-min(64, step + 1):, :2]
        base_points = base.source.baseline.forecast.plan_points(
            targets, "ridge_velocity", self.predictor, self.period, self.bounds)
        self.base_points = base_points
        coefficients = self.plan_coefficients(action)
        self.points = np.clip(
            base_points.astype(np.float64)
            + self.basis @ coefficients.reshape(2, spec.PLAN_BASIS).T,
            *self.bounds,
        ).astype(np.float32)
        self.reference_points = self.velocity_points = self.points
        self.executed_delta_squared_sum += float(
            np.square(self.points.astype(np.float64) - base_points).sum()
        )
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.coefficients.append(coefficients)
        self.ols_fits += int(step > 0)
        self.ridge_predictions += int(step > 0)
        return self.points[0].copy()


def upper_branch(model):
    return WidePlanResidualActor(model.upper_actor)


def plan_for_arm(arm, predictor, period, calibration, args):
    if arm == "forecast":
        return base.source.baseline.forecast.PlanReference("ridge_velocity", predictor, period)
    if arm == "learned":
        return WideBernsteinPlan(predictor, period, args.maximum_subgoal_delta, calibration["envelope"])
    raise ValueError("unknown wide-plan arm")


def training_pair(job):
    source_weights, lower_state, upper_state, scenario_seed, noise_seeds, period, predictor, calibration, args = job
    outputs = [base.native_episode(
        source_weights, lower_state, upper_state, seed=scenario_seed, noise_seed=noise,
        arm="learned", period=period, predictor=predictor, calibration=calibration, args=args,
        collect=True, upper_sample=True, upper_factory=upper_branch, plan_factory=plan_for_arm,
    ) for noise in noise_seeds]
    np.testing.assert_array_equal(outputs[0][2]["decision_steps"], outputs[1][2]["decision_steps"])
    return {
        "lower_batches": [out[0] for out in outputs],
        "upper_batches": [out[1] for out in outputs],
        "rows": [out[2] for out in outputs],
        "innovations": [out[3] for out in outputs],
        "pairing": "passed",
    }


def evaluation_group(job):
    source_weights, lower_state, upper_state, seed, period, predictor, calibration, args = job
    rows, innovations = {}, {}
    variants = {"forecast": ("forecast", None), "learned": ("learned", upper_state),
        "learned_blinded": ("forecast", upper_state)}
    for variant, (arm, state) in variants.items():
        _, _, row, innovation = base.native_episode(
            source_weights, lower_state, state, seed=seed, noise_seed=seed, arm=arm,
            period=period, predictor=predictor, calibration=calibration, args=args,
            collect=False, upper_sample=False, upper_factory=upper_branch, plan_factory=plan_for_arm,
        )
        row.update(variant=variant, state_arm=arm)
        rows[variant], innovations[variant] = row, innovation
    reference = innovations["forecast"]
    for innovation in innovations.values():
        np.testing.assert_allclose(innovation, reference, atol=3e-5, rtol=0)
    return {"seed": seed, "evaluation": rows, "pairing": "passed"}


def qualify(cell, *, preflight):
    root = cell["root"]
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL
            or cell["contract"] != spec.contract() or root not in spec.roots(preflight=preflight)
            or cell["preflight"] != preflight or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell.get(key, 0) for key in ("optimizer_steps", "critic_fits", "native_trace_writes"))):
        raise ValueError("Stage115 protocol, source, roster, budget or frozen path changed")
    horizon = spec.arguments(root, preflight=preflight).horizon
    for period, group in cell["groups"].items():
        if group["source_and_lower_unchanged"] != "passed":
            raise ValueError("Stage115 source or lower branch freeze failed")
        effects = base.paired_effects(int(period), group["evaluation"],
            cell["seed_roles"]["native_evaluation"], protocol=spec)
        if group["effects"] != effects or not np.isfinite(list(effects.values())).all():
            raise ValueError("Stage115 paired evaluation changed")
        if set(group["evaluation"]) != set(spec.ARMS):
            raise ValueError("Stage115 evaluation roster changed")
        for variant, rows in group["evaluation"].items():
            for row in rows:
                base.check_row(row, period=int(period), horizon=horizon, variant=variant)
                if row["seed"] not in cell["seed_roles"]["native_evaluation"]:
                    raise ValueError("Stage115 evaluation seed changed")
    return cell


def aggregate(cells, *, preflight):
    statistics = __import__("freq_hrl.experiments.pointmaze_feasible_credit", fromlist=["aggregate"])
    result = statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    passed = {} if preflight else {str(period): all(
        result["endpoints"][f"{period}/{a}_minus_{b}"]["ci"][0] > 0
        for a, b in spec.CONTRASTS) for period in spec.PERIODS}
    result.update(
        upper_gain_gate="mechanical_only" if preflight else
        "supported_both_periods" if all(passed.values()) else
        "partial" if any(passed.values()) else "not_supported",
        period_upper_gain_gate=passed,
        performance_claim="wider_Bernstein_plan_coordinate_local_option_credit_gain",
        lower_source="Stage112 learned lower branch frozen",
    )
    return result
