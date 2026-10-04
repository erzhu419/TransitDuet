"""Stage116: train the complete explicit Bernstein plan-coordinate readout."""

import copy

import numpy as np
import torch
from torch import nn

from . import pointmaze_upper_residual_train as base
from . import pointmaze_upper_wide_plan_train as wide
from . import pointmaze_upper_local_credit_train as local_credit
from scripts import pointmaze_upper_full_plan_train_stage116_spec as spec


class FullPlanCoordinateActor(nn.Module):
    """Train all eight explicit plan coordinates from an exact zero residual."""

    def __init__(self, donor):
        super().__init__()
        self.feedback_dim = int(donor.net[0].in_features)
        self.base = copy.deepcopy(donor).requires_grad_(False)
        self.donor_dim = int(donor.log_std.numel())
        self.readout = nn.Linear(self.feedback_dim, 2 * self.donor_dim)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def flat_input(self, state):
        return torch.cat((state[..., :self.feedback_dim],
                          torch.zeros_like(state[..., self.feedback_dim:])), -1)

    def distribution(self, state):
        donor = self.base.distribution(self.flat_input(state))
        donor_mean = torch.cat((donor.mean, torch.zeros_like(donor.mean)), -1)
        donor_std = torch.cat((donor.stddev, donor.stddev), -1)
        return torch.distributions.Normal(donor_mean + self.readout(state), donor_std)

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


def upper_branch(model):
    return FullPlanCoordinateActor(model.upper_actor)


def plan_for_arm(arm, predictor, period, calibration, args):
    return wide.plan_for_arm(arm, predictor, period, calibration, args)


def training_pair(job):
    source_weights, lower_state, upper_state, scenario_seed, noise_seeds, period, predictor, calibration, args = job
    outputs = [base.native_episode(
        source_weights, lower_state, upper_state, seed=scenario_seed, noise_seed=noise,
        arm="learned", period=period, predictor=predictor, calibration=calibration, args=args,
        collect=True, upper_sample=True, upper_factory=upper_branch, plan_factory=plan_for_arm,
    ) for noise in noise_seeds]
    np.testing.assert_array_equal(outputs[0][2]["decision_steps"], outputs[1][2]["decision_steps"])
    return {"lower_batches": [out[0] for out in outputs], "upper_batches": [out[1] for out in outputs],
        "rows": [out[2] for out in outputs], "innovations": [out[3] for out in outputs], "pairing": "passed"}


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
    for innovation in innovations.values():
        np.testing.assert_allclose(innovation, innovations["forecast"], atol=3e-5, rtol=0)
    return {"seed": seed, "evaluation": rows, "pairing": "passed"}


def qualify(cell, *, preflight):
    root = cell["root"]
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL
            or cell["contract"] != spec.contract() or root not in spec.roots(preflight=preflight)
            or cell["preflight"] != preflight or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell.get(key, 0) for key in ("optimizer_steps", "critic_fits", "native_trace_writes"))):
        raise ValueError("Stage116 protocol, source, roster, budget or frozen path changed")
    horizon = spec.arguments(root, preflight=preflight).horizon
    for period, group in cell["groups"].items():
        if group["source_and_lower_unchanged"] != "passed":
            raise ValueError("Stage116 source or lower branch freeze failed")
        effects = base.paired_effects(int(period), group["evaluation"],
            cell["seed_roles"]["native_evaluation"], protocol=spec)
        if group["effects"] != effects or not np.isfinite(list(effects.values())).all():
            raise ValueError("Stage116 paired evaluation changed")
        if set(group["evaluation"]) != set(spec.ARMS):
            raise ValueError("Stage116 evaluation roster changed")
        for variant, rows in group["evaluation"].items():
            for row in rows:
                base.check_row(row, period=int(period), horizon=horizon, variant=variant)
                if row["seed"] not in cell["seed_roles"]["native_evaluation"]:
                    raise ValueError("Stage116 evaluation seed changed")
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
        performance_claim="complete_explicit_bernstein_plan_coordinate_local_option_credit_gain",
        lower_source="Stage112 learned lower branch frozen",
    )
    return result

