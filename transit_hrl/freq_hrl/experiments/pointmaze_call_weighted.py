"""Qualify shared-sample learning under a decision-call-weighted KL budget."""

import numpy as np

from . import pointmaze_iterative_mc as learning
from scripts import pointmaze_call_weighted_stage87_spec as spec


def check_update(row, *, method, period, horizon, preflight):
    allocation = spec.allocation(method,period)
    o = spec.options(preflight=preflight)
    episodes = 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]
    nominal = spec.FISHER_RADIUS*sum(v/(period if a == "upper" else 1) for a,v in allocation.items())
    exact = sum(r["geometry"]["exact_kl"]["plus"]/(period if a == "upper" else 1) for a,r in row["actors"].items())
    if (set(row["actors"]) != set(allocation) or row["nominal_call_weighted_kl"] != nominal
            or row["exact_call_weighted_kl"] != exact or not .5*nominal <= exact <= 2*nominal):
        raise ValueError("Stage87 registered decision-call-weighted KL changed")
    for a,r in row["actors"].items():
        if (r["gradient_episodes"] != episodes or r["decision_calls_per_episode"] != (horizon//period if a == "upper" else horizon)
                or r["geometry"]["nominal_fisher_kl"] != spec.FISHER_RADIUS*allocation[a]):
            raise ValueError("Stage87 full shared gradient samples or per-level allocation changed")
    return nominal,exact


def qualify(cell, *, preflight):
    learning.qualify(cell,preflight=preflight,protocol=spec)
    h = spec.arguments(cell["root"],preflight=preflight).horizon
    for p,g in cell["groups"].items():
        for method,t in g["trained"].items():
            checks = [check_update(r,method=method,period=int(p),horizon=h,preflight=preflight) for r in t["history"]]
            if method != "joint_level" and not np.isclose(sum(n for n,_ in checks),spec.options(preflight=preflight)["updates"]*spec.FISHER_RADIUS,rtol=0,atol=1e-12):
                raise ValueError("Stage87 call-weighted cumulative nominal budget mismatch")
    return cell


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells,preflight=preflight,protocol=spec,qualifier=qualify)
    result["performance_claim"] = "shared_full_MC_mean_learning_call_weighted_nominal_KL_not_full_actor_critic_or_frequency_superiority"
    return result
