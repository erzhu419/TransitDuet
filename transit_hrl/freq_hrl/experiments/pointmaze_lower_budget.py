"""Decompose the lower deficit without repeating the Stage88 trained donors."""

import json

import numpy as np
import torch

from . import pointmaze_actor_swap as swaps
from . import pointmaze_call_weighted as call_budget
from scripts import pointmaze_lower_budget_stage90_spec as spec
from scripts import pointmaze_call_weighted_actor_swap_stage89_spec as swap_spec

learning = call_budget.learning


def check_training(training, root):
    if (training["status"],training["protocol"],training["root"],training["preflight"],training["contract"],training["seed_roles"]) != (
            "complete",spec.source.EXPERIMENT_PROTOCOL,root,False,spec.source.contract(),spec.source.seed_roles(root,preflight=False)):
        raise ValueError("Stage90 requires the registered full Stage88 training and exact training rosters")


def prepare_evaluation(training, root, period, weights, cost):
    original,trained = weights["base"],{"lower_matched":weights["lower_matched"]}
    checkpoints = {}
    for method in spec.CHECKPOINT_METHODS:
        path = spec.training_result(root).parent/"final_weights"/f"period_{period}_{method}.pt"
        record = training["groups"][str(period)]["trained"][method]
        if record["evaluation_update"] != 8 or record["final_freeze_check"] != "passed" or record["checkpoint"] != str(path):
            raise ValueError("Stage90 donor is not the registered final Stage88 checkpoint")
        trained[method] = swaps.check_checkpoint(torch.load(path,map_location="cpu",weights_only=False),original,
            root=root,period=period,method=method,protocol=swap_spec)
        cost["checkpoint_loads"] += 1
        cost["checkpoint_freeze_checks"] += 1
        checkpoints[method] = str(path)
    composed = swaps.compose_weights(original,trained,protocol=spec)
    cost["actor_composition_checks"] += len(composed)
    return composed,{"reused_checkpoints":checkpoints,"checkpoint_freeze":"passed","actor_composition":"passed"}


def check_decomposition(period, effects):
    for total,training,budget in (
            ("source_upper_joint_lower_minus_lower_full","source_upper_joint_lower_minus_lower_matched","lower_matched_minus_lower_full"),
            ("joint_call_minus_joint_upper_lower_only_lower","joint_call_minus_joint_upper_lower_matched","joint_upper_lower_matched_minus_joint_upper_lower_only_lower")):
        np.testing.assert_allclose(effects[f"{period}/{total}"],effects[f"{period}/{training}"]+effects[f"{period}/{budget}"],atol=1e-12,rtol=0)


def qualify(cell, *, preflight):
    learning.qualify(cell,preflight=preflight,protocol=spec)
    o,h = spec.options(preflight=preflight),spec.arguments(cell["root"],preflight=preflight).horizon
    for p,g in cell["groups"].items():
        expected = {m:str(spec.training_result(cell["root"]).parent/"final_weights"/f"period_{p}_{m}.pt") for m in spec.CHECKPOINT_METHODS}
        if g["reused_checkpoints"] != expected or any(g[k] != "passed" for k in ("checkpoint_freeze","actor_composition")):
            raise ValueError("Stage90 reused checkpoint identity or composition changed")
        check_decomposition(p,g["effects"])
        checks = [call_budget.check_update(row,method="lower_matched",period=int(p),horizon=h,preflight=preflight,protocol=spec)
            for row in g["trained"]["lower_matched"]["history"]]
        target = o["updates"]*spec.FISHER_RADIUS*spec.allocation("lower_matched",int(p))["lower"]
        if not np.isclose(sum(n for n,_ in checks),target,atol=1e-12,rtol=0):
            raise ValueError("Stage90 cumulative lower budget is not matched to joint-call")
    return cell


def run(root, *, preflight, output):
    training = json.loads(spec.training_result(root).read_text())
    check_training(training,root)
    return learning.run(root,preflight=preflight,output=output,protocol=spec,qualifier=qualify,
        evaluation_weights=lambda period,weights,cost:prepare_evaluation(training,root,period,weights,cost))


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells,preflight=preflight,protocol=spec,qualifier=qualify)
    result["performance_claim"] = "fixed_Stage88_lower_budget_and_training_effect_decomposition_not_new_algorithm_selection_or_frequency_superiority"
    return result
