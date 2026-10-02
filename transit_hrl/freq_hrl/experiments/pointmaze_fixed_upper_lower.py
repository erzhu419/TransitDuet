"""Train one lower under an immutable final learned upper, not a moving one."""

import copy
import json

import numpy as np
import torch

from . import pointmaze_lower_budget as previous
from scripts import pointmaze_fixed_upper_lower_stage91_spec as spec

learning, swaps = previous.learning, previous.swaps


def check_training(training, root, method):
    protocol = spec.teacher_source if method == "joint_call" else spec.source
    if (training["status"],training["protocol"],training["root"],training["preflight"],training["contract"],training["seed_roles"]) != (
            "complete",protocol.EXPERIMENT_PROTOCOL,root,False,protocol.contract(),protocol.seed_roles(root,preflight=False)):
        raise ValueError("Stage91 requires registered full Stage88/90 training and exact training rosters")


def prepare_training(training, root, period, models, cost):
    model = models["lower_fixed_upper"]
    before = copy.deepcopy(model.state_dict())
    original = learning.native.joint.inference_weights(model)
    donors,checkpoints = {},{}
    for method in spec.CHECKPOINT_METHODS:
        path = spec.donor_result(root,method).parent/"final_weights"/f"period_{period}_{method}.pt"
        record = training[method]["groups"][str(period)]["trained"][method]
        if record["evaluation_update"] != 8 or record["final_freeze_check"] != "passed" or record["checkpoint"] != str(path):
            raise ValueError("Stage91 donor is not the registered final Stage88/90 checkpoint")
        donors[method] = swaps.check_checkpoint(torch.load(path,map_location="cpu",weights_only=False),original,
            root=root,period=period,method=method,protocol=previous.swap_spec if method == "joint_call" else spec)
        cost["checkpoint_loads"] += 1
        cost["checkpoint_freeze_checks"] += 1
        checkpoints[method] = str(path)
    model.upper_actor.load_state_dict(donors["joint_call"]["upper_actor"])
    learning.native.curves.support.assert_frozen(model,{**before,"upper_actor":donors["joint_call"]["upper_actor"]})
    cost["training_initialization_checks"] += 1
    metadata = {"reused_checkpoints":checkpoints,"checkpoint_freeze":"passed",
        "training_initialization":{"method":"lower_fixed_upper","upper_checkpoint":checkpoints["joint_call"],
            "source_lower_std_values_Adam_unchanged":"passed","fixed_final_upper_installed":"passed"}}
    return donors,metadata


def prepare_evaluation(donors, period, weights, cost):
    trained = {**donors,"lower_fixed_upper":weights["lower_fixed_upper"]}
    torch.testing.assert_close(trained["lower_fixed_upper"]["upper_actor"],donors["joint_call"]["upper_actor"],atol=0,rtol=0)
    composed = swaps.compose_weights(weights["base"],trained,protocol=spec)
    cost["actor_composition_checks"] += len(composed)
    return composed,{"actor_composition":"passed","final_learned_upper_frozen":"passed"}


def check_decomposition(period, effects):
    for total,a,b in (
            ("joint_upper_fixed_lower_minus_joint_upper_lower_matched","joint_upper_fixed_lower_minus_joint_call","joint_call_minus_joint_upper_lower_matched"),
            ("source_upper_fixed_lower_minus_lower_matched","source_upper_fixed_lower_minus_source_upper_joint_lower","source_upper_joint_lower_minus_lower_matched")):
        np.testing.assert_allclose(effects[f"{period}/{total}"],effects[f"{period}/{a}"]+effects[f"{period}/{b}"],atol=1e-12,rtol=0)


def qualify(cell, *, preflight):
    learning.qualify(cell,preflight=preflight,protocol=spec)
    o,h = spec.options(preflight=preflight),spec.arguments(cell["root"],preflight=preflight).horizon
    for p,g in cell["groups"].items():
        expected = {m:str(spec.donor_result(cell["root"],m).parent/"final_weights"/f"period_{p}_{m}.pt") for m in spec.CHECKPOINT_METHODS}
        initialization = {"method":"lower_fixed_upper","upper_checkpoint":expected["joint_call"],
            "source_lower_std_values_Adam_unchanged":"passed","fixed_final_upper_installed":"passed"}
        if (g["reused_checkpoints"] != expected or g["training_initialization"] != initialization
                or any(g[k] != "passed" for k in ("checkpoint_freeze","actor_composition","final_learned_upper_frozen"))):
            raise ValueError("Stage91 donor identity, training initialization or frozen learned upper changed")
        check_decomposition(p,g["effects"])
        checks = [previous.call_budget.check_update(row,method="lower_fixed_upper",period=int(p),horizon=h,preflight=preflight,protocol=spec)
            for row in g["trained"]["lower_fixed_upper"]["history"]]
        target = o["updates"]*spec.FISHER_RADIUS*spec.allocation("lower_fixed_upper",int(p))["lower"]
        if not np.isclose(sum(n for n,_ in checks),target,atol=1e-12,rtol=0):
            raise ValueError("Stage91 cumulative lower budget no longer matches Stage88 joint-call and Stage90 LM")
    return cell


def run(root, *, preflight, output):
    training = {m:json.loads(spec.donor_result(root,m).read_text()) for m in spec.CHECKPOINT_METHODS}
    for method,data in training.items():check_training(data,root,method)
    donors = {}

    def initialize(period,models,cost):
        donors[period],metadata = prepare_training(training,root,period,models,cost)
        return metadata

    return learning.run(root,preflight=preflight,output=output,protocol=spec,qualifier=qualify,initialize_models=initialize,
        evaluation_weights=lambda period,weights,cost:prepare_evaluation(donors[period],period,weights,cost))


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells,preflight=preflight,protocol=spec,qualifier=qualify)
    result["performance_claim"] = "fixed_final_learned_upper_conditional_lower_training_diagnostic_not_isolated_movement_effect_equal_total_compute_or_frequency_superiority"
    result["primary_endpoints"] = list(spec.PRIMARY_ENDPOINTS)
    return result
