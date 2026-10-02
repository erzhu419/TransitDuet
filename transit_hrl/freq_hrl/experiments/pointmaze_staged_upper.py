"""Train an upper mean with each registered Stage93 source-upper lower frozen."""

import copy
import json

import numpy as np
import torch

from . import pointmaze_upper_noise_replication as previous
from scripts import pointmaze_staged_upper_stage94_spec as spec

learning, scenario, swaps = previous.learning, previous.scenario, previous.swaps
call_budget = previous.common.budget_training.call_budget


def check_training(training, root, method, *, protocol=spec):
    spec = protocol
    if method == "joint_call":
        previous.common.check_training(training,root,method)
    elif (training["status"],training["protocol"],training["root"],training["preflight"],training["contract"],
            training["seed_roles"],training["cost"]) != ("complete",spec.source.EXPERIMENT_PROTOCOL,root,False,
                spec.source.contract(),spec.source.seed_roles(root,preflight=False),spec.source.budget(preflight=False)):
        raise ValueError("Staged learning requires completed full Stage93 lower training without donor selection")


def check_scenario_pair(pairs, roster, *, root):
    scenario.check_scenario_pair(pairs,roster)
    for _,row in pairs:
        expected = scenario.spec.noise_seeds(root,row["seed"],row["noise_seed"])
        if ((row["policy_seed"],row["lower_seed"]) != expected or "upper_noise_seed" in row
                or row["upper_replay_forward_calls"] != 0):
            raise ValueError("Upper credit requires independent original noise, not shared-upper replay")


def prepare_training(training, root, period, models, cost, *, protocol=spec):
    spec = protocol
    original = learning.native.joint.inference_weights(models["staged_independent"])
    donors,checkpoints = {},{}
    for method in spec.CHECKPOINT_METHODS:
        path = spec.donor_result(root,method).parent/"final_weights"/f"period_{period}_{method}.pt"
        record = training[method]["groups"][str(period)]["trained"][method]
        if record["evaluation_update"] != 8 or record["final_freeze_check"] != "passed" or record["checkpoint"] != str(path):
            raise ValueError("Staged-learning donor is not the registered final update")
        checkpoint_protocol = previous.common.budget_training.swap_spec if method == "joint_call" else spec
        donors[method] = swaps.check_checkpoint(torch.load(path,map_location="cpu",weights_only=False),original,
            root=root,period=period,method=method,protocol=checkpoint_protocol)
        checkpoints[method] = str(path)
        cost["checkpoint_loads"] += 1
        cost["checkpoint_freeze_checks"] += 1
    for method,model in models.items():
        before = copy.deepcopy(model.state_dict())
        lower = donors[spec.LOWER_FOR_METHOD[method]]["lower_actor"]
        model.lower_actor.load_state_dict(lower)
        learning.native.curves.support.assert_frozen(model,{**before,"lower_actor":lower})
        cost["training_initialization_checks"] += 1
    return donors,{"reused_checkpoints":checkpoints,"checkpoint_freeze":"passed", "training_initialization":"passed",
        "lower_for_method":spec.LOWER_FOR_METHOD, "training_noise_pairing":"independent_upper_and_lower"}


def prepare_evaluation(donors, period, weights, cost, *, protocol=spec):
    spec = protocol
    for method,lower in spec.LOWER_FOR_METHOD.items():
        torch.testing.assert_close(weights[method]["lower_actor"],donors[lower]["lower_actor"],atol=0,rtol=0)
    composed = swaps.compose_weights(weights["base"],{**donors,**{m:weights[m] for m in spec.METHODS}},protocol=spec)
    cost["actor_composition_checks"] += len(composed)
    return composed,{"actor_composition":"passed","final_lower_freeze":"passed"}


def qualify(cell, *, preflight, protocol=spec):
    spec = protocol
    learning.qualify(cell,preflight=preflight,protocol=spec)
    o,h = spec.options(preflight=preflight),spec.arguments(cell["root"],preflight=preflight).horizon
    for p,g in cell["groups"].items():
        expected = {m:str(spec.donor_result(cell["root"],m).parent/"final_weights"/f"period_{p}_{m}.pt") for m in spec.CHECKPOINT_METHODS}
        if (g["reused_checkpoints"] != expected or g["lower_for_method"] != spec.LOWER_FOR_METHOD
                or g["training_noise_pairing"] != "independent_upper_and_lower"
                or any(g[k] != "passed" for k in ("checkpoint_freeze","training_initialization","actor_composition","final_lower_freeze"))):
            raise ValueError("Staged-learning donor, frozen lower or independent-noise contract changed")
        for method,t in g["trained"].items():
            checks = [call_budget.check_update(row,method=method,period=int(p),horizon=h,preflight=preflight,protocol=spec)
                for row in t["history"]]
            target = o["updates"]*spec.FISHER_RADIUS*spec.allocation(method,int(p))["upper"]/int(p)
            if not np.isclose(sum(n for n,_ in checks),target,atol=1e-12,rtol=0):raise ValueError("Staged-learning upper budget changed")
        for rows in g["evaluation"].values():
            for row in rows:
                expected = scenario.spec.noise_seeds(cell["root"],row["seed"],row["seed"])
                if ((row["noise_seed"],row["policy_seed"],row["lower_seed"]) != (row["seed"],*expected)
                        or "upper_noise_seed" in row or row["upper_replay_forward_calls"] != 0):
                    raise ValueError("Staged evaluation changed the original paired noise path")
    return cell


def run(root, *, preflight, output, protocol=spec):
    spec = protocol
    records = {path:json.loads(path.read_text()) for path in {spec.donor_result(root,m) for m in spec.CHECKPOINT_METHODS}}
    training = {m:records[spec.donor_result(root,m)] for m in spec.CHECKPOINT_METHODS}
    for method,data in training.items():check_training(data,root,method,protocol=spec)
    donors = {}

    def initialize(period,models,cost):
        donors[period],metadata = prepare_training(training,root,period,models,cost,protocol=spec)
        return metadata

    return learning.run(root,preflight=preflight,output=output,protocol=spec,
        qualifier=lambda c,**kw:qualify(c,protocol=spec,**kw),initialize_models=initialize,
        evaluation_weights=lambda p,w,c:prepare_evaluation(donors[p],p,w,c,protocol=spec),
        scenario_pair_check=lambda pairs,roster:check_scenario_pair(pairs,roster,root=root))


def aggregate(cells, *, preflight, protocol=spec):
    spec = protocol
    result = learning.aggregate(cells,preflight=preflight,protocol=spec,
        qualifier=lambda c,**kw:qualify(c,protocol=spec,**kw))
    result["performance_claim"] = "staged_upper_MC_mean_learning_given_frozen_Stage93_lowers_not_full_actor_critic_or_frequency_superiority"
    result["primary_endpoints"] = list(spec.PRIMARY_ENDPOINTS)
    result["staged_confirmation"] = "mechanical_only" if preflight else (
        "supported" if all(result["endpoints"][k]["ci"][0]>0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported")
    return result
