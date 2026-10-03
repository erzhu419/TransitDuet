"""Replicate conditional lower learning with fresh trained controls."""

import copy
import json

import numpy as np
import torch

from . import pointmaze_upper_common_noise as common
from scripts import pointmaze_upper_noise_replication_stage93_spec as spec

learning, scenario, swaps = common.learning, common.scenario, common.swaps


def worker_pair(jobs, *, protocol=spec):
    spec = protocol
    if spec.NOISE_MODES[jobs[0][3]] == "common_upper_independent_lower":return common.worker_pair(jobs)
    return [scenario.worker_native(job) for job in jobs]


def check_scenario_pair(pairs, roster, *, root, protocol=spec):
    spec = protocol
    rows = [r for _,r in pairs]
    method = rows[0]["variant"]
    if any(r["variant"] != method for r in rows):raise ValueError("Stage93 pair mixed training methods")
    if spec.NOISE_MODES[method] == "common_upper_independent_lower":
        common.check_scenario_pair(pairs,roster,root=root)
    else:
        scenario.check_scenario_pair(pairs,roster)
        for row in rows:
            expected = scenario.spec.noise_seeds(root,row["seed"],row["noise_seed"])
            if ((row["policy_seed"],row["lower_seed"]) != expected or "upper_noise_seed" in row
                    or row["upper_replay_forward_calls"] != 0):
                raise ValueError("Stage93 independent control replayed upper noise")


def prepare_training(training, root, period, models, cost, *, protocol=spec,
        checkpoint_protocol=common.budget_training.swap_spec):
    spec = protocol
    original = learning.native.joint.inference_weights(models["source_upper_independent"])
    path = spec.donor_result(root,"joint_call").parent/"final_weights"/f"period_{period}_joint_call.pt"
    record = training["groups"][str(period)]["trained"]["joint_call"]
    if record["evaluation_update"] != 8 or record["final_freeze_check"] != "passed" or record["checkpoint"] != str(path):
        raise ValueError(f"{spec.EXPERIMENT_PROTOCOL} requires the registered final joint upper")
    donor = swaps.check_checkpoint(torch.load(path,map_location="cpu",weights_only=False),original,
        root=root,period=period,method="joint_call",protocol=checkpoint_protocol)
    cost["checkpoint_loads"] += 1
    cost["checkpoint_freeze_checks"] += 1
    for method,model in models.items():
        before = copy.deepcopy(model.state_dict())
        upper = donor["upper_actor"] if method in ("joint_upper_independent","joint_upper_common") else original["upper_actor"]
        model.upper_actor.load_state_dict(upper)
        learning.native.curves.support.assert_frozen(model,{**before,"upper_actor":upper})
        cost["training_initialization_checks"] += 1
    return {"joint_call":donor},{"reused_checkpoints":{"joint_call":str(path)},"checkpoint_freeze":"passed",
        "training_initialization":"passed","training_noise_pairing":spec.NOISE_MODES}


def prepare_evaluation(donors, period, weights, cost, *, protocol=spec):
    spec = protocol
    for method in spec.METHODS:
        upper = donors["joint_call"]["upper_actor"] if method in ("joint_upper_independent","joint_upper_common") else weights["base"]["upper_actor"]
        torch.testing.assert_close(weights[method]["upper_actor"],upper,atol=0,rtol=0)
    composed = swaps.compose_weights(weights["base"],{**donors,**{m:weights[m] for m in spec.METHODS}},protocol=spec)
    cost["actor_composition_checks"] += len(composed)
    return composed,{"actor_composition":"passed","final_upper_freeze":"passed"}


def qualify(cell, *, preflight, protocol=spec):
    spec = protocol
    learning.qualify(cell,preflight=preflight,protocol=spec)
    o,h = spec.options(preflight=preflight),spec.arguments(cell["root"],preflight=preflight).horizon
    for p,g in cell["groups"].items():
        expected = {"joint_call":str(spec.donor_result(cell["root"],"joint_call").parent/"final_weights"/f"period_{p}_joint_call.pt")}
        if (g["reused_checkpoints"] != expected or g["training_noise_pairing"] != spec.NOISE_MODES
                or any(g[k] != "passed" for k in ("checkpoint_freeze","training_initialization","actor_composition","final_upper_freeze"))):
            raise ValueError("Stage93 donor, fixed-upper or paired-noise contract changed")
        for method,t in g["trained"].items():
            checks = [common.budget_training.call_budget.check_update(row,method=method,period=int(p),horizon=h,preflight=preflight,protocol=spec)
                for row in t["history"]]
            target = o["updates"]*spec.FISHER_RADIUS*spec.allocation(method,int(p))["lower"]
            if not np.isclose(sum(n for n,_ in checks),target,atol=1e-12,rtol=0):raise ValueError("Stage93 lower budget changed")
        for rows in g["evaluation"].values():
            for row in rows:
                expected = scenario.spec.noise_seeds(cell["root"],row["seed"],row["seed"])
                if ((row["noise_seed"],row["policy_seed"],row["lower_seed"]) != (row["seed"],*expected)
                        or "upper_noise_seed" in row or row["upper_replay_forward_calls"] != 0):
                    raise ValueError("Stage93 paired training noise leaked into evaluation")
    return cell


def run(root, *, preflight, output):
    training = json.loads(spec.donor_result(root,"joint_call").read_text())
    common.check_training(training,root,"joint_call")
    donors = {}

    def initialize(period,models,cost):
        donors[period],metadata = prepare_training(training,root,period,models,cost)
        return metadata

    return learning.run(root,preflight=preflight,output=output,protocol=spec,qualifier=qualify,initialize_models=initialize,
        evaluation_weights=lambda p,w,c:prepare_evaluation(donors[p],p,w,c),training_pair_worker=worker_pair,
        scenario_pair_check=lambda pairs,roster:check_scenario_pair(pairs,roster,root=root))


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells,preflight=preflight,protocol=spec,qualifier=qualify)
    result["performance_claim"] = "fresh_sample_fixed_upper_lower_credit_replication_with_retrained_controls_not_joint_HRL_or_frequency_superiority"
    result["primary_endpoints"] = list(spec.PRIMARY_ENDPOINTS)
    result["conditioning_confirmation"] = "mechanical_only" if preflight else (
        "supported" if all(result["endpoints"][k]["ci"][0]>0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported")
    return result
