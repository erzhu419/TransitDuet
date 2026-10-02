"""Condition a lower-only paired MC baseline on the same upper innovations."""

import copy
import json

import numpy as np
import torch

from . import pointmaze_fixed_upper_lower as previous
from . import pointmaze_lower_budget as budget_training
from scripts import pointmaze_upper_common_noise_stage92_spec as spec

learning,swaps = previous.learning,previous.swaps
scenario = learning.parts.scenario


def worker_pair(jobs):
    first = scenario.worker_native(jobs[0])
    first[1]["upper_noise_seed"] = first[1]["noise_seed"]
    weights,seed,noise_seed,variant,period,predictor,alpha,envelope,collect = jobs[1]
    _,args = learning.native._WORKER
    lower_seed = scenario.spec.noise_seeds(args.optimizer_seed,seed,noise_seed)[1]
    batch,row = learning.native.native_episode(weights,seed=seed,variant=variant,period=period,predictor=predictor,
        alpha=alpha,envelope=envelope,collect=collect,policy_seed=first[1]["policy_seed"],lower_seed=lower_seed,
        upper_standard_noise=first[1]["upper_standard_noise"])
    row.update(noise_seed=noise_seed,upper_noise_seed=first[1]["noise_seed"])
    return [first,(batch,row)]


def check_scenario_pair(pairs, roster, *, root):
    expected_upper = scenario.spec.noise_seeds(root,roster["scenario_seed"],roster["noise_seeds"][0])[0]
    expected_lower = [scenario.spec.noise_seeds(root,roster["scenario_seed"],n)[1] for n in roster["noise_seeds"]]
    rows = [r for _,r in pairs]
    if ([r["seed"] for r in rows] != [roster["scenario_seed"]]*2 or
            [r["noise_seed"] for r in rows] != roster["noise_seeds"] or
            [r["upper_noise_seed"] for r in rows] != [roster["noise_seeds"][0]]*2 or
            [r["policy_seed"] for r in rows] != [expected_upper]*2 or
            [r["lower_seed"] for r in rows] != expected_lower or len(set(expected_lower)) != 2 or
            [r["upper_replay_forward_calls"] for r in rows] != [0,len(rows[0]["upper_standard_noise"])]):
        raise ValueError("Stage92 requires common upper and independent registered lower noise within each pair")
    reference = pairs[0][0].lower.state
    for batch,row in pairs[1:]:
        np.testing.assert_array_equal(batch.lower.state[0,:4],reference[0,:4])
        np.testing.assert_array_equal(batch.lower.state[:,6:390],reference[:,6:390])
        np.testing.assert_allclose(row["upper_standard_noise"],rows[0]["upper_standard_noise"],atol=2e-6,rtol=0)


def check_training(training, root, method):
    protocol = {"joint_call":spec.source.teacher_source,"lower_matched":spec.source.source,
        "lower_fixed_upper":spec.source}[method]
    if (training["status"],training["protocol"],training["root"],training["preflight"],training["contract"],training["seed_roles"]) != (
            "complete",protocol.EXPERIMENT_PROTOCOL,root,False,protocol.contract(),protocol.seed_roles(root,preflight=False)):
        raise ValueError("Stage92 requires completed registered Stage88/90/91 final donors")


def prepare_training(training, root, period, models, cost):
    original = learning.native.joint.inference_weights(models["source_upper_common"])
    donors,checkpoints,covariance = {},{},{}
    for method in spec.CHECKPOINT_METHODS:
        path = spec.donor_result(root,method).parent/"final_weights"/f"period_{period}_{method}.pt"
        record = training[method]["groups"][str(period)]["trained"][method]
        if record["evaluation_update"] != 8 or record["final_freeze_check"] != "passed" or record["checkpoint"] != str(path):
            raise ValueError("Stage92 donor is not the registered last update")
        reference = {**original,"upper_actor":donors["joint_call"]["upper_actor"]} if method == "lower_fixed_upper" else original
        protocol = {"joint_call":budget_training.swap_spec,"lower_matched":spec.source,"lower_fixed_upper":spec}[method]
        donors[method] = swaps.check_checkpoint(torch.load(path,map_location="cpu",weights_only=False),reference,
            root=root,period=period,method=method,protocol=protocol)
        checkpoints[method] = str(path)
        cost["checkpoint_loads"] += 1
        cost["checkpoint_freeze_checks"] += 1
        if method != "joint_call":
            covariance[method] = record["history"][0]["actors"]["lower"]["scenario_covariance_trace"]
    for method,model in models.items():
        before = copy.deepcopy(model.state_dict())
        upper = original["upper_actor"] if method == "source_upper_common" else donors["joint_call"]["upper_actor"]
        model.upper_actor.load_state_dict(upper)
        learning.native.curves.support.assert_frozen(model,{**before,"upper_actor":upper})
        cost["training_initialization_checks"] += 1
    return donors,{"reused_checkpoints":checkpoints,"checkpoint_freeze":"passed","training_initialization":"passed",
        "baseline_initial_covariance":covariance,"training_noise_pairing":"common_upper_independent_lower"}


def prepare_evaluation(donors, period, weights, cost):
    torch.testing.assert_close(weights["source_upper_common"]["upper_actor"],weights["base"]["upper_actor"],atol=0,rtol=0)
    torch.testing.assert_close(weights["joint_upper_common"]["upper_actor"],donors["joint_call"]["upper_actor"],atol=0,rtol=0)
    composed = swaps.compose_weights(weights["base"],{**donors,**{m:weights[m] for m in spec.METHODS}},protocol=spec)
    cost["actor_composition_checks"] += len(composed)
    return composed,{"actor_composition":"passed","final_upper_freeze":"passed"}


def qualify(cell, *, preflight):
    learning.qualify(cell,preflight=preflight,protocol=spec)
    o,h = spec.options(preflight=preflight),spec.arguments(cell["root"],preflight=preflight).horizon
    for p,g in cell["groups"].items():
        expected = {m:str(spec.donor_result(cell["root"],m).parent/"final_weights"/f"period_{p}_{m}.pt") for m in spec.CHECKPOINT_METHODS}
        if (g["reused_checkpoints"] != expected or g["training_noise_pairing"] != "common_upper_independent_lower"
                or any(g[k] != "passed" for k in ("checkpoint_freeze","training_initialization","actor_composition","final_upper_freeze"))):
            raise ValueError("Stage92 checkpoint, fixed upper or noise contract changed")
        for method,t in g["trained"].items():
            checks = [budget_training.call_budget.check_update(row,method=method,period=int(p),horizon=h,preflight=preflight,protocol=spec)
                for row in t["history"]]
            target = o["updates"]*spec.FISHER_RADIUS*spec.allocation(method,int(p))["lower"]
            if not np.isclose(sum(n for n,_ in checks),target,atol=1e-12,rtol=0):raise ValueError("Stage92 lower budget changed")
        for variant,rows in g["evaluation"].items():
            for row in rows:
                policy,lower = scenario.spec.noise_seeds(cell["root"],row["seed"],row["seed"])
                if ((row["noise_seed"],row["policy_seed"],row["lower_seed"]) != (row["seed"],policy,lower)
                        or "upper_noise_seed" in row or row["upper_replay_forward_calls"] != 0):
                    raise ValueError("Stage92 common training noise leaked into evaluation")
    return cell


def run(root, *, preflight, output):
    training = {m:json.loads(spec.donor_result(root,m).read_text()) for m in spec.CHECKPOINT_METHODS}
    for method,data in training.items():check_training(data,root,method)
    donors = {}

    def initialize(period,models,cost):
        donors[period],metadata = prepare_training(training,root,period,models,cost)
        return metadata

    return learning.run(root,preflight=preflight,output=output,protocol=spec,qualifier=qualify,initialize_models=initialize,
        evaluation_weights=lambda p,w,c:prepare_evaluation(donors[p],p,w,c),training_pair_worker=worker_pair,
        scenario_pair_check=lambda pairs,roster:check_scenario_pair(pairs,roster,root=root))


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells,preflight=preflight,protocol=spec,qualifier=qualify)
    result["performance_claim"] = "fixed_upper_lower_credit_noise_conditioning_not_joint_training_or_frequency_superiority"
    result["primary_endpoints"] = list(spec.PRIMARY_ENDPOINTS)
    result["conditioning_confirmation"] = "mechanical_only" if preflight else (
        "supported" if all(result["endpoints"][k]["ci"][0]>0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported")
    return result
