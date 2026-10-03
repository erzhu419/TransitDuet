"""Native causal actor swaps using the registered last MC-trained weights."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from . import pointmaze_actor_parts as parts
from .pointmaze_root_response import write_json
from scripts import pointmaze_actor_swap_stage84_spec as spec

native = parts.native


def check_checkpoint(payload, source_weights, *, root, period, method, protocol=spec, training_protocol=None):
    spec = protocol
    trained = spec.source if training_protocol is None else training_protocol
    expected = {"protocol":trained.EXPERIMENT_PROTOCOL,"root":root,"period":period,"method":method,"updates":8}
    if {k:v for k,v in payload.items() if k != "weights"} != expected:
        raise ValueError(f"{spec.EXPERIMENT_PROTOCOL} requires the registered final checkpoint")
    weights = payload["weights"]
    if set(weights) != set(source_weights):raise ValueError("Stage84 checkpoint networks changed")
    for name, original in source_weights.items():
        current = weights[name]
        if set(current) != set(original) or any(current[k].shape != original[k].shape for k in original):
            raise ValueError("Stage84 checkpoint parameter schema changed")
        active = name.endswith("_actor") and name[:-6] in trained.METHODS[method]
        if active:
            torch.testing.assert_close(current["log_std"],original["log_std"],atol=0,rtol=0)
            if all(torch.equal(current[k],original[k]) for k in original):
                raise ValueError("Stage84 checkpoint has no trained actor mean")
        else:
            torch.testing.assert_close(current,original,atol=0,rtol=0)
    return weights


def compose_weights(source_weights, trained, *, protocol=spec):
    spec = protocol
    donors = {"source":source_weights,**trained}
    composed = {}
    for variant,(upper,lower) in spec.COMPOSITIONS.items():
        weights = copy.deepcopy(source_weights)
        for name,donor in (("upper_actor",upper),("lower_actor",lower)):
            weights[name] = copy.deepcopy(donors[donor][name])
        expected = {**source_weights,"upper_actor":donors[upper]["upper_actor"],"lower_actor":donors[lower]["lower_actor"]}
        torch.testing.assert_close(weights,expected,atol=0,rtol=0)
        composed[variant] = weights
    return composed


def paired_effects(period, evaluation, seeds, *, protocol=spec):
    spec = protocol
    result = native.paired_effects(period,evaluation,seeds,protocol=spec)
    variants = (*spec.CONTRAST_PAIRS[0],*spec.CONTRAST_PAIRS[1])
    result[f"{period}/upper_by_lower_interaction"] = float(np.mean([
        a["episode_return"]-b["episode_return"]-c["episode_return"]+d["episode_return"]
        for a,b,c,d in zip(*(evaluation[v] for v in variants))]))
    return result


def run(root, *, preflight, output, protocol=spec):
    spec = protocol
    decoder = json.loads(spec.source_result(root).read_text())
    decoder_spec = parts.scenario.spec.source.source
    if (decoder["status"],decoder["protocol"],decoder["root"],decoder["preflight"],decoder["contract"]) != (
            "complete",decoder_spec.EXPERIMENT_PROTOCOL,root,False,decoder_spec.contract()):
        raise ValueError("Stage84 requires the frozen full Stage78 decoder")
    training = json.loads(spec.training_result(root).read_text())
    if (training["status"],training["protocol"],training["root"],training["preflight"],training["contract"]) != (
            "complete",spec.source.EXPERIMENT_PROTOCOL,root,False,spec.source.contract()):
        raise ValueError(f"{spec.EXPERIMENT_PROTOCOL} requires completed full {spec.source.EXPERIMENT_PROTOCOL} training")
    clones,predictor,initialization = native.curves.support.native.load_source(root,preflight=False)
    args,roles,o = spec.arguments(root,preflight=preflight),spec.seed_roles(root,preflight=preflight),spec.options(preflight=preflight)
    cost,groups,started = dict.fromkeys(spec.budget(preflight=preflight),0),{},time.monotonic()
    cost.update(source_clone_loads=len(clones),forecaster_loads=1,decoder_loads=len(clones))
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS,0)
    with ProcessPoolExecutor(max_workers=o["workers"],mp_context=mp.get_context("spawn"),initializer=native.init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config,args)) as pool:
        for period in spec.PERIODS:
            original = clones[str(period)]
            snapshot = copy.deepcopy(original.state_dict())
            source_weights = native.joint.inference_weights(original)
            trained,checkpoints = {},{}
            for method in spec.CHECKPOINT_METHODS:
                path = spec.training_result(root).parent/"final_weights"/f"period_{period}_{method}.pt"
                record = training["groups"][str(period)]["trained"][method]
                if record["evaluation_update"] != 8 or record["final_freeze_check"] != "passed" or record["checkpoint"] != str(path):
                    raise ValueError("Stage84 final checkpoint record changed")
                trained[method] = check_checkpoint(torch.load(path,map_location="cpu",weights_only=False),
                    source_weights,root=root,period=period,method=method,protocol=spec)
                cost["checkpoint_loads"] += 1
                cost["checkpoint_freeze_checks"] += 1
                checkpoints[method] = str(path)
            weights = compose_weights(source_weights,trained,protocol=spec)
            cost["actor_composition_checks"] += len(weights)
            calibration = decoder["groups"][str(period)]["calibration"]
            native.bounded.check_calibration(calibration)
            alpha,envelope = calibration["alpha"],calibration["envelope"]
            evaluation = {}
            for variant in spec.VARIANTS:
                pairs = list(pool.map(parts.scenario.worker_native,[(weights[variant],s,s,variant,period,predictor,
                    0. if variant == "zero" else alpha,envelope,False) for s in roles["native_evaluation"]]))
                evaluation[variant] = [r for _,r in pairs]
                for batch,r in pairs:
                    if batch is not None:raise ValueError("Stage84 collected an unregistered training trace")
                    native.curves.paths.check_row(r,period,args.horizon)
                    cost["native_episodes"] += 1
                    cost["evaluation_episodes"] += 1
                    cost["native_steps"] += r["episode_length"]
                    cost["native_lower_calls"] += r["lower_calls"]
                    cost["native_upper_calls"] += r["upper_calls"]
                    cost["pairing_upper_forward_calls"] += r["upper_calls"]
                    cost["native_network_checks"] += 1
                    for key in planning:planning[key] += r[key]
                print(f"{spec.EXPERIMENT_PROTOCOL} {root}/{period}: {variant} evaluated, no training",flush=True)
            effects = paired_effects(period,evaluation,roles["native_evaluation"],protocol=spec)
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            native.curves.support.assert_frozen(original,snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"alpha":alpha,"checkpoints":checkpoints,"evaluation":evaluation,"effects":effects,
                "checkpoint_freeze":"passed","composition":"passed","pairing":"passed","source_and_Adam_unchanged":"passed"}
    cell = {"status":"complete","protocol":spec.EXPERIMENT_PROTOCOL,"contract":spec.contract(),"root":root,"preflight":preflight,
        "seed_roles":roles,"cost":cost,"native_planning_cost":planning,"groups":groups,"source_initialization":initialization,
        "optimizer_steps":0,"critic_fits":0,"forecaster_fits":0,"native_trace_writes":0,"wall_seconds":time.monotonic()-started}
    qualify(cell,preflight=preflight,protocol=spec)
    write_json(output,cell)
    write_json(output.parent/"completion"/"ready.json",{"protocol":spec.EXPERIMENT_PROTOCOL,"root":root,"preflight":preflight})
    return cell


def qualify(cell, *, preflight, protocol=spec):
    spec = protocol
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(cell["root"],preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("optimizer_steps","critic_fits","forecaster_fits","native_trace_writes"))):
        raise ValueError("Stage84 protocol, seeds or evaluation-only budget changed")
    h = spec.arguments(cell["root"],preflight=preflight).horizon
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS,0)
    for p,g in cell["groups"].items():
        if any(g[k] != "passed" for k in ("checkpoint_freeze","composition","pairing","source_and_Adam_unchanged")):
            raise ValueError("Stage84 checkpoint composition or source freeze failed")
        if set(g["checkpoints"]) != set(spec.CHECKPOINT_METHODS):raise ValueError("Registered checkpoint donors changed")
        effects = paired_effects(p,g["evaluation"],cell["seed_roles"]["native_evaluation"],protocol=spec)
        if g["effects"] != effects or not np.isfinite(list(effects.values())).all():raise ValueError("Stage84 paired effects changed")
        for variant,rows in g["evaluation"].items():
            for row in rows:
                native.curves.paths.check_row(row,int(p),h)
                if row["variant"] != variant or row["alpha"] != (0. if variant == "zero" else g["alpha"]):
                    raise ValueError("Stage84 frozen decoder changed")
                for key in planning:planning[key] += row[key]
    if planning != cell["native_planning_cost"]:raise ValueError("Stage84 native planning budget changed")
    return cell


def aggregate(cells, *, preflight, protocol=spec):
    result = native.aggregate(cells,preflight=preflight,protocol=protocol,
        qualifier=lambda c, **kw: qualify(c,protocol=protocol,**kw))
    result["performance_claim"] = "fixed_final_checkpoint_causal_upper_lower_swap_not_retraining_or_frequency_superiority"
    result["native_trial_prerequisite"] = "Stage67_critic_route_HOLD_unchanged_independent_MC_checkpoint_interventions"
    return result
