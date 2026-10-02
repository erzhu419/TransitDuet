"""Fix staged per-level scenario assignment while reusing Stage85 baselines."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from . import pointmaze_training_order as order
from .pointmaze_root_response import write_json
from scripts import pointmaze_aligned_order_stage86_spec as spec

native,parts = order.native,order.parts


def baseline_model(payload, original, *, root, period, method):
    updates = 8 if method == "lower_trained" else 16
    expected = {"protocol":spec.source.EXPERIMENT_PROTOCOL,"root":root,"period":period,"method":method,
        "updates":updates,"credit_chunks":16}
    if {k:v for k,v in payload.items() if k != "weights"} != expected:
        raise ValueError("Stage86 requires registered final Stage85 baseline weights")
    if set(payload["weights"]) != set(native.joint.inference_weights(original)):
        raise ValueError("Stage86 baseline network set changed")
    model = copy.deepcopy(original)
    model.load_state_dict(payload["weights"])
    active = ("lower",) if method == "lower_trained" else ("upper","lower")
    order.learning.check_training_freeze(model,original.state_dict(),active)
    torch.testing.assert_close(native.joint.inference_weights(model),payload["weights"],atol=0,rtol=0)
    return model


def annotate_history(history, *, preflight):
    mapping = spec.source_chunk_order(preflight=preflight)
    for row in history:row["source_credit_chunks"] = [mapping[c] for c in row["credit_chunks"]]
    return history


def check_history(history, *, preflight):
    cumulative = order.check_history(history,method="staged",preflight=preflight)
    mapping = spec.source_chunk_order(preflight=preflight)
    for row in history:
        expected = [mapping[c] for c in row["credit_chunks"]]
        actor = next(iter(row["actors"]))
        if row["source_credit_chunks"] != expected or any(c%2 != int(actor == "upper") for c in expected):
            raise ValueError("Stage86 changed the paired/alternating per-level scenario roster")
    return cumulative


def final_checkpoint(model, output, *, root, period, history):
    path = output.parent/"final_weights"/f"period_{period}_staged_aligned.pt"
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"protocol":spec.EXPERIMENT_PROTOCOL,"root":root,"period":period,"method":"staged_aligned",
        "updates":len(history),"source_chunk_order":spec.source_chunk_order(preflight=False),
        "weights":native.joint.inference_weights(model)},path)
    return str(path)


def run(root, *, preflight, output):
    decoder = json.loads(spec.source_result(root).read_text())
    decoder_spec = parts.scenario.spec.source.source
    if (decoder["status"],decoder["protocol"],decoder["root"],decoder["preflight"],decoder["contract"]) != (
            "complete",decoder_spec.EXPERIMENT_PROTOCOL,root,False,decoder_spec.contract()):
        raise ValueError("Stage86 requires frozen full Stage78 decoder")
    training = json.loads(spec.training_result(root).read_text())
    if (training["status"],training["protocol"],training["root"],training["preflight"],training["contract"]) != (
            "complete",spec.source.EXPERIMENT_PROTOCOL,root,False,spec.source.contract()):
        raise ValueError("Stage86 requires completed full Stage85 training")
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
            calibration = decoder["groups"][str(period)]["calibration"]
            native.bounded.check_calibration(calibration)
            alpha,envelope = calibration["alpha"],calibration["envelope"]
            weights,checkpoints = {},{}
            for variant,method in spec.BASELINES.items():
                path = spec.training_result(root).parent/"final_weights"/f"period_{period}_{method}.pt"
                record = training["groups"][str(period)]["trained"][method]
                if record["checkpoint"] != str(path) or record["final_freeze_check"] != "passed":
                    raise ValueError("Stage86 baseline checkpoint record changed")
                model = baseline_model(torch.load(path,map_location="cpu",weights_only=False),original,
                    root=root,period=period,method=method)
                weights[variant] = native.joint.inference_weights(model)
                checkpoints[variant] = str(path)
                cost["baseline_checkpoint_loads"] += 1
                cost["baseline_freeze_checks"] += 1

            def episodes(w,roster,variant,collect=False):
                pairs = list(pool.map(parts.scenario.worker_native,[(w,s,n,variant,period,predictor,
                    0. if variant == "zero" else alpha,envelope,collect) for s,n in roster]))
                for _,r in pairs:
                    native.curves.paths.check_row(r,period,args.horizon)
                    cost["native_episodes"] += 1
                    cost["native_steps"] += r["episode_length"]
                    cost["native_lower_calls"] += r["lower_calls"]
                    cost["native_upper_calls"] += r["upper_calls"]
                    cost["pairing_upper_forward_calls"] += r["upper_calls"]
                    cost["native_network_checks"] += 1
                    cost["credit_episodes" if collect else "evaluation_episodes"] += 1
                    for key in planning:planning[key] += r[key]
                return pairs

            def collect_chunk(current,index):
                batches = {}
                for b in ("A","B"):
                    roster = roles["training_chunks"][index]["credit_"+b]
                    pairs = episodes(native.joint.inference_weights(current),[(s["scenario_seed"],n)
                        for s in roster for n in s["noise_seeds"]],"staged_aligned",True)
                    batches[b] = [pairs[i:i+2] for i in range(0,len(pairs),2)]
                    for group,s in zip(batches[b],roster):
                        parts.scenario.check_scenario_pair(group,s)
                        cost["scenario_pair_checks"] += 1
                return batches

            model = copy.deepcopy(original)
            cost["training_models_initialized"] += 1
            history = order.train_model(model,method="staged",period=period,horizon=args.horizon,preflight=preflight,
                collect_chunk=collect_chunk,cost=cost)
            annotate_history(history,preflight=preflight)
            cumulative = check_history(history,preflight=preflight)
            order.learning.check_training_freeze(model,snapshot,("upper","lower"))
            checkpoint = None if preflight else final_checkpoint(model,output,root=root,period=period,history=history)
            cost["checkpoint_writes"] += int(checkpoint is not None)
            weights.update(staged_aligned=native.joint.inference_weights(model),
                base=native.joint.inference_weights(original),zero=native.joint.inference_weights(original))
            evaluation = {v:[r for _,r in episodes(weights[v],[(s,s) for s in roles["native_evaluation"]],v)] for v in spec.VARIANTS}
            effects = native.paired_effects(period,evaluation,roles["native_evaluation"],protocol=spec)
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            native.curves.support.assert_frozen(original,snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"alpha":alpha,"baseline_checkpoints":checkpoints,"baseline_freeze":"passed",
                "trained":{"history":history,"cumulative_KL":cumulative,"evaluation_update":len(history),
                    "checkpoint":checkpoint,"final_freeze_check":"passed"},"evaluation":evaluation,"effects":effects,
                "scenario_pairing":"passed","pairing":"passed","source_and_Adam_unchanged":"passed"}
    cell = {"status":"complete","protocol":spec.EXPERIMENT_PROTOCOL,"contract":spec.contract(),"root":root,"preflight":preflight,
        "seed_roles":roles,"cost":cost,"native_planning_cost":planning,"groups":groups,"source_initialization":initialization,
        "optimizer_steps":0,"critic_fits":0,"forecaster_fits":0,"native_trace_writes":0,"wall_seconds":time.monotonic()-started}
    qualify(cell,preflight=preflight)
    write_json(output,cell)
    write_json(output.parent/"completion"/"ready.json",{"protocol":spec.EXPERIMENT_PROTOCOL,"root":root,"preflight":preflight})
    return cell


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(cell["root"],preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("optimizer_steps","critic_fits","forecaster_fits","native_trace_writes"))):
        raise ValueError("Stage86 protocol, repaired datasets or registered budget changed")
    o,h = spec.options(preflight=preflight),spec.arguments(cell["root"],preflight=preflight).horizon
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS,0)
    for p,g in cell["groups"].items():
        if any(g[k] != "passed" for k in ("baseline_freeze","scenario_pairing","pairing","source_and_Adam_unchanged")):
            raise ValueError("Stage86 baseline freeze or pairing failed")
        if set(g["baseline_checkpoints"]) != set(spec.BASELINES):raise ValueError("Stage86 baseline donors changed")
        t = g["trained"]
        if (t["evaluation_update"] != len(t["history"]) or t["final_freeze_check"] != "passed" or bool(t["checkpoint"]) == preflight
                or t["cumulative_KL"] != check_history(t["history"],preflight=preflight)):
            raise ValueError("Stage86 final checkpoint or repaired actor data assignment changed")
        expected = native.paired_effects(p,g["evaluation"],cell["seed_roles"]["native_evaluation"],protocol=spec)
        if g["effects"] != expected or not np.isfinite(list(expected.values())).all():raise ValueError("Stage86 final contrasts changed")
        for variant,rows in g["evaluation"].items():
            for row in rows:
                native.curves.paths.check_row(row,int(p),h)
                if row["variant"] != variant or row["alpha"] != (0. if variant == "zero" else g["alpha"]):
                    raise ValueError("Stage86 frozen decoder changed")
                for key in planning:planning[key] += row[key]
        n = o["credit_chunks"]*2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]
        for key in planning:planning[key] += n*(h if key in ("reference_evaluations","actor_context_evaluations") else h//int(p)-1)
    if planning != cell["native_planning_cost"]:raise ValueError("Stage86 native planning budget changed")
    return cell


def aggregate(cells, *, preflight):
    result = native.aggregate(cells,preflight=preflight,protocol=spec,qualifier=qualify)
    result["performance_claim"] = "corrected_per_actor_scenario_aligned_training_order_not_new_independent_training_replication"
    result["native_trial_prerequisite"] = "Stage67_critic_route_HOLD_unchanged_independent_MC_order_repair"
    return result
