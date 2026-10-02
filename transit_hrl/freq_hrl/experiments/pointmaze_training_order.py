"""Native MC training with matched per-level updates and frozen order schedules."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from . import pointmaze_iterative_mc as learning
from .pointmaze_root_response import write_json
from scripts import pointmaze_training_order_stage85_spec as spec

native,parts = learning.native,learning.parts


def train_model(model, *, method, period, horizon, preflight, collect_chunk, cost):
    history = []
    for index,group in enumerate(spec.training_groups(method,preflight=preflight)):
        before = copy.deepcopy(model.state_dict())
        version = len(history)
        batches = {c:collect_chunk(model,c) for c in group["chunks"]}
        native.curves.support.assert_frozen(model,before)
        cost["collection_freeze_checks"] += 1
        # Paired updates use different chunks collected before either actor changes.
        for update in group["updates"]:
            selected = {b:[pairs for c in update["chunks"] for pairs in batches[c][b]] for b in ("A","B")}
            allocation = {update["actor"]:update["fraction"]}
            row = learning.update_mean(model,selected,method=method,period=period,horizon=horizon,cost=cost,allocation=allocation)
            row.update(update=len(history)+1,group_index=index,credit_chunks=update["chunks"],allocation=allocation,
                rollout_policy_update_count=version,credit_episodes_used=sum(len(pairs) for groups in selected.values() for pairs in groups),
                credit_mean_reward_before_update=float(np.mean([r["episode_return"] for groups in selected.values() for pairs in groups for _,r in pairs])))
            history.append(row)
        del batches
        print(f"Stage85 {period}/{method}: collection group {index+1}/{len(spec.training_groups(method,preflight=preflight))} done",flush=True)
    return history


def check_history(history, *, method, preflight):
    groups = spec.training_groups(method,preflight=preflight)
    expected = [(i,u,version) for i,g in enumerate(groups)
        for version in [sum(len(x["updates"]) for x in groups[:i])] for u in g["updates"]]
    if len(history) != len(expected):raise ValueError("Stage85 registered update count changed")
    nominal,exact = dict.fromkeys(("upper","lower"),0.),dict.fromkeys(("upper","lower"),0.)
    o = spec.options(preflight=preflight)
    n = 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]
    for j,(row,(index,u,version)) in enumerate(zip(history,expected),1):
        allocation = {u["actor"]:u["fraction"]}
        if (row["update"] != j or row["group_index"] != index or row["credit_chunks"] != u["chunks"]
                or row["allocation"] != allocation or row["rollout_policy_update_count"] != version
                or row["credit_episodes_used"] != n*len(u["chunks"]) or set(row["actors"]) != set(allocation)
                or row["freeze_check"] != "passed"):
            raise ValueError("Stage85 order, collection policy or per-gradient samples changed")
        actor = u["actor"]
        diagnostic = row["actors"][actor]
        geometry = diagnostic["geometry"]
        radius = spec.FISHER_RADIUS*u["fraction"]
        actual = geometry["exact_kl"]["plus"]
        if (diagnostic["std_check"] != "passed" or geometry["radius_check"] != "passed"
                or geometry["nominal_fisher_kl"] != radius or row["exact_sum_kl"] != actual
                or not .5*radius <= actual <= 2*radius):
            raise ValueError("Stage85 registered per-level mean KL or std freeze changed")
        nominal[actor] += radius
        exact[actor] += actual
    if not np.isclose(sum(nominal.values()),o["credit_chunks"]*.5*spec.FISHER_RADIUS,atol=1e-12,rtol=0):
        raise ValueError("Stage85 cumulative nominal KL changed")
    return {"nominal_by_level":nominal,"exact_by_level":exact,"matching":"passed"}


def final_checkpoint(model, output, *, root, period, method, history):
    path = output.parent/"final_weights"/f"period_{period}_{method}.pt"
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"protocol":spec.EXPERIMENT_PROTOCOL,"root":root,"period":period,"method":method,
        "updates":len(history),"credit_chunks":spec.options(preflight=False)["credit_chunks"],
        "weights":native.joint.inference_weights(model)},path)
    return str(path)


def run(root, *, preflight, output):
    source = json.loads(spec.source_result(root).read_text())
    decoder_spec = parts.scenario.spec.source.source
    if (source["status"],source["protocol"],source["root"],source["preflight"],source["contract"]) != (
            "complete",decoder_spec.EXPERIMENT_PROTOCOL,root,False,decoder_spec.contract()):
        raise ValueError("Stage85 requires the frozen full Stage78 decoder")
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
            calibration = source["groups"][str(period)]["calibration"]
            native.bounded.check_calibration(calibration)
            alpha,envelope = calibration["alpha"],calibration["envelope"]

            def episodes(weights,roster,variant,collect=False):
                pairs = list(pool.map(parts.scenario.worker_native,[(weights,s,n,variant,period,predictor,
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

            models = {m:copy.deepcopy(original) for m in spec.METHODS}
            cost["training_models_initialized"] += len(models)
            trained = {}
            for method,model in models.items():
                def collect_chunk(current,index):
                    batches = {}
                    for b in ("A","B"):
                        roster = roles["training_chunks"][index]["credit_"+b]
                        pairs = episodes(native.joint.inference_weights(current),[(s["scenario_seed"],n)
                            for s in roster for n in s["noise_seeds"]],method,True)
                        batches[b] = [pairs[i:i+2] for i in range(0,len(pairs),2)]
                        for group,s in zip(batches[b],roster):
                            parts.scenario.check_scenario_pair(group,s)
                            cost["scenario_pair_checks"] += 1
                    return batches

                history = train_model(model,method=method,period=period,horizon=args.horizon,preflight=preflight,
                    collect_chunk=collect_chunk,cost=cost)
                cumulative = check_history(history,method=method,preflight=preflight)
                active = ("lower",) if method == "lower_trained" else ("upper","lower")
                learning.check_training_freeze(model,snapshot,active)
                checkpoint = None if preflight else final_checkpoint(model,output,root=root,period=period,method=method,history=history)
                cost["checkpoint_writes"] += int(checkpoint is not None)
                trained[method] = {"history":history,"cumulative_KL":cumulative,"evaluation_update":len(history),
                    "checkpoint":checkpoint,"final_freeze_check":"passed"}
            weights = {m:native.joint.inference_weights(model) for m,model in models.items()}
            weights.update(base=native.joint.inference_weights(original),zero=native.joint.inference_weights(original))
            evaluation = {v:[r for _,r in episodes(weights[v],[(s,s) for s in roles["native_evaluation"]],v)] for v in spec.VARIANTS}
            effects = native.paired_effects(period,evaluation,roles["native_evaluation"],protocol=spec)
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            native.curves.support.assert_frozen(original,snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"alpha":alpha,"trained":trained,"evaluation":evaluation,"effects":effects,
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
        raise ValueError("Stage85 protocol, seeds or registered training budget changed")
    o,h = spec.options(preflight=preflight),spec.arguments(cell["root"],preflight=preflight).horizon
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS,0)
    for p,g in cell["groups"].items():
        if any(g[k] != "passed" for k in ("scenario_pairing","pairing","source_and_Adam_unchanged")):
            raise ValueError("Stage85 native pairing or source freeze failed")
        expected = native.paired_effects(p,g["evaluation"],cell["seed_roles"]["native_evaluation"],protocol=spec)
        if g["effects"] != expected or not np.isfinite(list(expected.values())).all():raise ValueError("Stage85 final reward contrasts changed")
        if set(g["trained"]) != set(spec.METHODS):raise ValueError("Stage85 training methods changed")
        for method,t in g["trained"].items():
            if (t["evaluation_update"] != len(t["history"]) or t["final_freeze_check"] != "passed" or bool(t["checkpoint"]) == preflight
                    or t["cumulative_KL"] != check_history(t["history"],method=method,preflight=preflight)):
                raise ValueError("Stage85 selected checkpoint or cumulative KL changed")
        for variant,rows in g["evaluation"].items():
            for row in rows:
                native.curves.paths.check_row(row,int(p),h)
                if row["variant"] != variant or row["alpha"] != (0. if variant == "zero" else g["alpha"]):
                    raise ValueError("Stage85 frozen decoder changed")
                for key in planning:planning[key] += row[key]
        n = len(spec.METHODS)*o["credit_chunks"]*2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]
        for key in planning:planning[key] += n*(h if key in ("reference_evaluations","actor_context_evaluations") else h//int(p)-1)
    if planning != cell["native_planning_cost"]:raise ValueError("Stage85 native planning budget changed")
    return cell


def aggregate(cells, *, preflight):
    result = native.aggregate(cells,preflight=preflight,protocol=spec,qualifier=qualify)
    result["performance_claim"] = "matched_sample_cumulative_nominal_KL_MC_training_order_not_full_actor_critic_or_frequency_superiority"
    result["native_trial_prerequisite"] = "Stage67_critic_route_HOLD_unchanged_independent_MC_training_protocol"
    return result
