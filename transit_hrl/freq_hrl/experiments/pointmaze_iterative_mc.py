"""Learn mean networks over fresh native rounds without fitted critic credit."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from . import pointmaze_actor_parts as parts
from .pointmaze_root_response import write_json
from scripts import pointmaze_iterative_mc_stage83_spec as spec

native = parts.native


def check_training_freeze(model, before, active):
    after = model.state_dict()
    expected = copy.deepcopy(before)
    for actor in active:
        key = actor + "_actor"
        expected[key] = {**after[key], "log_std": before[key]["log_std"]}
    native.curves.support.assert_frozen(model, expected)


def update_mean(model, batches, *, method, period, horizon, cost, allocation=None, actor_batches=None, score_builder=None):
    allocation = spec.METHODS[method] if allocation is None else allocation
    score_builder = score_builder or parts.scenario_actor_scores
    before = copy.deepcopy(model.state_dict())
    if actor_batches is None:
        scored = score_builder(model, batches, period=period, horizon=horizon, cost=cost, actor_names=tuple(allocation))
    else:
        if set(actor_batches) != set(allocation):
            raise ValueError("Actor-specific credit must cover exactly the updated actors")
        scored = {}
        for name in allocation:
            scored.update(score_builder(model, actor_batches[name], period=period,
                horizon=horizon, cost=cost, actor_names=(name,)))
    changed, actors = {}, {}
    for name, score in scored.items():
        score_batches = batches if actor_batches is None else actor_batches[name]
        gradient = np.where(score["sigma_mask"], 0., np.concatenate(list(score["gradients"].values())).mean(0))
        actor = getattr(model, name + "_actor")
        candidates, geometry, c = native.direction.matched_perturbations(actor, score["states"], gradient,
            delta=spec.FISHER_RADIUS*allocation[name], chunk_size=spec.CHUNK_SIZE)
        for key in c:cost[key] += c[key]
        cost["actor_parameter_perturbations"] += len(candidates)
        parts.check_parameter_part(actor, candidates["plus"], "mean")
        cost["parameter_part_checks"] += 1
        changed[name] = candidates["plus"].state_dict()
        mask, gradients = ~score["sigma_mask"], score["gradients"]
        actors[name] = {"geometry": geometry, "std_check": "passed",
            "decision_calls_per_episode": horizon//period if name == "upper" else horizon,
            "gradient_episodes": len(score["states"])//(horizon//period if name == "upper" else horizon),
            "max_abs_old_logp_difference": max(r["max_abs_old_logp_difference"] for r in score["score_costs"].values()),
            "credit_batch_cosine": native.reliability.scores.cosine(gradients["A"].mean(0)[mask], gradients["B"].mean(0)[mask]),
            "scenario_covariance_trace": {b: parts.scenario.group_noise(g[:,mask],len(score_batches[b]),len(score_batches[b][0]))["covariance_trace"]
                for b,g in gradients.items()}}
    exact = sum(r["geometry"]["exact_kl"]["plus"] for r in actors.values())
    nominal = spec.FISHER_RADIUS*sum(allocation.values())
    if not .5*nominal <= exact <= 2*nominal:
        raise ValueError("Registered update sum-of-level KL failed")
    nominal_call = spec.FISHER_RADIUS*sum(v/(period if a == "upper" else 1) for a,v in allocation.items())
    exact_call = sum(r["geometry"]["exact_kl"]["plus"]/(period if a == "upper" else 1) for a,r in actors.items())
    for name, weights in changed.items():
        getattr(model,name+"_actor").load_state_dict(weights)
        cost["actor_mean_parameter_updates"] += 1
    check_training_freeze(model,before,allocation)
    cost["training_freeze_checks"] += 1
    cost["policy_updates"] += 1
    return {"actors": actors, "exact_sum_kl": exact, "nominal_call_weighted_kl": nominal_call,
        "exact_call_weighted_kl": exact_call, "freeze_check": "passed"}


def final_checkpoint(model, output, *, root, period, method, updates, protocol=spec):
    path = output.parent / "final_weights" / f"period_{period}_{method}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": protocol.EXPERIMENT_PROTOCOL, "root": root, "period": period, "method": method,
        "updates": updates, "weights": native.joint.inference_weights(model)},path)
    return str(path)


def load_source(root, *, protocol=spec):
    source = json.loads(protocol.source_result(root).read_text())
    decoder_spec = parts.scenario.spec.source.source
    if (source["status"],source["protocol"],source["root"],source["preflight"],source["contract"]) != (
            "complete",decoder_spec.EXPERIMENT_PROTOCOL,root,False,decoder_spec.contract()):
        raise ValueError("MC training requires the frozen full Stage78 decoder")
    clones,predictor,initialization = native.curves.support.native.load_source(root,preflight=False)
    calibrations = {str(p): source["groups"][str(p)]["calibration"] for p in protocol.PERIODS}
    for calibration in calibrations.values():
        native.bounded.check_calibration(calibration)
    return clones, predictor, initialization, calibrations


def run(root, *, preflight, output, protocol=spec, qualifier=None, evaluation_weights=None, initialize_models=None,
        training_pair_worker=None, scenario_pair_check=None, source_loader=None, actor_credit_collector=None,
        training_loop=None):
    spec = protocol
    qualifier = qualifier or (lambda c, **kw: qualify(c, protocol=protocol, **kw))
    scenario_pair_check = scenario_pair_check or parts.scenario.check_scenario_pair
    source_loader = source_loader or (lambda r: load_source(r, protocol=protocol))
    clones,predictor,initialization,calibrations = source_loader(root)
    args,roles,o = spec.arguments(root,preflight=preflight),spec.seed_roles(root,preflight=preflight),spec.options(preflight=preflight)
    cost,groups,started = dict.fromkeys(spec.budget(preflight=preflight),0),{},time.monotonic()
    cost.update(source_clone_loads=len(clones),forecaster_loads=1,decoder_loads=len(clones))
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS,0)
    with ProcessPoolExecutor(max_workers=o["workers"],mp_context=mp.get_context("spawn"),initializer=native.init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config,args)) as pool:
        for period in spec.PERIODS:
            original = clones[str(period)]
            snapshot = copy.deepcopy(original.state_dict())
            calibration = calibrations[str(period)]
            alpha,envelope = calibration["alpha"],calibration["envelope"]

            def episodes(weights,roster,variant,collect=False,*,pair_worker=training_pair_worker):
                jobs = [(weights,s,n,variant,period,predictor,
                    0. if variant == "zero" else alpha,envelope,collect) for s,n in roster]
                if collect and pair_worker is not None:
                    replicas = o["rollouts_per_scenario"]
                    pairs = [pair for group in pool.map(pair_worker,
                        [jobs[i:i+replicas] for i in range(0,len(jobs),replicas)]) for pair in group]
                else:
                    pairs = list(pool.map(parts.scenario.worker_native,jobs))
                for _,r in pairs:
                    native.curves.paths.check_row(r,period,args.horizon)
                    cost["native_episodes"] += 1
                    cost["native_steps"] += r["episode_length"]
                    cost["native_lower_calls"] += r["lower_calls"]
                    cost["native_upper_calls"] += r["upper_calls"]
                    cost["pairing_upper_forward_calls"] += r["upper_calls"]
                    cost["native_network_checks"] += 1
                    if "upper_replay_forward_calls" in cost:
                        cost["upper_replay_forward_calls"] += r["upper_replay_forward_calls"]
                    cost["credit_episodes" if collect else "evaluation_episodes"] += 1
                    for key in planning:planning[key] += r[key]
                return pairs

            models = {m:copy.deepcopy(original) for m in spec.METHODS}
            cost["training_models_initialized"] += len(models)
            initialization_metadata = {} if initialize_models is None else initialize_models(period,models,cost)
            training_snapshots = {m:copy.deepcopy(model.state_dict()) for m,model in models.items()}
            history = {m:[] for m in spec.METHODS}
            if training_loop is not None:
                history = training_loop(period, models, roles, episodes, cost)
            else:
                for iteration,round_roles in enumerate(roles["training_rounds"],1):
                    for method,model in models.items():
                        actor_batches = None
                        if actor_credit_collector is None:
                            batches = {}
                            for b in ("A","B"):
                                roster = round_roles["credit_"+b]
                                pairs = episodes(native.joint.inference_weights(model),[(s["scenario_seed"],n) for s in roster for n in s["noise_seeds"]],method,True)
                                replicas = o["rollouts_per_scenario"]
                                batches[b] = [pairs[i:i+replicas] for i in range(0,len(pairs),replicas)]
                                for group,s in zip(batches[b],roster):
                                    scenario_pair_check(group,s)
                                    cost["scenario_pair_checks"] += 1
                            batch_sets = [batches]
                        else:
                            actor_batches = actor_credit_collector(episodes,native.joint.inference_weights(model),
                                round_roles,method,cost)
                            batches = None
                            batch_sets = actor_batches.values()
                        mean_reward = float(np.mean([r["episode_return"] for bs in batch_sets for g in bs.values() for pairs in g for _,r in pairs]))
                        row = update_mean(model,batches,method=method,period=period,horizon=args.horizon,cost=cost,
                            allocation=spec.allocation(method,period),actor_batches=actor_batches)
                        del batches, actor_batches, batch_sets
                        row.update(iteration=iteration,credit_mean_reward_before_update=mean_reward)
                        history[method].append(row)
                    print(f"{spec.EXPERIMENT_PROTOCOL} {root}/{period}: registered update {iteration}/{o['updates']} done, no evaluation selection",flush=True)
            weights = {m:native.joint.inference_weights(model) for m,model in models.items()}
            weights.update(base=native.joint.inference_weights(original),zero=native.joint.inference_weights(original))
            evaluation_metadata = {}
            if evaluation_weights is not None:
                weights,evaluation_metadata = evaluation_weights(period,weights,cost)
            evaluation = {v:[r for _,r in episodes(weights[v],[(s,s) for s in roles["native_evaluation"]],v)] for v in spec.VARIANTS}
            effects = native.paired_effects(period,evaluation,roles["native_evaluation"],protocol=spec)
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            trained = {}
            for method,model in models.items():
                checkpoint = None if preflight else final_checkpoint(model,output,root=root,period=period,method=method,
                    updates=o["updates"],protocol=spec)
                cost["checkpoint_writes"] += int(checkpoint is not None)
                check_training_freeze(model,training_snapshots[method],spec.METHODS[method])
                trained[method] = {"history":history[method],"evaluation_update":o["updates"],"checkpoint":checkpoint,"final_freeze_check":"passed"}
            native.curves.support.assert_frozen(original,snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"alpha":alpha,"trained":trained,"evaluation":evaluation,"effects":effects,
                "scenario_pairing":"passed","pairing":"passed","source_and_Adam_unchanged":"passed",
                **initialization_metadata,**evaluation_metadata}
    cell = {"status":"complete","protocol":spec.EXPERIMENT_PROTOCOL,"contract":spec.contract(),"root":root,"preflight":preflight,
        "seed_roles":roles,"cost":cost,"native_planning_cost":planning,"groups":groups,"source_initialization":initialization,
        "optimizer_steps":0,"critic_fits":0,"forecaster_fits":0,"native_trace_writes":0,"wall_seconds":time.monotonic()-started}
    qualifier(cell,preflight=preflight)
    write_json(output,cell)
    write_json(output.parent/"completion"/"ready.json",{"protocol":spec.EXPERIMENT_PROTOCOL,"root":root,"preflight":preflight})
    return cell


def qualify(cell, *, preflight, protocol=spec, update_schedule=None):
    spec = protocol
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(cell["root"],preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("optimizer_steps","critic_fits","forecaster_fits","native_trace_writes"))):
        raise ValueError("MC protocol, seeds or registered training budget changed")
    o,h = spec.options(preflight=preflight),spec.arguments(cell["root"],preflight=preflight).horizon
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS,0)
    for p,g in cell["groups"].items():
        if any(g[k] != "passed" for k in ("scenario_pairing","pairing","source_and_Adam_unchanged")):
            raise ValueError("MC on-policy independence or source freeze failed")
        expected = native.paired_effects(p,g["evaluation"],cell["seed_roles"]["native_evaluation"],protocol=spec)
        if g["effects"] != expected or not np.isfinite(list(expected.values())).all():raise ValueError("MC final reward contrasts changed")
        if set(g["trained"]) != set(spec.METHODS):raise ValueError("MC training methods changed")
        for method,t in g["trained"].items():
            steps = ([{"iteration": i, "allocation": spec.allocation(method,int(p))} for i in range(1,o["updates"]+1)]
                if update_schedule is None else [s for s in update_schedule(int(p),preflight=preflight) if s["method"] == method])
            if (t["evaluation_update"] != o["updates"] or t["final_freeze_check"] != "passed"
                    or bool(t["checkpoint"]) == preflight or [r["iteration"] for r in t["history"]] != [s["iteration"] for s in steps]):
                raise ValueError("MC intermediate selection or final model changed")
            for r,step in zip(t["history"],steps):
                allocation = step["allocation"]
                if r["freeze_check"] != "passed" or set(r["actors"]) != set(allocation):raise ValueError("Registered actor or frozen parameters changed")
                exact = 0.
                for a,row in r["actors"].items():
                    geometry = row["geometry"]
                    if (row["std_check"] != "passed" or geometry["radius_check"] != "passed"
                            or geometry["nominal_fisher_kl"] != spec.FISHER_RADIUS*allocation[a]):
                        raise ValueError("MC fixed allocation/radius or std changed")
                    exact += geometry["exact_kl"]["plus"]
                nominal = spec.FISHER_RADIUS*sum(allocation.values())
                if exact != r["exact_sum_kl"] or not .5*nominal <= exact <= 2*nominal:
                    raise ValueError("MC total per-round conditional KL changed")
        for v,rows in g["evaluation"].items():
            for r in rows:
                native.curves.paths.check_row(r,int(p),h)
                if r["variant"] != v or r["alpha"] != (0. if v == "zero" else g["alpha"]):raise ValueError("MC frozen decoder changed")
                for key in planning:planning[key] += r[key]
        n = cell["cost"]["credit_episodes"]//len(spec.PERIODS)
        for key in planning:planning[key] += n*(h if key in ("reference_evaluations","actor_context_evaluations") else h//int(p)-1)
    if planning != cell["native_planning_cost"]:raise ValueError("MC native planning budget changed")
    return cell


def aggregate(cells, *, preflight, protocol=spec, qualifier=None):
    qualifier = qualifier or (lambda c, **kw: qualify(c, protocol=protocol, **kw))
    result = native.aggregate(cells,preflight=preflight,protocol=protocol,qualifier=qualifier)
    result["performance_claim"] = "teacher_initialized_fixed_std_decoder_iterative_MC_mean_learning_not_full_actor_critic"
    result["native_trial_prerequisite"] = "Stage67_critic_route_HOLD_unchanged_independent_MC_training_protocol"
    return result
