"""Separate direction-fitting effects from native upper-execution effects."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from . import pointmaze_native_direction as previous
from .pointmaze_root_response import write_json
from scripts import pointmaze_crossed_direction_stage74_spec as spec


def paired_endpoints(period, evaluation, seeds):
    if set(evaluation) != set(spec.EXECUTION_ARMS):
        raise ValueError("Stage74 execution roster changed")
    for rows_by_variant in evaluation.values():
        if set(rows_by_variant) != set(spec.VARIANTS) or any([r["seed"] for r in rows] != seeds for rows in rows_by_variant.values()):
            raise ValueError("Stage74 variant or seed roster changed")
    anchor = evaluation[spec.EXECUTION_ARMS[0]]["base"]
    for rows_by_variant in evaluation.values():
        for rows in rows_by_variant.values():
            for row, base in zip(rows, anchor):
                for key in ("policy_seed", "lower_seed", "decision_steps", "upper_proposed_actions"):
                    if row[key] != base[key]:raise ValueError("Stage74 within/across execution pairing changed")
    effects = {}
    for e, rows in evaluation.items():
        reward = {v: np.asarray([r["episode_return"] for r in rs]) for v, rs in rows.items()}
        for f in spec.FIT_ARMS:
            for d in spec.DIRECTIONS:
                plus, minus, base = reward[spec.variant(f, d, "plus")], reward[spec.variant(f, d, "minus")], reward["base"]
                for c, a in (("plus_minus", plus - minus), ("plus_base", plus - base), ("minus_base", minus - base)):
                    effects[f"cell/{period}/{f}/{e}/{d}/{c}"] = float(a.mean())
    z, j = spec.EXECUTION_ARMS
    fz, fj = spec.FIT_ARMS
    for d in spec.DIRECTIONS:
        for f in spec.FIT_ARMS:
            effects[f"execution/{period}/{f}/{d}"] = (effects[f"cell/{period}/{f}/{j}/{d}/plus_base"] -
                effects[f"cell/{period}/{f}/{z}/{d}/plus_base"])
        for e in spec.EXECUTION_ARMS:
            effects[f"fitting/{period}/{e}/{d}"] = (effects[f"cell/{period}/{fj}/{e}/{d}/plus_base"] -
                effects[f"cell/{period}/{fz}/{e}/{d}/plus_base"])
        effects[f"interaction/{period}/{d}"] = effects[f"execution/{period}/{fj}/{d}"] - effects[f"execution/{period}/{fz}/{d}"]
    return effects


def check_reproduction(fitted, reference):
    if any(fitted[key] != reference[key] for key in ("geometry", "native_baseline_frame")):
        raise ValueError("Stage74 failed to reproduce frozen Stage73 directions")


def run(root, *, preflight, output):
    old = json.loads(spec.source_result(root, preflight=preflight).read_text())
    previous.qualify(old, preflight=preflight)
    if old["root"] != root:raise ValueError("Stage74 prerequisite root changed")
    legacy = previous.spec.source.legacy
    historical_file = legacy.values_source.training_result(root, preflight=preflight)
    history = json.loads(historical_file.read_text())
    archive = historical_file.parent.with_name(historical_file.parent.name + "_raw")
    controls = json.loads(legacy.values_source.source_result(root, preflight=preflight).read_text())
    factored = json.loads(legacy.source.source_result(root, preflight=preflight).read_text())
    clones, predictor, initialization = previous.native.load_source(root, preflight=preflight)
    opt, args, roles = spec.options(preflight=preflight), spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight)
    cost, groups, started = dict.fromkeys(spec.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    planning = dict.fromkeys(("plan_ols_fits", "plan_ridge_predictions", "reference_evaluations", "actor_context_evaluations"), 0)
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=previous.init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            snapshot = copy.deepcopy(clone.state_dict())
            fitting, weights = {}, {"base": previous.joint.inference_weights(clone)}
            for f in spec.FIT_ARMS:
                candidates, fitted = previous.historical_directions(clone, root=root, period=period, arm=f,
                    history=history, archive=archive, controls=controls, factored=factored,
                    roles=roles, args=args, opt=opt, pool=pool, cost=cost)
                check_reproduction(fitted, old["groups"][p][f])
                cost["stage73_direction_reproductions"] += len(spec.DIRECTIONS)
                fitting[f] = {**fitted, "stage73_direction_reproduction": "passed"}
                for d in spec.DIRECTIONS:
                    for sign in ("plus", "minus"):weights[spec.variant(f, d, sign)] = candidates[d + "_" + sign]
            print(f"crossed direction {root}/{p}: both historical arms reproduced; all directions frozen", flush=True)
            evaluation = {}
            for e in spec.EXECUTION_ARMS:
                evaluation[e] = {}
                for v in spec.VARIANTS:
                    rows = list(pool.map(previous.worker_native, [(weights[v], s, e, period, predictor) for s in roles["native_evaluation"]]))
                    for row in rows:
                        if (row["episode_length"] != args.horizon or row["lower_calls"] != args.horizon or
                                row["upper_calls"] != args.horizon // period or row["network_check"] != "passed"):
                            raise ValueError("Stage74 native inference or source weights changed")
                        cost["native_episodes"] += 1
                        cost["native_steps"] += row["episode_length"]
                        cost["native_lower_calls"] += row["lower_calls"]
                        cost["native_upper_calls"] += row["upper_calls"]
                        cost["native_network_checks"] += 1
                        for k in planning:planning[k] += row[k]
                    evaluation[e][v] = rows
                cost["native_pair_checks"] += len(roles["native_evaluation"])
                print(f"crossed direction {root}/{p}/{e}: all13 variants completed", flush=True)
            effects = paired_endpoints(p, evaluation, roles["native_evaluation"])
            cost["cross_execution_pair_checks"] += len(roles["native_evaluation"])
            previous.independent.assert_frozen(clone, snapshot)
            cost["frozen_model_checks"] += 1
            groups[p] = {"fitting": fitting, "evaluation": evaluation, "effects": effects,
                "pairing": "passed", "source_and_Adam_unchanged": "passed", "candidate_reuse_across_execution": "same_parameters_and_step"}
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": cost, "native_planning_cost": planning,
        "groups": groups, "source_initialization": initialization, "optimizer_steps": 0, "critic_fits": 0,
        "forecaster_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0, "wall_seconds": time.monotonic() - started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or any(cell[k] for k in ("optimizer_steps", "critic_fits", "forecaster_fits", "checkpoint_writes", "native_trace_writes"))
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage74 frozen protocol or budget changed")
    for p, group in cell["groups"].items():
        if (set(group["fitting"]) != set(spec.FIT_ARMS) or group["pairing"] != "passed" or
                group["source_and_Adam_unchanged"] != "passed" or group["candidate_reuse_across_execution"] != "same_parameters_and_step"):
            raise ValueError("Stage74 direction fitting or source changed")
        for fit in group["fitting"].values():
            if (fit["stage73_direction_reproduction"] != "passed" or fit["model_and_Adam_unchanged"] != "passed" or
                    set(fit["geometry"]) != set(spec.DIRECTIONS)):
                raise ValueError("Stage74 historical direction reproduction failed")
        if group["effects"] != paired_endpoints(p, group["evaluation"], cell["seed_roles"]["native_evaluation"]):
            raise ValueError("Stage74 paired contrast accounting changed")
    return cell


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage74 requires every frozen root")
    rows = [qualify(c, preflight=preflight) for c in sorted(cells, key=lambda c: c["root"])]
    effects = [{key: value for g in c["groups"].values() for key, value in g["effects"].items()} for c in rows]
    x = np.asarray([[e[k] for k in spec.ENDPOINTS] for e in effects])
    endpoints = {k: {"mean": float(x[:, i].mean())} for i, k in enumerate(spec.ENDPOINTS)}
    if not preflight:
        idx = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(rows), (spec.BOOTSTRAP_DRAWS, len(rows)))
        tail = .05 / (2 * len(spec.ENDPOINTS))
        bounds = np.quantile(x[idx].mean(1), [tail, 1 - tail], axis=0)
        for i, k in enumerate(spec.ENDPOINTS):
            endpoints[k].update(ci=bounds[:, i].tolist(), effect="positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive")
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "root_rows": rows, "endpoints": endpoints,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in spec.budget(preflight=preflight)},
        "native_planning_cost": {k: sum(c["native_planning_cost"][k] for c in rows) for k in rows[0]["native_planning_cost"]},
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "none_crossed_response_diagnosis_only"}
