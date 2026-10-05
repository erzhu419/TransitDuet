"""Separate plan amplitude from the strength of its lower execution channel."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from . import pointmaze_local_plan_gain as probe
from .pointmaze_root_response import write_json
from scripts import pointmaze_plan_authority_stage120_spec as spec


def variants():
    return [("forecast", "forecast", "advice", None)] + [
        (f"{channel}/zero", "zero", channel, None) for channel in spec.CHANNELS] + [
        (f"{channel}/{size}/{direction}", direction, channel, amplitude)
        for channel in spec.CHANNELS for size, amplitude in spec.AMPLITUDES.items() for direction in spec.DIRECTIONS]


def worker_query(job):
    weights, lower_state, query, period, predictor, calibration = job
    panels, common, largest = {}, None, 0.
    for panel in spec.PANELS:
        rows, reference = {}, None
        for name, variant, channel, amplitude in variants():
            action = probe.intervention_action(variant)
            if amplitude is not None:
                action *= amplitude / probe.spec.EPSILON
            row, audit = probe.episode(weights, lower_state, query=query, panel=panel, variant=variant,
                period=period, predictor=predictor, calibration=calibration, action_delta=action,
                execution=channel, reference_limit=spec.REFERENCE_LIMIT)
            if common is None:
                common = audit
            else:
                for key in ("query_state", "upper_state", "prefix_rewards", "measurements"):
                    np.testing.assert_array_equal(audit[key], common[key])
                np.testing.assert_array_equal(audit["commands"][:query["start"]], common["commands"][:query["start"]])
            if reference is None:
                reference = audit
            else:
                largest = max(largest, float(np.max(np.abs(audit["innovations"] - reference["innovations"]))))
                np.testing.assert_allclose(audit["innovations"], reference["innovations"], atol=3e-5, rtol=0)
                np.testing.assert_allclose(row["episode_return"] - rows["forecast"]["episode_return"],
                    row["suffix_return"] - rows["forecast"]["suffix_return"], atol=1e-9, rtol=0)
            if variant == "zero":
                for key in ("commands", "means", "rewards"):
                    np.testing.assert_array_equal(audit[key], reference[key])
            start, stop = query["start"], query["start"] + period
            tail_delta = audit["commands"][stop:] - reference["commands"][stop:]
            row.update(option_command_delta_rms=float(np.sqrt(np.square(audit["commands"][start:stop] - reference["commands"][start:stop]).mean())),
                tail_command_delta_rms=float(np.sqrt(np.square(tail_delta).mean())) if tail_delta.size else 0.)
            rows[name] = row
        panels[panel] = rows
    model, _ = probe.base.source.native._WORKER
    torch.testing.assert_close(probe.base.source.native.joint.inference_weights(model), weights, atol=0, rtol=0)
    return {"query": query, "panels": panels, "pairing": "passed", "source_freeze": "passed", "innovation_max_error": largest}


def crossfit_gain(panels, channel, size):
    candidates = (f"{channel}/zero", *(f"{channel}/{size}/{v}" for v in spec.DIRECTIONS))
    selected, gains = {}, []
    for fit, test in (("A", "B"), ("B", "A")):
        choice = max(candidates, key=lambda v: panels[fit][v]["suffix_return"])
        selected[fit] = choice
        gains.append(panels[test][choice]["suffix_return"] - panels[test][f"{channel}/zero"]["suffix_return"])
    return float(np.mean(gains)), selected


def effects(period, queries):
    values = {f"{channel}_{size}_gain": np.asarray([crossfit_gain(q["panels"], channel, size)[0] for q in queries])
        for channel in spec.CHANNELS for size in spec.AMPLITUDES}
    values["reference_large_minus_advice_large"] = values["reference_large_gain"] - values["advice_large_gain"]
    return {f"{period}/{key}": float(value.mean()) for key, value in values.items()}


def run(root, *, preflight, output):
    source = probe.base.source
    source.qualify(json.loads(spec.source_result(root).read_text()), preflight=False)
    models, predictor, _, calibrations = source.load_source(root)
    args, roles = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight)
    cost = dict.fromkeys(spec.budget(preflight=preflight), 0)
    cost.update(source_cell_loads=1, source_clone_loads=len(models), lower_checkpoint_loads=len(models))
    groups = {}
    with ProcessPoolExecutor(max_workers=spec.options(preflight=preflight)["workers"], mp_context=mp.get_context("spawn"),
            initializer=source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            before = copy.deepcopy(model.state_dict())
            lower_state = probe.base.load_lower_state(root, period, protocol=spec)
            jobs = [(source.native.joint.inference_weights(model), lower_state, query, period, predictor, calibrations[str(period)])
                for query in roles["queries"]]
            queries = list(pool.map(worker_query, jobs))
            for query in queries:
                cost["native_pair_groups"] += 1
                count = len(variants())
                for key in ("prefix_pair_checks", "external_pair_checks"):
                    cost[key] += len(spec.PANELS) * count - 1
                cost["innovation_pair_checks"] += len(spec.PANELS) * (count - 1)
                cost["zero_forecast_checks"] += len(spec.PANELS) * len(spec.CHANNELS)
                for rows in query["panels"].values():
                    for row in rows.values():
                        cost["native_episodes"] += 1
                        cost["native_steps"] += row["episode_length"]
                        cost["native_lower_calls"] += row["lower_calls"]
                        cost["suffix_credit_checks"] += 1
                        for key in ("reference_donor_calls", "plan_renewals", "plan_fits", "reference_evaluations", "actor_context_evaluations"):
                            cost[key] += row[key]
            source.native.curves.support.assert_frozen(model, before)
            cost["frozen_source_checks"] += 1
            groups[str(period)] = {"queries": queries, "effects": effects(period, queries)}
            print(f"Stage120 root{root} period{period}: bounded plan authority queries complete", flush=True)
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": cost, "groups": groups,
        "policy_updates": 0, "critic_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    root, h = cell["root"], spec.arguments(cell["root"], preflight=preflight).horizon
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or root not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight) or cell["cost"] != spec.budget(preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("policy_updates", "critic_fits", "checkpoint_writes", "native_trace_writes"))):
        raise ValueError("Stage120 frozen source, budget or query roster changed")
    for p, group in cell["groups"].items():
        period = int(p)
        if [q["query"] for q in group["queries"]] != cell["seed_roles"]["queries"]:
            raise ValueError("Stage120 query roster changed")
        for q in group["queries"]:
            if q["pairing"] != "passed" or q["source_freeze"] != "passed" or not 0 <= q["innovation_max_error"] <= 3e-5:
                raise ValueError("Stage120 prefix or lower noise pairing failed")
            for panel in spec.PANELS:
                rows = q["panels"][panel]
                if set(rows) != {v[0] for v in variants()}:
                    raise ValueError("Stage120 plan intervention roster changed")
                for name, variant, channel, amplitude in variants():
                    row = rows[name]
                    expected = {"variant": variant, "panel": panel, "execution": channel, "episode_length": h,
                        "start": q["query"]["start"], "scenario_seed": q["query"]["scenario_seed"],
                        "upper_actor_calls": 0, "lower_calls": h, "plan_renewals": h // period,
                        "plan_fits": h // period - 1, "reference_evaluations": h, "actor_context_evaluations": h,
                        "intervention_decisions": int(amplitude is not None),
                        "reference_donor_calls": 2 * h if channel == "reference" else 0}
                    if any(row[k] != value for k, value in expected.items()):
                        raise ValueError("Stage120 native schedule or execution channel changed")
                    for role, noise in (("prefix_seed", q["query"]["prefix_noise_seed"]), ("suffix_seed", q["query"]["suffix_noise_seeds"][panel])):
                        if row[role] != probe.base.source.scenario.spec.noise_seeds(root, row["scenario_seed"], noise)[1]:
                            raise ValueError("Stage120 prefix/suffix noise roles changed")
                    if not 0 <= row["reference_correction_peak"] <= spec.REFERENCE_LIMIT + 1e-8:
                        raise ValueError("Stage120 reference action correction exceeded its budget")
                    np.testing.assert_allclose(row["episode_return"], row["prefix_return"] + row["suffix_return"], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(row["suffix_return"], row["option_return"] + row["tail_return"], atol=1e-9, rtol=0)
                for channel in spec.CHANNELS:
                    if any(rows[f"{channel}/zero"][k] != rows["forecast"][k] for k in (
                            "episode_return", "suffix_return", "option_command_delta_rms", "tail_command_delta_rms")):
                        raise ValueError("Stage120 zero plan changed the strong forecast baseline")
        if group["effects"] != effects(period, group["queries"]) or not np.isfinite(list(group["effects"].values())).all():
            raise ValueError("Stage120 paired root effects changed")
    return cell


def aggregate(cells, *, preflight):
    result = probe.statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    decision = "mechanical_only"
    if not preflight:
        e = result["endpoints"]
        if all(e[f"{p}/reference_large_gain"]["ci"][0] > spec.MINIMUM_GAIN and
                e[f"{p}/reference_large_minus_advice_large"]["ci"][0] > 0 for p in spec.PERIODS):
            decision = "learn_bounded_reference_channel"
        elif all(e[f"{p}/advice_large_gain"]["ci"][0] > spec.MINIMUM_GAIN for p in spec.PERIODS):
            decision = "learn_larger_existing_plan_channel"
        elif all(e[f"{p}/{c}_{size}_gain"]["ci"][1] < spec.MINIMUM_GAIN
                for p in spec.PERIODS for c in spec.CHANNELS for size in spec.AMPLITUDES):
            decision = "stop_tested_fixed_lower_branch"
        else:
            decision = "inconclusive_no_automatic_seed_extension"
    result.update(plan_authority_decision=decision,
        performance_claim="bounded_one_option_conditional_headroom_not_deployable_policy_or_joint_HRL")
    return result
