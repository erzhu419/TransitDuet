"""Whole-scene transfer of cached option credit and the resulting native policy."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from . import pointmaze_option_conditioning as previous
from .pointmaze_actor_credit import cosine
from .pointmaze_root_response import write_json
from scripts import pointmaze_option_crossfit_stage144_spec as spec

joint, warm, query = previous.joint, previous.warm, previous.query


def scene_split(rows, seed):
    return ([r for r in rows if r["row"]["seed"] != seed],
            [r for r in rows if r["row"]["seed"] == seed])


def held_out_geometry(trainer, uppers, held):
    initial = joint.weights(trainer)
    records, cost = [], 0
    for row in held:
        state = torch.as_tensor(row["batch"].state)
        with torch.no_grad():
            baseline = trainer.upper_actor.distribution(state).mean.double().numpy()
        cost += 1
        changes, predictions, rms = {}, {}, {}
        for variant in spec.VARIANTS:
            trainer.upper_actor.load_state_dict(uppers[variant])
            with torch.no_grad():
                change = trainer.upper_actor.distribution(state).mean.double().numpy()-baseline
            probe = variant.split("_")[0]
            predictions[variant] = float(np.sum(change*row["gradients"][probe]))
            rms[variant] = float(np.sqrt(np.mean(change**2)))
            changes[variant] = change
            cost += 1
        trainer.upper_actor.load_state_dict(initial["upper_actor"])
        records.append({"scenario_seed": row["row"]["seed"], "noise_seed": row["row"]["noise_seed"],
            "panel": row["panel"], "predicted_return_increment": predictions, "held_out_mean_step_RMS": rms,
            "raw_to_compact_mean_step_cosine": {p: cosine(changes[p+"_raw_plus"].ravel(),
                changes[p+"_compact_plus"].ravel()) for p in spec.PROBES}})
    torch.testing.assert_close(joint.weights(trainer), initial, atol=0, rtol=0)
    return records, cost


def worker_evaluate(job):
    source_weights, teacher, initial, uppers, warm_row, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    joint.load_weights(trainer, initial)
    rows, common = {"warm_start": warm_row}, None
    for variant in spec.VARIANTS:
        trainer.upper_actor.load_state_dict(uppers[variant])
        batch, row, audit = joint.native_episode(trainer, args=args, seed=warm_row["seed"],
            noise_seed=warm_row["noise_seed"], arm="joint", period=period, predictor=predictor,
            envelope=envelope, collect=False, sample_upper=False)
        assert batch is None
        for key in ("seed", "noise_seed", "decision_steps", "upper_sample", "lower_sample"):
            if row[key] != warm_row[key]:
                raise ValueError("Stage144 excluded-scene control pairing changed")
        if common is None: common = audit
        else:
            np.testing.assert_array_equal(audit["measurements"], common["measurements"])
            np.testing.assert_allclose(audit["innovations"], common["innovations"], atol=3e-5, rtol=0)
        torch.testing.assert_close(joint.weights(trainer), {**initial, "upper_actor": uppers[variant]}, atol=0, rtol=0)
        rows[variant] = row
    return rows


def evaluation_summary(rows):
    return {name: {a+"_minus_"+b: query.previous.paired_effect([
        sign*(r[a][metric]-r[b][metric]) for r in rows]) for a, b in spec.CONTRASTS}
        for name, metric, sign in (("effects", "episode_return", 1),
            ("tracking_error_reduction_positive_is_better", "tracking_squared_error_integral", -1))}


def run(root, output):
    conditioning, local, wide, warm_cached = [json.loads(p.read_text()) for p in (
        spec.source_result(root), spec.source.source_result(root),
        spec.source.source_result(root, wide=True), spec.source.warm_result(root))]
    for cell, protocol in ((conditioning, spec.source), (local, spec.source.source),
                           (wide, spec.source.source.source)):
        if (cell["status"], cell["protocol"], cell["root"], cell["cost"], cell["seed_roles"], cell["contract"]) != (
                "complete", protocol.PROTOCOL, root, protocol.budget(), protocol.seed_roles(root), protocol.contract()):
            raise ValueError("Stage144 completed cached credit source changed")
    warm_spec = spec.source.source.source.source.warm_source
    if (warm_cached["status"], warm_cached["protocol"], warm_cached["root"], warm_cached["cost"]) != (
            "complete", warm_spec.PROTOCOL, root, warm_spec.budget()):
        raise ValueError("Stage144 warm source differs from Stage135")
    models, predictor, _, calibrations = joint.source.load_source(root)
    args, roles = spec.arguments(root), spec.seed_roles(root)
    cost, groups = dict.fromkeys(spec.budget(), 0), {}
    cost.update(source_cell_loads=4, lower_checkpoint_loads=len(spec.PERIODS), upper_checkpoint_loads=len(spec.PERIODS))
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            snapshot = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=joint.spec)
            trainer = joint.make_trainer(model, teacher, args)
            upper, provenance = warm.load_selected_upper(warm_cached, root, period)
            trainer.upper_actor.load_state_dict(upper)
            initial = query.previous.scale_initial(joint.weights(trainer), "reduced")
            joint.load_weights(trainer, initial)
            source_weights, envelope = joint.weights(model), calibrations[str(period)]["envelope"]
            labels = [dict(zip(spec.PROBES, paths)) for paths in zip(
                wide["groups"][str(period)]["mean_query_paths"], local["groups"][str(period)]["local_query_paths"])]
            jobs = [(source_weights, teacher, initial, r, p, period, predictor, envelope, label)
                for (r, p), label in zip(((r, p) for r in roles["replayed_training"] for p in spec.PANELS), labels)]
            rows = list(pool.map(previous.worker_replay, jobs))
            for row in rows:
                warm.count_row(cost, row["row"], "replay"); cost["credit_checks"] += 1
            folds, evaluation = [], []
            for seed in roles["held_out_scene_order"]:
                training, held = scene_split(rows, seed)
                assert len(held) == len(spec.PANELS) and len(training) == len(rows)-len(held)
                assert {r["panel"] for r in held} == set(spec.PANELS)
                uppers, learning, similarities, work = previous.fit_candidates(trainer, training)
                for k, v in work.items(): cost[k] += v
                geometry, forwards = held_out_geometry(trainer, uppers, held)
                cost["held_out_geometry_forward_batches"] += forwards
                outputs = list(pool.map(worker_evaluate, [(source_weights, teacher, initial, uppers,
                    r["row"], period, predictor, envelope) for r in held]))
                for output_rows in outputs:
                    for variant in spec.VARIANTS: warm.count_row(cost, output_rows[variant], "evaluation")
                evaluation.extend(outputs)
                folds.append({"held_out_scenario_seed": seed,
                    "training_scenario_seeds": [s for s in roles["held_out_scene_order"] if s != seed],
                    "excluded_noise_panels": list(spec.PANELS), "learning": learning,
                    "training_raw_to_compact_mean_step_cosine": similarities,
                    "held_out_geometry": geometry, "native_control": evaluation_summary(outputs)})
            joint.source.native.curves.support.assert_frozen(model, snapshot)
            groups[str(period)] = {"selected_source": provenance, "folds": folds,
                "both_probe_label_replays": "passed", "frozen_deployment_and_noise_pairing": "passed",
                "whole_scene_exclusion": "passed", "held_out_control": evaluation_summary(evaluation)}
            print(f"root={root} period={period}: all {len(folds)} excluded-scene native folds complete", flush=True)
    if cost != spec.budget(): raise ValueError(f"Stage144 measured crossfit budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "contract": spec.contract(),
        "seed_roles": roles, "cost": cost, "groups": groups,
        "kind": "whole_scene_cached_credit_crossfit_not_confirmation_or_joint_HRL",
        "inherited_Stage143_cost": conditioning["cost"],
        "inherited_source_cost": {k: v for k, v in conditioning.items() if k.startswith("inherited_")}}
    write_json(output, result)
    write_json(output.parent/"completion"/"ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    return result
