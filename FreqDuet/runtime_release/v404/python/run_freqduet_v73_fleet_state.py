#!/usr/bin/env python3
"""Causal fleet/commitment sufficiency probe on unchanged V70 action labels."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import freqduet_fleet_observation as telemetry
import run_freqduet_v72_action_increment as old


source, model = old.source, old.model
FAMILIES = ("fleet_only", "plan_only", "fleet_plan")
PRIMARY = "fleet_plan"
CONTROLS = old.CHANNELS
SMOKE_REFERENCES = (0, 36)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class FleetObserver(source.labels.deployment.CreditObserver):
    def __init__(self):
        super().__init__()
        self.snapshots = {}

    def attach(self, runner):
        super().attach(runner)
        original = runner._lower_policy_action

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            target = runner._label_capture
            if target is not None and target not in self.snapshots:
                # The proposed action was drawn, but not appended or executed.
                self.snapshots[target] = telemetry.capture_fleet(runner.env)
            return result

        runner._lower_policy_action = observe


def collect_reference(root, v59_root, v70_root, index):
    path = root / "references" / f"r{index}.json"
    if path.exists():
        return index
    spec = source.reference_spec(index)
    observer = FleetObserver()
    result = source.labels.run_spec(root, v59_root, spec, source.TARGETS,
                                    spec["label"] + "_fleet_capture", observer=observer)
    base = torch.load(source.reference_path(v70_root, index), map_location="cpu", weights_only=False)
    captures = []
    for target in source.TARGETS:
        current = result["captures"][target]
        source.labels.same_prefix(base["captures"][target], current)
        snapshot = observer.snapshots[target]
        if snapshot["time_s"] != current["identity"][0]:
            raise AssertionError("fleet observation is not from the exact pre-action tick")
        fleet, plans = telemetry.encode_fleet(snapshot)
        captures.append(dict(target_time_s=target, identity=current["identity"], snapshot=snapshot,
                             fleet_dimensions=len(fleet), plan_dimensions=len(plans)))
    for key in ("summary", "holds", "upper_path"):
        source.joint.assert_same_state(base[key], result[key], "fleet_capture_native_" + key)
    write_json(path, dict(reference_index=index, spec=spec, captures=captures, exact_prefixes=True,
                          exact_day=True, exact_lower_path=True, exact_upper_path=True,
                          diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")))
    print("V73_REFERENCE_DONE " + json.dumps(dict(reference_index=index, prefixes=len(captures))), flush=True)
    return index


def baseline_identity(v72_root):
    data = old.read_data(v72_root)
    day = source.DEMAND_SEEDS[0]
    targets = model.mean_differences(data["outcomes"], data["native"])[:, :, old.TARGET_COLUMNS]
    fitted = model.fit_excluding_day(data["states"], data["native"], targets, data["days"], day, "state_quadratic")
    with np.load(v72_root / "folds" / str(day) / "state_quadratic.npz") as cached:
        for name in ("center", "scale", "coefficients"):
            np.testing.assert_array_equal(getattr(fitted, name), cached[name])
    testing = np.flatnonzero(data["days"] == day)
    rows = old.score_predictions(data, testing, dict(state_quadratic=fitted.predict(data["states"][testing], data["native"][testing])))
    cached = json.loads((v72_root / "folds" / str(day) / "done.json").read_text())
    for row, previous in zip(rows, cached["rows"]):
        source.joint.assert_same_state(row["channels"]["state_quadratic"],
                                       previous["channels"]["state_quadratic"], "V72_cached_scoring_identity")
    return dict(held_day=day, coefficients_exact=True, scored_contexts=len(rows), scoring_exact=True)


def prepare(root, v72_root):
    data = old.read_data(v72_root)
    references = [json.loads((root / "references" / f"r{i}.json").read_text()) for i in range(len(source.REFERENCES))]
    fleet, plans = [], []
    for reference, window in source.CONTEXTS:
        record = references[reference]
        if not all(record[k] for k in ("exact_prefixes", "exact_day", "exact_lower_path", "exact_upper_path")):
            raise ValueError("cannot reuse action labels after a changed native trajectory")
        capture = record["captures"][window]
        if record["spec"] != source.reference_spec(reference) or capture["target_time_s"] != source.TARGETS[window]:
            raise ValueError("fleet capture does not align with the original V70 panel")
        f, p = telemetry.encode_fleet(capture["snapshot"])
        fleet.append(f)
        plans.append(p)
    data.update(fleet_states=np.asarray(fleet), plan_states=np.asarray(plans))
    np.savez_compressed(root / "prepared.npz", **data)
    return data


def inputs(data, family):
    parts = [data["states"]]
    if family in ("fleet_only", "fleet_plan"):
        parts.append(data["fleet_states"])
    if family in ("plan_only", "fleet_plan"):
        parts.append(data["plan_states"])
    if family not in FAMILIES:
        raise ValueError("unknown registered observation family")
    return np.column_stack(parts)


def fit_fold(root, held_day):
    destination = root / "folds" / str(held_day)
    if (destination / "done.json").exists():
        return held_day
    data = old.read_data(root)
    targets = model.mean_differences(data["outcomes"], data["native"])[:, :, old.TARGET_COLUMNS]
    testing = np.flatnonzero(data["days"] == held_day)
    predictions, parameters, effective = {}, {}, {}
    for family in FAMILIES:
        states = inputs(data, family)
        fitted = model.fit_excluding_day(states, data["native"], targets, data["days"], held_day, "state_quadratic")
        predictions[family] = fitted.predict(states[testing], data["native"][testing])
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / f"{family}.npz", format="freqduet-v73-fleet-state-development",
            family=family, kind=fitted.kind, center=fitted.center, scale=fitted.scale, coefficients=fitted.coefficients,
            actions=np.asarray(model.ACTIONS), metrics=np.asarray(model.METRICS), ridge=model.RIDGE,
            fleet_fields=np.asarray(telemetry.FLEET_FIELDS), plan_fields=np.asarray(telemetry.PLAN_FIELDS))
        parameters[family], effective[family] = int(fitted.coefficients.size), fitted.effective_parameters_per_output
    write_json(destination / "done.json", dict(held_day=int(held_day),
        fitting_days=sorted(set(data["days"].tolist()) - {int(held_day)}),
        rows=old.score_predictions(data, testing, predictions), parameters=parameters,
        effective_parameters_per_output=effective, diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")))
    print("V73_FOLD_DONE " + json.dumps(dict(held_day=int(held_day), contexts=len(testing))), flush=True)
    return held_day


def gates(contrasts, effects):
    return dict(prediction_gate_passed=all(contrasts[c][m][loss]["ci95"][1] < 0
                for c in CONTROLS for m in old.PRIMARY for loss in ("mae", "mse")),
                joint_effect_gate_passed=effects["passenger_person_s"]["ci95"][1] < 0
                and all(effects[m]["ci95"][1] <= 0 for m in model.METRICS[1:]))


def aggregate(root, v72_root):
    folds = [json.loads((root / "folds" / str(day) / "done.json").read_text()) for day in source.DEMAND_SEEDS]
    rows = []
    for fold in folds:
        if fold["fitting_days"] != sorted(set(source.DEMAND_SEEDS) - {fold["held_day"]}):
            raise ValueError("held day entered extended-state fitting")
        cached = json.loads((v72_root / "folds" / str(fold["held_day"]) / "done.json").read_text())
        if len(fold["rows"]) != len(cached["rows"]):
            raise ValueError("extended and original state folds have different panels")
        for row, previous in zip(fold["rows"], cached["rows"]):
            for key in ("context_index", "scenario_seed", "case_index", "target_time_s", "native_action_s"):
                if row[key] != previous[key]:
                    raise ValueError("extended and original state predictions are not paired")
            row["channels"].update(previous["channels"])
            rows.append(row)
    if sorted(r["context_index"] for r in rows) != list(range(len(source.CONTEXTS))):
        raise ValueError("incomplete V73 whole-day-excluded panel")
    channels = (*CONTROLS, *FAMILIES)
    errors, effects, choices = {}, {}, {}
    for channel in channels:
        errors[channel] = {m: {loss: old.interval(rows, lambda r, m=m, loss=loss:
            r["channels"][channel]["errors"][m][loss]) for loss in ("mae", "mse")} for m in old.PRIMARY}
        for m in old.PRIMARY:
            errors[channel][m]["rmse"] = float(np.sqrt(errors[channel][m]["mse"]["mean"]))
        effects[channel] = {m: old.interval(rows, lambda r, m=m: r["channels"][channel]["effects"][m]) for m in old.OUTCOMES}
        choices[channel] = dict(changed=sum(r["channels"][channel]["changed"] for r in rows),
            passenger_improved=sum(r["channels"][channel]["effects"]["passenger_person_s"] < 0 for r in rows),
            passenger_worsened=sum(r["channels"][channel]["effects"]["passenger_person_s"] > 0 for r in rows),
            joint_nonworse=sum(r["channels"][channel]["effects"]["passenger_person_s"] < 0 and
                all(r["channels"][channel]["effects"][m] <= 0 for m in model.METRICS[1:]) for r in rows))
    contrasts = {control: {m: {loss: old.interval(rows, lambda r, m=m, loss=loss, control=control:
        r["channels"][PRIMARY]["errors"][m][loss] - r["channels"][control]["errors"][m][loss])
        for loss in ("mae", "mse")} for m in old.PRIMARY} for control in CONTROLS}
    data = old.read_data(root)
    result = dict(status="causal_fleet_state_day_exclusion_complete", contexts=len(rows),
        native_reference_replays=len(source.REFERENCES), exact_native_prefixes=len(rows), demand_day_clusters=len(folds),
        futures_per_context=len(source.FUTURES), primary_candidate=PRIMARY, ridge=model.RIDGE,
        errors=errors, effects=effects, choices=choices, prediction_contrasts=contrasts,
        **gates(contrasts, effects[PRIMARY]),
        by_day=[dict(scenario_seed=day, **{c: {m: float(np.mean([r["channels"][c]["effects"][m]
            for r in rows if r["scenario_seed"] == day])) for m in model.METRICS} for c in channels}) for day in source.DEMAND_SEEDS],
        effects_without_evaluation_day700006={m: old.interval([r for r in rows if r["scenario_seed"] != 700006],
            lambda r, m=m: r["channels"][PRIMARY]["effects"][m]) for m in model.METRICS},
        input_dimensions={c: inputs(data, c).shape[1] for c in FAMILIES}, parameters=folds[0]["parameters"],
        effective_parameters_per_output=[f["effective_parameters_per_output"] for f in folds],
        fleet_fields=telemetry.FLEET_FIELDS, plan_fields=telemetry.PLAN_FIELDS,
        action_rollouts_added=0, actor_updates=0, dual_updates=0, production_value_updates=0,
        main_policy_changed=False, development_day_crossfit_only=True, single_intervention_labels=True, controller_admitted=False)
    write_json(root / "summary" / "status.json", result)
    print("V73_AGGREGATE " + json.dumps(dict(contexts=len(rows), **gates(contrasts, effects[PRIMARY]))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "references", "folds", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v70-root", type=Path)
    parser.add_argument("--v72-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.command == "aggregate":
        aggregate(args.out_root, args.v72_root)
        return
    if args.command == "smoke":
        for pattern in ("test_v72_*.py", "test_v73_*.py"):
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                str(Path(__file__).resolve().parents[1] / "tests"), "-p", pattern, "-v"], check=True)
        identity = baseline_identity(args.v72_root)
        indices = SMOKE_REFERENCES
        export = "smoke_export"
    elif args.command == "references":
        indices, export = range(len(source.REFERENCES)), "reference_exports"
    else:
        prepare(args.out_root, args.v72_root)
        fit_fold(args.out_root, source.DEMAND_SEEDS[0])
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            list(pool.map(fit_fold, itertools.repeat(args.out_root), source.DEMAND_SEEDS))
        write_json(args.out_root / "fit_exports" / "done.json", dict(status="folds_complete", folds=len(source.DEMAND_SEEDS)))
        return
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        list(pool.map(collect_reference, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                      itertools.repeat(args.v70_root), indices))
    write_json(args.out_root / export / "done.json", dict(status=args.command + "_complete", references=len(indices),
        **(dict(v72_identity=identity) if args.command == "smoke" else {})))


if __name__ == "__main__":
    main()
