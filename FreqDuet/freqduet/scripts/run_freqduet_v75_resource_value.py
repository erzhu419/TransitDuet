#!/usr/bin/env python3
"""Fixed resource-value models: independent-day expansion and held-day scoring."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import run_freqduet_v75_resource_labels as panel


previous, old, source = panel.previous, panel.old, panel.source
model, resources = previous.model, panel.resources
FAMILIES = ("action_only", "state_quadratic", "full_fleet", "structured")
BUDGETS = ("old12", "expanded44")
PRIMARY = "expanded44/structured"
CONTROLS = ("zero", "expanded44/action_only", "expanded44/state_quadratic", "expanded44/full_fleet")


def inputs(data, family):
    if family in model.KINDS:
        return data["states"]
    if family == "full_fleet":
        return previous.inputs(data, "fleet_plan")
    if family == "structured":
        return np.column_stack((data["states"], data["resources"]))
    raise ValueError("unknown fixed V75 representation")


def old_data(v70_root, v72_root, v73_root):
    data = old.read_data(v72_root)
    fleet, plans, structured = [], [], []
    for reference in range(len(source.REFERENCES)):
        bundle = torch.load(source.reference_path(v70_root, reference), map_location="cpu", weights_only=False)
        snapshots = json.loads((v73_root / "references" / f"r{reference}.json").read_text())
        for window, target in enumerate(source.TARGETS):
            snapshot = snapshots["captures"][window]["snapshot"]
            capture = bundle["captures"][target]
            f, p = previous.telemetry.encode_fleet(snapshot)
            fleet.append(f)
            plans.append(p)
            structured.append(resources.encode_resources(capture, snapshot))
    data.update(fleet_states=np.asarray(fleet), plan_states=np.asarray(plans), resources=np.asarray(structured))
    return data


def new_data(root, days):
    columns = {k: [] for k in ("states", "outcomes", "native", "days", "cases", "windows", "fleet_states", "plan_states", "resources")}
    for reference in range(len(panel.REFERENCES)):
        spec = panel.reference_spec(reference)
        if spec["scenario_seed"] not in days:
            continue
        bundle = torch.load(panel.reference_path(root, spec), map_location="cpu", weights_only=False)
        for window, target in enumerate(source.TARGETS):
            context = panel.CONTEXTS.index((reference, window))
            capture, snapshot = bundle["captures"][target], bundle["fleet"][target]
            groups = [json.loads(panel.group_path(root, context, f).read_text()) for f in source.FUTURES]
            for future, group in enumerate(groups):
                if (group["scenario_seed"], group["context_index"], group["future_index"], group["future_seed"], group["native_action_s"]) != (
                        spec["scenario_seed"], context, future, panel.future_seed(context, future), capture["actor_action_s"]):
                    raise ValueError("new action labels do not align with their frozen native context")
                if [r["action_s"] for r in group["rows"]] != list(model.ACTIONS) or not group["exact_prefix"] or not group["common_future_start"]:
                    raise ValueError("new labels lack the exact paired seven-action contract")
            f, p = previous.telemetry.encode_fleet(snapshot)
            values = dict(states=model.observable_features(capture),
                outcomes=[[[r[m] for m in old.OUTCOMES] for r in g["rows"]] for g in groups],
                native=model.ACTIONS.index(capture["actor_action_s"]), days=spec["scenario_seed"],
                cases=spec["case_index"], windows=target, fleet_states=f, plan_states=p,
                resources=resources.encode_resources(capture, snapshot))
            for key in columns:
                columns[key].append(values[key])
    return {k: np.asarray(v) for k, v in columns.items()}


def fit_candidate(data, family, training_days):
    training = np.isin(data["days"], training_days)
    if set(data["days"][training].tolist()) != set(training_days):
        raise ValueError("fitting panel is missing a registered training day")
    targets = model.mean_differences(data["outcomes"][training], data["native"][training])[:, :, old.TARGET_COLUMNS]
    kind = "action_only" if family == "action_only" else "state_quadratic"
    return model.fit_model(inputs(data, family)[training], data["native"][training], targets, data["days"][training], kind)


def save_model(path, fitted, family, budget):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, format="freqduet-v75-resource-value-development", kind=fitted.kind,
        family=family, budget=budget, center=fitted.center, scale=fitted.scale, coefficients=fitted.coefficients,
        actions=np.asarray(model.ACTIONS), metrics=np.asarray(model.METRICS), ridge=model.RIDGE,
        resource_fields=np.asarray(resources.FEATURES))


def smoke(root, v70_root, v72_root, v73_root):
    data = old_data(v70_root, v72_root, v73_root)
    for family in ("state_quadratic", "structured"):
        fitted = fit_candidate(data, family, source.DEMAND_SEEDS)
        predicted = fitted.predict(inputs(data, family), data["native"])
        if not np.isfinite(predicted).all():
            raise AssertionError("resource-value fitting generated non-finite predictions")
        np.testing.assert_array_equal(predicted[np.arange(len(predicted)), data["native"]], 0.)
        save_model(root / "smoke_details" / f"{family}.npz", fitted, family, "development_smoke")
    print("V75_VALUE_SMOKE " + json.dumps(dict(contexts=len(data["days"]), structured_inputs=inputs(data, "structured").shape[1])), flush=True)


def fit(root, v70_root, v72_root, v73_root):
    original = old_data(v70_root, v72_root, v73_root)
    added = new_data(root, panel.TRAIN_DAYS)
    data = {k: np.concatenate((original[k], added[k])) for k in original}
    expected = set(source.DEMAND_SEEDS) | set(panel.TRAIN_DAYS)
    if set(data["days"].tolist()) != expected or expected & set(panel.VALID_DAYS):
        raise ValueError("training/validation demand-day roles are not disjoint")
    np.savez_compressed(root / "prepared_train.npz", **data)
    info = {}
    for budget in BUDGETS:
        days = source.DEMAND_SEEDS if budget == "old12" else tuple(sorted(expected))
        for family in FAMILIES:
            fitted = fit_candidate(data, family, days)
            save_model(root / "models" / budget / f"{family}.npz", fitted, family, budget)
            info[budget + "/" + family] = dict(training_days=len(days), input_dimensions=len(fitted.center),
                parameters=int(fitted.coefficients.size), effective_parameters_per_output=fitted.effective_parameters_per_output)
            print("V75_FIT_DONE " + json.dumps(dict(budget=budget, family=family)), flush=True)
    previous.write_json(root / "fit_export" / "done.json", dict(status="models_frozen", models=info,
        training_days=sorted(expected), validation_days=panel.VALID_DAYS, validation_labels_read=False,
        diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")))


def gates(contrasts, effects):
    return dict(prediction_gate_passed=all(contrasts[c][m][loss]["ci95"][1] < 0
        for c in CONTROLS for m in old.PRIMARY for loss in ("mae", "mse")),
        joint_effect_gate_passed=effects["passenger_person_s"]["ci95"][1] < 0 and
        all(effects[m]["ci95"][1] <= 0 for m in model.METRICS[1:]))


def evaluate(root):
    from run_freqduet_v74_fit_gap import load_model

    frozen = json.loads((root / "fit_export" / "done.json").read_text())
    if frozen["validation_labels_read"] or set(frozen["training_days"]) & set(panel.VALID_DAYS):
        raise ValueError("value fits did not preserve the independent validation-day split")
    data = new_data(root, panel.VALID_DAYS)
    np.savez_compressed(root / "prepared_validation.npz", **data)
    predictions = {}
    for budget in BUDGETS:
        for family in FAMILIES:
            fitted = load_model(root / "models" / budget / f"{family}.npz")
            predictions[budget + "/" + family] = fitted.predict(inputs(data, family), data["native"])
    predictions["zero"] = np.zeros_like(predictions[PRIMARY])
    rows = old.score_predictions(data, np.arange(len(data["days"])), predictions)
    channels = tuple(predictions)
    errors = {c: {m: {loss: old.interval(rows, lambda r, c=c, m=m, loss=loss: r["channels"][c]["errors"][m][loss])
        for loss in ("mae", "mse")} for m in old.PRIMARY} for c in channels}
    effects = {c: {m: old.interval(rows, lambda r, c=c, m=m: r["channels"][c]["effects"][m]) for m in old.OUTCOMES} for c in channels}
    contrasts = {c: {m: {loss: old.interval(rows, lambda r, c=c, m=m, loss=loss:
        r["channels"][PRIMARY]["errors"][m][loss] - r["channels"][c]["errors"][m][loss])
        for loss in ("mae", "mse")} for m in old.PRIMARY} for c in CONTROLS}
    budget_effect = {family: {m: {loss: old.interval(rows, lambda r, family=family, m=m, loss=loss:
        r["channels"]["expanded44/" + family]["errors"][m][loss] - r["channels"]["old12/" + family]["errors"][m][loss])
        for loss in ("mae", "mse")} for m in old.PRIMARY} for family in FAMILIES}
    choices = {c: dict(changed=sum(r["channels"][c]["changed"] for r in rows),
        passenger_improved=sum(r["channels"][c]["effects"]["passenger_person_s"] < 0 for r in rows),
        passenger_worsened=sum(r["channels"][c]["effects"]["passenger_person_s"] > 0 for r in rows),
        joint_nonworse=sum(r["channels"][c]["effects"]["passenger_person_s"] < 0 and
            all(r["channels"][c]["effects"][m] <= 0 for m in model.METRICS[1:]) for r in rows)) for c in channels}
    previous.write_json(root / "validation_context_audit.json", rows)
    previous.write_json(root / "summary" / "status.json", dict(status="independent_resource_value_validation_complete",
        primary_candidate=PRIMARY, contexts=len(rows), validation_day_clusters=len(panel.VALID_DAYS),
        added_training_days=panel.TRAIN_DAYS, validation_days=panel.VALID_DAYS, futures_per_context=len(source.FUTURES),
        errors=errors, effects=effects, choices=choices, primary_prediction_contrasts=contrasts,
        expanded_minus_old_prediction=budget_effect, **gates(contrasts, effects[PRIMARY]),
        models=frozen["models"], resource_fields=resources.FEATURES,
        by_day=[dict(scenario_seed=day, **{c: {m: float(np.mean([r["channels"][c]["effects"][m]
            for r in rows if r["scenario_seed"] == day])) for m in model.METRICS} for c in channels}) for day in panel.VALID_DAYS],
        original_training_seeds=2, formal_action_rollouts=len(panel.ITEMS) * len(model.ACTIONS),
        actor_updates=0, dual_updates=0, production_value_updates=0, main_policy_changed=False,
        single_intervention_validation=True, controller_admitted=False, gates_changed=False,
        diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")))
    print("V75_VALIDATION " + json.dumps(gates(contrasts, effects[PRIMARY])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "fit", "evaluate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v70-root", type=Path)
    parser.add_argument("--v72-root", type=Path)
    parser.add_argument("--v73-root", type=Path)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.out_root)
    else:
        (smoke if args.command == "smoke" else fit)(args.out_root, args.v70_root, args.v72_root, args.v73_root)


if __name__ == "__main__":
    main()
