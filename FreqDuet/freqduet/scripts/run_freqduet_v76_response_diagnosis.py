#!/usr/bin/env python3
"""Action-shape projection and exact source attribution of the frozen V75 fit."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_freqduet_v74_fit_gap import load_model
import run_freqduet_v75_resource_value as value


model, old, previous = value.model, value.old, value.previous
FAMILIES = ("state_quadratic", "structured")


def action_design(native):
    actions = np.asarray(model.ACTIONS, dtype=float) / max(model.ACTIONS)
    return np.column_stack((actions - actions[native], actions ** 2 - actions[native] ** 2))


def project_response(truth, native):
    x = action_design(native)
    valid = np.arange(len(x)) != native
    return x @ np.linalg.lstsq(x[valid], truth[valid], rcond=None)[0]


def decompose(prediction, truth, native):
    projected = project_response(truth, native)
    valid = np.arange(len(truth)) != native
    shape, mapping = projected[valid] - truth[valid], prediction[valid] - projected[valid]
    total = np.square(prediction[valid] - truth[valid]).mean(axis=0)
    shape_mse, mapping_mse = np.square(shape).mean(axis=0), np.square(mapping).mean(axis=0)
    np.testing.assert_allclose(total, shape_mse + mapping_mse, rtol=1e-9, atol=1e-12)
    return projected, total, shape_mse, mapping_mse


def source_coefficients(states, native, outcomes, days, fitted, context_groups):
    weights = model.day_weights(days)
    design = model.design(states, native, fitted.center, fitted.scale, True)
    valid = np.arange(len(model.ACTIONS))[None, :] != native[:, None]
    row_weights = np.broadcast_to(weights[:, None] / 6., valid.shape)[valid]
    x, y = design[valid], outcomes[valid]
    gram = x.T @ (row_weights[:, None] * x) + model.RIDGE * np.eye(x.shape[1])
    rhs = np.column_stack([x.T @ (row_weights * y * np.broadcast_to(group[:, None], valid.shape)[valid])
                           for group in context_groups])
    return np.linalg.solve(gram, rhs)


def shape_rows(data, predictions):
    truth = model.mean_differences(data["outcomes"], data["native"])
    targets = truth[:, :, old.TARGET_COLUMNS]
    rows = []
    for i, prediction in enumerate(predictions):
        native = int(data["native"][i])
        projected, total, shape, mapping = decompose(prediction, targets[i], native)
        selected = model.choose_action(prediction, native)
        ideal = model.choose_action(projected, native)
        full = model.choose_action(targets[i], native)
        rows.append(dict(context_index=i, scenario_seed=int(data["days"][i]), case_index=int(data["cases"][i]),
            target_time_s=int(data["windows"][i]), native_action_s=model.ACTIONS[native], action_s=model.ACTIONS[selected],
            projected_action_s=model.ACTIONS[ideal], true_mean32_action_s=model.ACTIONS[full],
            changed=selected != native, projected_changed=ideal != native,
            errors={m: dict(total_mse=float(total[j]), shape_mse=float(shape[j]), mapping_mse=float(mapping[j]))
                    for j, m in enumerate(old.PRIMARY)},
            selected={m: dict(predicted=float(prediction[selected, j]), projected=float(projected[selected, j]),
                             actual=float(targets[i, selected, j])) for j, m in enumerate(model.METRICS)},
            projected_effects={m: float(truth[i, ideal, j]) for j, m in enumerate(old.OUTCOMES)},
            true_mean32_effects={m: float(truth[i, full, j]) for j, m in enumerate(old.OUTCOMES)}))
    return rows


def shape_summary(rows):
    errors = {m: {component: old.interval(rows, lambda r, m=m, component=component: r["errors"][m][component])
                  for component in ("total_mse", "shape_mse", "mapping_mse")} for m in old.PRIMARY}
    for m in old.PRIMARY:
        errors[m]["shape_fraction_of_mse"] = errors[m]["shape_mse"]["mean"] / errors[m]["total_mse"]["mean"] if errors[m]["total_mse"]["mean"] else 0.
    selected = {m: {component: old.interval(rows, lambda r, m=m, component=component: r["selected"][m][component])
                    for component in ("predicted", "projected", "actual")} for m in model.METRICS}
    return dict(contexts=len(rows), errors=errors, selected=selected,
        projected_effects={m: old.interval(rows, lambda r, m=m: r["projected_effects"][m]) for m in model.METRICS},
        true_mean32_effects={m: old.interval(rows, lambda r, m=m: r["true_mean32_effects"][m]) for m in model.METRICS},
        projected_choices=dict(changed=sum(r["projected_changed"] for r in rows),
            harmful=sum(r["projected_effects"]["passenger_person_s"] > 0 for r in rows),
            mean32_choice_disagreements=sum(r["projected_action_s"] != r["true_mean32_action_s"] for r in rows),
            constraint_violations={m: sum(r["projected_effects"][m] > 0 for r in rows) for m in model.METRICS[1:]}),
        actual_harm_predicted_gain=sum(r["changed"] and r["selected"]["passenger_person_s"]["actual"] > 0
                                      and r["selected"]["passenger_person_s"]["predicted"] < 0 for r in rows),
        actual_harm_projected_gain=sum(r["changed"] and r["selected"]["passenger_person_s"]["actual"] > 0
                                      and r["selected"]["passenger_person_s"]["projected"] < 0 for r in rows))


def influence(data, fitted, evaluations):
    states = value.inputs(data, "structured")
    contrasts = model.mean_differences(data["outcomes"], data["native"])
    passenger = contrasts[:, :, old.OUTCOMES.index("passenger_person_s")]
    valid = np.arange(7)[None, :] != data["native"][:, None]
    energy = np.square(passenger[valid].reshape(len(passenger), 6)).mean(axis=1)
    days = sorted(set(data["days"].tolist()))
    order = np.argsort(-energy, kind="stable")
    groups = [data["days"] == day for day in days]
    tail_info = []
    for fraction in (.01, .05, .1):
        count = max(1, int(np.ceil(fraction * len(data["days"]))))
        group = np.zeros(len(data["days"]), dtype=bool)
        group[order[:count]] = True
        groups.append(group)
        tail_info.append(dict(context_fraction=fraction, contexts=count,
            label_squared_mass_fraction=float(energy[group].sum() / energy.sum()) if energy.sum() else 0.))
    coefficients = source_coefficients(states, data["native"], passenger, data["days"], fitted, groups)
    np.testing.assert_allclose(coefficients[:, :len(days)].sum(axis=1), fitted.coefficients[:, 0], rtol=1e-8, atol=1e-7)
    result = {}
    for split, (current, predictions) in evaluations.items():
        selected = np.asarray([model.choose_action(p, int(n)) for p, n in zip(predictions, current["native"])])
        design = model.design(value.inputs(current, "structured"), current["native"], fitted.center, fitted.scale, True)
        current_x = design[np.arange(len(selected)), selected]
        contributions = current_x @ coefficients
        expected = predictions[np.arange(len(selected)), selected, 0]
        np.testing.assert_allclose(contributions[:, :len(days)].sum(axis=1), expected, rtol=1e-8, atol=1e-7)
        rows = [dict(scenario_seed=int(day), context_index=i) for i, day in enumerate(current["days"])]
        day_contributions = [dict(training_day=day, selected_prediction_contribution=old.interval(rows,
            lambda r, j=j: contributions[r["context_index"], j])) for j, day in enumerate(days)]
        result[split] = dict(by_training_day=day_contributions,
            tail=[dict(info, selected_prediction_contribution=old.interval(rows,
                lambda r, j=j: contributions[r["context_index"], len(days) + j])) for j, info in enumerate(tail_info)])
    label_rows = []
    for i in range(len(passenger)):
        native_action = model.ACTIONS[int(data["native"][i])]
        for j, action in enumerate(model.ACTIONS):
            if action != native_action:
                label_rows.append(dict(direction="longer" if action > native_action else "shorter", value=float(passenger[i, j])))
    distribution = {}
    for direction in ("longer", "shorter"):
        y = np.asarray([r["value"] for r in label_rows if r["direction"] == direction])
        distribution[direction] = dict(action_contrasts=len(y), mean=float(y.mean()), median=float(np.median(y)),
            passenger_gain_fraction=float((y < 0).mean()), quantiles=np.quantile(y, [.01, .05, .5, .95, .99]).tolist())
    return dict(exact_frozen_prediction_reconstruction=True, by_split=result, training_label_distribution=distribution)


def run(root, out):
    training = dict(np.load(root / "prepared_train.npz"))
    validation = dict(np.load(root / "prepared_validation.npz"))
    cached = json.loads((root / "validation_context_audit.json").read_text())
    summaries, detailed, primary_evaluations = {}, {}, {}
    primary_fit = None
    for family in FAMILIES:
        fitted = load_model(root / "models" / "expanded44" / f"{family}.npz")
        summaries[family], detailed[family] = {}, {}
        for split, data in (("in_sample", training), ("consumed_validation", validation)):
            predictions = fitted.predict(value.inputs(data, family), data["native"])
            if split == "consumed_validation":
                channel = "expanded44/" + family
                reproduced = old.score_predictions(data, np.arange(len(data["days"])), {channel: predictions})
                for row, expected in zip(reproduced, cached):
                    value.source.joint.assert_same_state(row["channels"][channel], expected["channels"][channel], "V75_frozen_score_identity")
            rows = shape_rows(data, predictions)
            detailed[family][split], summaries[family][split] = rows, shape_summary(rows)
            if family == "structured":
                primary_fit = fitted
                primary_evaluations[split] = (data, predictions)
        print("V76_SHAPE_DONE " + family, flush=True)
    attribution = influence(training, primary_fit, primary_evaluations)
    previous.write_json(out / "response_contexts.json", detailed)
    previous.write_json(out / "summary" / "status.json", dict(status="frozen_response_shape_and_source_diagnosis_complete",
        response=summaries, influence=attribution, cached_validation_predictions_verified=True,
        candidate_fits=0, rollouts=0, main_policy_changed=False, gates_changed=False,
        label_projection_is_diagnostic_only=True, projected_choices_are_label_selected=True,
        source_attribution_is_not_leave_day_out_refitting=True))
    print("V76_DIAGNOSIS_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v75-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v76_*.py", "-v"], check=True)
    run(args.v75_root, args.out_root)


if __name__ == "__main__":
    main()
