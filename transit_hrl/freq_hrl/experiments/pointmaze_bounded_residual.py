"""Constrain the actual historical nonlinear command response of a curve."""

import numpy as np
from . import pointmaze_calibrated_residual as previous
from scripts import pointmaze_bounded_residual_stage78_spec as spec


def contract_response(alpha, target_rms, evaluate, initial_response):
    trace = [{"alpha": alpha, "response": initial_response}]
    while trace[-1]["response"]["command_change_rms"] > target_rms:
        alpha *= .5
        trace.append({"alpha": alpha, "response": evaluate(alpha)})
    return trace


def calibrate(root, period, *, model, predictor, args, roles, bounds, old, cost, preflight):
    prior = old["calibration"]
    label_args = spec.arguments(root, preflight=False)
    cache, target, base_command, responses = previous.prepare_calibration(root, period,
        model=model, predictor=predictor, args=label_args, roles=roles, bounds=bounds,
        saved_bc_mse=prior["saved_bc_mse"], cost=cost, preflight=False)
    ratio_alpha = previous.calibration_alpha(responses["zero"]["bc_command_mse"], responses["original"]["command_change_rms"])
    np.testing.assert_allclose(ratio_alpha, prior["alpha"], atol=0, rtol=1e-7)
    for mode in ("zero", "original"):
        for k, value in responses[mode].items():
            np.testing.assert_allclose(value, prior["responses"][mode][k], atol=1e-9, rtol=1e-7)

    def evaluate(alpha):
        cost["constraint_response_evaluations"] += 1
        command = previous.evaluate_curve(model.lower_actor, cache, alpha, cost)
        return previous.response(command, base_command, target)

    responses["ratio"] = evaluate(ratio_alpha)
    for k, value in responses["ratio"].items():
        np.testing.assert_allclose(value, prior["responses"]["calibrated"][k], atol=1e-9, rtol=1e-7)
    cost["ratio_reproductions"] += 1
    target_rms = float(np.sqrt(responses["zero"]["bc_command_mse"]))
    trace = contract_response(ratio_alpha, target_rms, evaluate, responses["ratio"])
    responses["bounded"] = trace[-1]["response"]
    result = {"alpha": trace[-1]["alpha"], "ratio_alpha": ratio_alpha, "target_rms": target_rms,
        "responses": responses, "solver_trace": trace, "saved_bc_mse": prior["saved_bc_mse"],
        "bounded_to_bc_rmse_ratio": responses["bounded"]["command_change_rms"] / target_rms,
        "data_role": "historical_BC_labels_only", "bc_mse_reproduction": "passed", "label_plan_checks": "passed",
        "ratio_reproduction": "passed", "constraint": "passed", "envelope": prior["envelope"]}
    check_calibration(result)
    return result


def check_response_constraint(c):
    trace = c["solver_trace"]
    if (c["constraint"] != "passed"
            or c["ratio_alpha"] != previous.calibration_alpha(c["responses"]["zero"]["bc_command_mse"], c["responses"]["original"]["command_change_rms"])
            or c["target_rms"] != float(np.sqrt(c["responses"]["zero"]["bc_command_mse"]))
            or trace[0] != {"alpha": c["ratio_alpha"], "response": c["responses"]["ratio"]}
            or trace[-1] != {"alpha": c["alpha"], "response": c["responses"]["bounded"]}
            or not c["responses"]["bounded"]["command_change_rms"] <= c["target_rms"]):
        raise ValueError("Stage78 historical response constraint changed")
    for i, step in enumerate(trace[:-1]):
        if step["response"]["command_change_rms"] <= c["target_rms"] or trace[i+1]["alpha"] != step["alpha"] * .5:
            raise ValueError("Stage78 solver is not first-feasible deterministic contraction")


def check_calibration(c):
    check_response_constraint(c)
    if c["ratio_reproduction"] != "passed":
        raise ValueError("Stage78 historical ratio reproduction failed")
