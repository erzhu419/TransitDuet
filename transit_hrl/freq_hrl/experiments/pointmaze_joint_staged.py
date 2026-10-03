"""Compare final joint and staged policies without further learning."""

import json

import torch

from . import pointmaze_joint_conditioned as joint
from . import pointmaze_fresh_staged_upper as upper
from . import pointmaze_fresh_lower as lower
from . import pointmaze_fresh_joint as fresh
from . import pointmaze_actor_swap as swaps
from scripts import pointmaze_joint_staged_stage103_spec as spec

learning = joint.learning


def load_donors(records, original, root, period, cost):
    donors, checkpoints = {}, {}
    for method, owner in spec.DONORS.items():
        path = spec.checkpoint_path(root, period, method)
        record = records[owner]["groups"][str(period)]["trained"][method]
        if (record["evaluation_update"] != 8 or record["final_freeze_check"] != "passed"
                or record["checkpoint"] != str(path)):
            raise ValueError("Joint/staged donor is not the registered final checkpoint")
        reference = original
        if owner == "upper":
            reference = {**original, "lower_actor": donors[spec.upper.LOWER_FOR_METHOD[method]]["lower_actor"]}
        donors[method] = swaps.check_checkpoint(torch.load(path, map_location="cpu", weights_only=False), reference,
            root=root, period=period, method=method, protocol=spec, training_protocol=spec.SOURCE_PROTOCOLS[owner])
        checkpoints[method] = str(path)
        cost["checkpoint_loads"] += 1
        cost["checkpoint_freeze_checks"] += 1
    return donors, {"checkpoints": checkpoints, "checkpoint_freeze": "passed"}


def qualify(cell, *, preflight):
    learning.qualify(cell, preflight=preflight, protocol=spec)
    if cell["source_initialization"] != spec.source_record(cell["root"]):
        raise ValueError("Joint/staged source teacher or decoder changed")
    for p, g in cell["groups"].items():
        expected = {m: str(spec.checkpoint_path(cell["root"], int(p), m)) for m in spec.DONORS}
        if (g["checkpoints"] != expected or g["checkpoint_freeze"] != "passed" or g["composition"] != "passed"
                or g["matched_training_budget"] != spec.method_path_budgets(cell["root"], int(p))):
            raise ValueError("Joint/staged donor composition or training budget changed")
        for rows in g["evaluation"].values():
            for row in rows:
                expected_noise = joint.scenario.spec.noise_seeds(cell["root"], row["seed"], row["seed"])
                if ((row["noise_seed"], row["policy_seed"], row["lower_seed"]) != (row["seed"], *expected_noise)
                        or "upper_noise_seed" in row or row["upper_replay_forward_calls"]):
                    raise ValueError("Joint/staged evaluation changed the independent paired noise path")
    return cell


def run(root, *, preflight, output):
    records = {owner: json.loads(spec.donor_result(root, owner).read_text()) for owner in spec.SOURCE_PROTOCOLS}
    for owner, experiment in (("joint", joint), ("upper", upper), ("lower", lower)):
        if records[owner]["root"] != root:
            raise ValueError("Joint/staged donor root changed")
        experiment.qualify(records[owner], preflight=False)
    clones, predictor, initialization, calibrations = fresh.load_source(root)
    donors = {}

    def initialize(period, models, cost):
        if models:
            raise ValueError("Joint/staged comparison must not create training models")
        weights = learning.native.joint.inference_weights(clones[str(period)])
        donors[period], metadata = load_donors(records, weights, root, period, cost)
        matched = spec.method_path_budgets(root, period)
        cost["matched_training_budget_checks"] += 1
        return {**metadata, "matched_training_budget": matched}

    def compose(period, weights, cost):
        result = swaps.compose_weights(weights["base"], donors[period], protocol=spec)
        cost["actor_composition_checks"] += len(result)
        return result, {"composition": "passed"}

    return learning.run(root, preflight=preflight, output=output, protocol=spec, qualifier=qualify,
        source_loader=lambda r: (clones, predictor, initialization, calibrations),
        initialize_models=initialize, evaluation_weights=compose)


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    result.update(primary_endpoints=list(spec.PRIMARY_ENDPOINTS),
        performance_claim="matched_method_path_joint_vs_staged_frozen_policy_comparison_not_update_order_causality_or_frequency_superiority",
        joint_recipe_confirmation="mechanical_only" if preflight else (
            "supported" if all(result["endpoints"][k]["ci"][0] > 0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported"))
    return result
