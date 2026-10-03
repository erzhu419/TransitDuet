"""Refine the registered joint upper while holding each matched lower fixed."""

import copy
import json

from . import pointmaze_fresh_staged_upper as fresh
from scripts import pointmaze_upper_refinement_stage101_spec as spec

learning, previous = fresh.learning, fresh.previous


def initialize_upper(models, donors, cost):
    upper = donors["joint_call"]["upper_actor"]
    for model in models.values():
        before = copy.deepcopy(model.state_dict())
        model.upper_actor.load_state_dict(upper)
        learning.native.curves.support.assert_frozen(model, {**before, "upper_actor": upper})
        cost["upper_initialization_checks"] += 1


def qualify(cell, *, preflight):
    previous.qualify(cell, preflight=preflight, protocol=spec)
    if cell["source_initialization"] != spec.source_record(cell["root"]):
        raise ValueError("Upper-refinement source record changed")
    if any(g["upper_initialization"] != spec.UPPER_INITIALIZATION for g in cell["groups"].values()):
        raise ValueError("Upper refinement must start from the registered final UJ")
    return cell


def run(root, *, preflight, output):
    records = {p: json.loads(p.read_text()) for p in {spec.donor_result(root, m) for m in spec.CHECKPOINT_METHODS}}
    for path, record in records.items():
        if record["root"] != root:
            raise ValueError("Upper-refinement donor root changed")
        if path == spec.donor_result(root, "joint_call"):
            fresh.fresh_joint.qualify(record, preflight=False)
        else:
            fresh.fresh_lower.qualify(record, preflight=False)
    training = {m: records[spec.donor_result(root, m)] for m in spec.CHECKPOINT_METHODS}
    donors = {}

    def initialize(period, models, cost):
        donors[period], metadata = previous.prepare_training(training, root, period, models, cost,
            protocol=spec, joint_checkpoint_protocol=spec.source)
        initialize_upper(models, donors[period], cost)
        return {**metadata, "upper_initialization": spec.UPPER_INITIALIZATION}

    return learning.run(root, preflight=preflight, output=output, protocol=spec, qualifier=qualify,
        source_loader=fresh.fresh_joint.load_source, initialize_models=initialize,
        evaluation_weights=lambda p, w, c: previous.prepare_evaluation(donors[p], p, w, c, protocol=spec),
        scenario_pair_check=lambda pairs, roster: previous.check_scenario_pair(pairs, roster, root=root))


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    refinement, matching = "mechanical_only", "mechanical_only"
    if not preflight:
        endpoints = result["endpoints"]
        refinement = "supported" if all(endpoints[k]["ci"][0] > 0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported"
        matching = "supported" if all(
            endpoints[f"{p}/refined_common_minus_independent_upper_common_lower"]["ci"][0] > 0 and
            endpoints[f"{p}/common_upper_independent_lower_minus_refined_independent"]["ci"][1] < 0
            for p in spec.PERIODS) else "not_supported"
    result.update(performance_claim="conditional_UJ_upper_refinement_with_additional_compute_not_frequency_superiority",
        primary_endpoints=list(spec.PRIMARY_ENDPOINTS), refinement_confirmation=refinement, matched_upper_confirmation=matching)
    return result
