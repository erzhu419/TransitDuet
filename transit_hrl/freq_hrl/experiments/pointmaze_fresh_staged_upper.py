"""Native staged-upper learning with the new teacher cohort's fixed lowers."""

import json

from . import pointmaze_staged_upper as previous
from . import pointmaze_fresh_lower as fresh_lower
from . import pointmaze_fresh_joint as fresh_joint
from scripts import pointmaze_fresh_staged_upper_stage100_spec as spec

learning = previous.learning


def qualify(cell, *, preflight):
    previous.qualify(cell, preflight=preflight, protocol=spec)
    if cell["source_initialization"] != spec.source_record(cell["root"]):
        raise ValueError("Fresh staged-upper source record changed")
    return cell


def run(root, *, preflight, output):
    records = {p: json.loads(p.read_text()) for p in {spec.donor_result(root, m) for m in spec.CHECKPOINT_METHODS}}
    for path, record in records.items():
        if record["root"] != root:
            raise ValueError("Fresh staged-upper donor root changed")
        if path == spec.donor_result(root, "joint_call"):
            fresh_joint.qualify(record, preflight=False)
        else:
            fresh_lower.qualify(record, preflight=False)
    training = {m: records[spec.donor_result(root, m)] for m in spec.CHECKPOINT_METHODS}
    donors = {}

    def initialize(period, models, cost):
        donors[period], metadata = previous.prepare_training(training, root, period, models, cost,
            protocol=spec, joint_checkpoint_protocol=spec.source)
        return metadata

    return learning.run(root, preflight=preflight, output=output, protocol=spec, qualifier=qualify,
        source_loader=fresh_joint.load_source, initialize_models=initialize,
        evaluation_weights=lambda p, w, c: previous.prepare_evaluation(donors[p], p, w, c, protocol=spec),
        scenario_pair_check=lambda pairs, roster: previous.check_scenario_pair(pairs, roster, root=root))


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    result.update(performance_claim="new_teacher_staged_upper_MC_mean_learning_given_frozen_lowers_not_frequency_superiority",
        primary_endpoints=list(spec.PRIMARY_ENDPOINTS), staged_confirmation="mechanical_only" if preflight else
        ("supported" if all(result["endpoints"][k]["ci"][0] > 0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported"))
    return result
