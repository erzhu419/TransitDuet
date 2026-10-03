"""Use the same MC mean learner with fresh teacher and decoder artifacts."""

import json

from . import pointmaze_call_weighted as learning
from . import pointmaze_fresh_decoder as decoder
from scripts import pointmaze_fresh_joint_stage98_spec as spec


def load_source(root):
    source = json.loads(spec.source_result(root).read_text())
    decoder.qualify(source, preflight=False)
    models, predictor, teacher = decoder.load_source(root)
    if source["source_checkpoints"] != teacher["checkpoints"]:
        raise ValueError("fresh joint decoder and teacher refer to different checkpoints")
    expected = spec.source_record(root)
    actual = {**expected, "checkpoints": {str(p): teacher["checkpoints"][f"clone_{p}"] for p in spec.PERIODS},
        "forecaster": teacher["forecaster"]}
    if actual != expected:
        raise ValueError("fresh joint source is not the full new teacher cohort")
    calibrations = {str(p): source["groups"][str(p)]["calibration"] for p in spec.PERIODS}
    return models, predictor, actual, calibrations


def qualify(cell, *, preflight):
    learning.qualify(cell, preflight=preflight, protocol=spec)
    if cell["source_initialization"] != spec.source_record(cell["root"]):
        raise ValueError("fresh joint source record changed")
    return cell


def run(root, *, preflight, output):
    return learning.learning.run(root, preflight=preflight, output=output, protocol=spec,
        qualifier=qualify, source_loader=load_source)


def aggregate(cells, *, preflight):
    result = learning.learning.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    result.update(performance_claim="new_teacher_fixed_std_decoder_MC_mean_learning_not_full_actor_critic",
        independent_confirmation=spec.confirmation(result))
    return result
