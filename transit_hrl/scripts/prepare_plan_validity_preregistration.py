"""Write/validate the unified preregistration; no training, replay or scheduler API."""

import argparse
import json
from pathlib import Path

from freq_hrl.experiments.plan_validity_preregistration import draft_plan, registered_claims, validate_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, nargs="+", default=[])
    args = parser.parse_args(argv)
    plan = json.loads(args.input.read_text())["plan"] if args.input else draft_plan()
    result = {"plan": plan, "readiness": validate_plan(plan)}
    if args.summaries:
        result["claims"] = registered_claims(plan, [json.loads(path.read_text()) for path in args.summaries])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
