"""Orchestrate all Route B cluster experiments (B1 ReacherPolygonal, B2 PointMaze Strong Baselines, B3 CEM Planner) via scheduleurm."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_route_a_scheduleurm import remote_python, submit_jobs, tasks_with


PM_PLAN = Path("protocols/plan_validity_route_b_strong_baselines_2026-11-18.json")
PM_RUNNER = "scripts/run_route_b_pointmaze_baselines.py"

REACHER_PLAN = Path("protocols/plan_validity_reacher_polygonal_confirmation_2026-11-18.json")
REACHER_RUNNER = "scripts/run_route_b_reacher_polygonal.py"

CEM_PLAN = Path("protocols/plan_validity_route_b_expensive_cem_planner_2026-11-18.json")
CEM_RUNNER = "scripts/run_route_b_expensive_planner.py"


def check_remote_paths(rel_paths: list[str]) -> dict:
    probe = (
        "import json\n"
        "from pathlib import Path\n"
        f"paths = {rel_paths!r}\n"
        "rows = [json.loads(Path(p).read_text()) if Path(p).is_file() else None for p in paths]\n"
        "print(json.dumps({'statuses': [r['status'] if r else None for r in rows], "
        "'done': all(rows)}, separators=(',', ':')))\n"
    )
    return json.loads(remote_python(["-c", probe], timeout=60))


def wait_paths(label: str, rel_paths: list[str], signatures: set[str], *, hours: float = 3.0) -> list[str]:
    cutoff = time.monotonic() + 3600.0 * hours
    previous = None
    next_q = 0.0
    while time.monotonic() < cutoff:
        res = check_remote_paths(rel_paths)
        done_cnt = sum(1 for s in res["statuses"] if s is not None)
        if res["statuses"] != previous:
            print(f"[{label}] {done_cnt}/{len(rel_paths)} finished: {res['statuses']}", flush=True)
            previous = res["statuses"]
        if res["done"]:
            if any(s != "complete" for s in res["statuses"]):
                err_probe = (
                    "import json\nfrom pathlib import Path\n"
                    f"paths = {rel_paths!r}\n"
                    "print(json.dumps([json.loads(Path(p).read_text()) for p in paths if Path(p).is_file()]))\n"
                )
                raise RuntimeError(f"{label} failed cells: {remote_python(['-c', err_probe])}")
            return res["statuses"]
        if time.monotonic() >= next_q:
            next_q = time.monotonic() + 300.0
            st = tasks_with(signatures)
            if len(st) == len(rel_paths) and all(
                t.get("status") in {"done", "failed", "cancelled", "canceled"} for t in st
            ):
                res2 = check_remote_paths(rel_paths)
                if res2["done"] and all(s == "complete" for s in res2["statuses"]):
                    return res2["statuses"]
                raise RuntimeError(
                    f"{label} terminated incomplete: "
                    + str([(t["id"], t.get("status"), t.get("exit_code")) for t in st])
                )
        time.sleep(30)
    raise TimeoutError(f"{label} timed out after {hours}h")


def pull_summary(run_name: str, runner_rel: str, task_ids: list[str]) -> None:
    local_dir = ROOT / "results" / run_name
    local_dir.mkdir(parents=True, exist_ok=True)
    headline = remote_python([runner_rel, "readout"], timeout=180)
    print(f"=== {run_name} READOUT ===\n{headline}", flush=True)
    full_summary = remote_python(
        [
            "-c",
            f"from pathlib import Path; print(Path('results/{run_name}/summary.json').read_text(), end='')",
        ],
        timeout=180,
    )
    (local_dir / "summary.json").write_text(full_summary, encoding="utf-8")
    (local_dir / "completion.json").write_text(
        json.dumps(
            {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "task_ids": task_ids,
                "summary_bytes": len(full_summary.encode("utf-8")),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    pm = json.loads((ROOT / PM_PLAN).read_text())
    rc = json.loads((ROOT / REACHER_PLAN).read_text())
    cem = json.loads((ROOT / CEM_PLAN).read_text())

    # Wave 1: Submit all 40 independent build/CEM jobs in parallel!
    cem_ids, cem_sigs = submit_jobs(cem["jobs"], cem["run_name"])
    pm_build_jobs = pm["jobs"]["train_periodic"] + pm["jobs"]["fit_baselines"]
    pm_b_ids, pm_b_sigs = submit_jobs(pm_build_jobs, pm["run_name"] + "_build")
    rc_build_jobs = rc["jobs"]["build_shared"] + rc["jobs"]["build_periodic"]
    rc_b_ids, rc_b_sigs = submit_jobs(rc_build_jobs, rc["run_name"] + "_build")

    # Wait for CEM (fastest: ~3-5 min) and pull summary immediately
    cem_paths = [f"{c['public']}/result.json" for c in cem["cells"]]
    wait_paths("B3-CEM", cem_paths, cem_sigs, hours=2.0)
    pull_summary(cem["run_name"], CEM_RUNNER, cem_ids)

    # Wait for PointMaze B2 builds, then submit & wait PointMaze B2 evaluations
    pm_build_paths = (
        [f"{c['train_periodic_public']}/result.json" for c in pm["cells"]]
        + [f"{c['fit_baselines_public']}/result.json" for c in pm["cells"]]
    )
    wait_paths("B2-PM-Build", pm_build_paths, pm_b_sigs, hours=3.0)
    pm_e_ids, pm_e_sigs = submit_jobs(pm["jobs"]["evaluate"], pm["run_name"] + "_eval")

    # Wait for ReacherPolygonal B1 builds, then submit & wait ReacherPolygonal B1 evaluations
    rc_build_paths = (
        [f"{c['build_shared_public']}/result.json" for c in rc["cells"]]
        + [f"{c['build_periodic_public']}/result.json" for c in rc["cells"]]
    )
    wait_paths("B1-Reacher-Build", rc_build_paths, rc_b_sigs, hours=3.0)
    rc_e_ids, rc_e_sigs = submit_jobs(rc["jobs"]["evaluate"], rc["run_name"] + "_eval")

    # Wait for both evaluation waves and pull full summaries
    pm_eval_paths = [f"{c['evaluate_public']}/result.json" for c in pm["cells"]]
    wait_paths("B2-PM-Eval", pm_eval_paths, pm_e_sigs, hours=2.0)
    pull_summary(pm["run_name"], PM_RUNNER, pm_b_ids + pm_e_ids)

    rc_eval_paths = [f"{c['evaluate_public']}/result.json" for c in rc["cells"]]
    wait_paths("B1-Reacher-Eval", rc_eval_paths, rc_e_sigs, hours=2.0)
    pull_summary(rc["run_name"], REACHER_RUNNER, rc_b_ids + rc_e_ids)

    print("ALL ROUTE B EXPERIMENTS COMPLETED SUCCESSFULLY.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
