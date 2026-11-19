"""Stage once, then manually dispatch the frozen smoke or full V503 matrix."""

import argparse
import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

from study import HERE, PLAN, aggregate

sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler


EXPERIMENTS = HERE.parent
DEPLOY = json.loads((EXPERIMENTS / "v400_independent_confirmation/deployment.json").read_text())
ORIGINAL = Path(json.loads((EXPERIMENTS / "v502_pathb_trc_matrix/staged_source02.json").read_text())["source"])
BASE = ORIGINAL.parent.parent / "FreqDuet-v503-matched-routing"
SOURCE = BASE / "source01"
RUN = BASE / "run01"
NODE = "node004"
NODES = [f"node{i:03}" for i in range(1, 7)]
REVIEW = HERE / "review"


def stage():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name in ("runtime.patch", "study.py", "plan.json", "PROTOCOL.md"):
            archive.add(HERE / name, arcname=name)
    payload = base64.b64encode(buf.getvalue()).decode()
    script = f"""
from pathlib import Path
import base64, io, subprocess, tarfile, tempfile
old = Path({str(ORIGINAL)!r})
source = Path({str(SOURCE)!r})
if not source.exists():
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='stage-v503-', dir=source.parent) as tmp:
        stage = Path(tmp) / 'release'
        stage.mkdir()
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as ar:
            ar.extractall(stage)
        subprocess.run(['patch', '--batch', '--fuzz=0', '-o', str(stage/'freqduet_v7_b1.py'),
                        str(old/'freqduet_v7_b1.py'), str(stage/'runtime.patch')], check=True)
        for name in ('freqduet_src', 'corridors'):
            target = old / name
            assert target.is_dir(), target
            (stage / name).symlink_to(target, target_is_directory=True)
        subprocess.run([{DEPLOY["python"]!r}, '-B', '-I', str(stage/'study.py'), 'check'],
                       check=True)
        stage.rename(source)
subprocess.run([{DEPLOY["python"]!r}, '-B', '-I', str(source/'study.py'), 'check'], check=True)
"""
    command = shlex.join([DEPLOY["python"], "-B", "-c", script])
    rc, out, err = scheduler.run_on(NODE, command, timeout=90, check=False)
    print(out, flush=True)
    if rc:
        raise RuntimeError(err[-2400:])


def task(phase, arm, seed):
    label = f"{arm}_{seed}"
    output = RUN / phase / label
    return {
        "description": f"FreqDuet V503 {phase} {label}",
        "project": "FreqDuet",
        "signature": f"FreqDuet/v503_r01/{phase}/{label}",
        "priority": "high",
        "allowed_nodes": NODES,
        "cpu": 1,
        "ram_mb": 3072,
        "vram": 0,
        "allow_cpu_training": True,
        "cpu_training_justification": "CPU transit simulator dominates runtime; small CPU SAC heads.",
        "allow_no_ckpt": True,
        "no_auto_requeue": True,
        "skip_launch_staging": True,
        "cwd": str(SOURCE),
        "env": {k: "1" for k in ("PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS",
                                  "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS")}
               | {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
        "cmd": shlex.join([DEPLOY["python"], "-B", "-I", "-u", str(SOURCE / "study.py"),
                           phase, "--arm", arm, "--seed", str(seed), "--output", str(output)]),
        "result_dir": str(output / "summary"),
        "local_result_dir": str(REVIEW / phase / label),
    }


def dispatch(phase):
    receipt = HERE / f"submission_{phase}.json"
    if receipt.exists():
        raise SystemExit(f"Already registered: {receipt}")
    if phase == "train":
        print(json.dumps(aggregate(REVIEW / "smoke", smoke=True)), flush=True)
    seeds = [PLAN["smoke"]["seed"]] if phase == "smoke" else PLAN["training_seeds"]
    specs = [task(phase, arm, seed) for seed in seeds for arm in PLAN["arms"]]
    signatures = {spec["signature"] for spec in specs}
    if any(t.get("signature") in signatures for t in scheduler.load_state()["tasks"]):
        raise SystemExit("Some V503 tasks are already registered; inspect them before dispatch.")
    args = [sys.executable, scheduler.__file__]
    res = subprocess.run(
        args + ["submit-jsonl", "--stdin", "--trusted", "--json", "--lock-timeout", "600"],
        input=json.dumps(specs), capture_output=True, text=True, check=True)
    tasks = [{"id": t["id"], "signature": t["signature"]}
             for t in json.loads(res.stdout)["submitted"]]
    receipt.write_text(json.dumps({"tasks": tasks}, indent=2) + "\n")
    print(tasks, flush=True)
    assert len(tasks) == len(specs), tasks
    ids = [arg for t in tasks for arg in ("--task-id", t["id"])]
    subprocess.run(args + ["dispatch", "--lock-timeout", "600", *ids], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("stage", "smoke", "train"))
    phase = parser.parse_args().phase
    if phase == "stage":
        stage()
    else:
        dispatch(phase)
