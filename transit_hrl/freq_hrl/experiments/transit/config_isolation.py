"""Config-isolation audit for copied Transit ablation configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_RULES = {
    "T_allfreq_terminal.yaml": {
        "_name",
        "frequency.upper_mode",
        "frequency.lower_mode",
        "leakage.enable",
    },
    "T_swapped_terminal.yaml": {
        "_name",
        "frequency.upper_mode",
        "frequency.lower_mode",
        "leakage.enable",
    },
    "T_nopromotion_terminal.yaml": {
        "_name",
        "frequency.promotion.enable",
        "upper.timetable_planner.promotion_replan",
    },
    "T_noleakage_terminal.yaml": {
        "_name",
        "leakage.enable",
    },
}


def deep_merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, Mapping)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "_extends" not in cfg:
        return cfg
    parent = cfg.pop("_extends")
    candidates = [
        path.parent / parent,
        path.parent / ".." / parent,
        path.parent.parent / parent,
    ]
    parent_path = next((p.resolve() for p in candidates if p.exists()), None)
    if parent_path is None:
        raise FileNotFoundError(f"could not resolve parent {parent!r} from {path}")
    return deep_merge(load_config(parent_path), cfg)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    out: dict[str, Any] = {}
    for key, item in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        out.update(flatten(item, child))
    return out


def diff_paths(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    left = flatten(base)
    right = flatten(other)
    out = {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            out[key] = (left.get(key), right.get(key))
    return out


def audit_config_isolation(
    config_dir: str | Path,
    base_name: str = "T_freqhrl_terminal.yaml",
    rules: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    config_dir = Path(config_dir)
    base = load_config(config_dir / base_name)
    rules = rules or DEFAULT_RULES
    reports = []
    violations = []
    for name, allowed in rules.items():
        cfg = load_config(config_dir / name)
        diffs = diff_paths(base, cfg)
        unexpected = sorted(path for path in diffs if path not in allowed)
        missing = sorted(path for path in allowed if path not in diffs and path != "_name")
        row = {
            "config": name,
            "changed_paths": sorted(diffs.keys()),
            "allowed_paths": sorted(allowed),
            "unexpected_paths": unexpected,
            "missing_expected_paths": missing,
            "passed": not unexpected and not missing,
        }
        reports.append(row)
        if not row["passed"]:
            violations.append(row)
    return {
        "base": base_name,
        "reports": reports,
        "passed": not violations,
        "violations": violations,
    }


def write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Transit Config Isolation Audit",
        "",
        f"- base: `{result['base']}`",
        f"- passed: {result['passed']}",
        "",
        "| config | passed | changed paths | unexpected | missing expected |",
        "|---|---:|---|---|---|",
    ]
    for row in result["reports"]:
        lines.append(
            f"| {row['config']} "
            f"| {row['passed']} "
            f"| `{', '.join(row['changed_paths'])}` "
            f"| `{', '.join(row['unexpected_paths'])}` "
            f"| `{', '.join(row['missing_expected_paths'])}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("transit_hrl/freq_transitduet/configs_freqduet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/transit_config_isolation"),
    )
    args = parser.parse_args()
    result = audit_config_isolation(args.config_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_report(args.output_dir / "report.md", result)
    print(f"wrote {args.output_dir}")
    print(f"config_isolation passed={result['passed']}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
