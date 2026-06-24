#!/usr/bin/env python3
"""Generate fixed-action configs for action-level counterfactual rollouts.

These configs use ``upper.action_override`` to replace the learned upper action
with a fixed delta/headway-plan vector. Running the generated configs over the
same domain/seed set gives common-random-number labels for terminal/headway
actions, which is stronger evidence than same-trajectory proxy value targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DOMAINS = {
    "terminal": "F_freqduet_terminal_main_hiro",
    "highnoise": "F_freqduet_gen_highnoise_main_hiro",
    "odshift": "F_freqduet_gen_odshift_main_hiro",
    "rushshift": "F_freqduet_gen_rushshift_main_hiro",
}


def parse_csv_floats(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return values


def token_for_delta(delta: float) -> str:
    value = int(round(float(delta)))
    if abs(float(delta) - value) > 1e-6:
        text = f"{float(delta):+.1f}".replace("+", "p").replace("-", "m")
        return text.replace(".", "p")
    return f"p{value}" if value >= 0 else f"m{abs(value)}"


def domain_from_config(config_name: str) -> str:
    text = config_name.lower()
    if "highnoise" in text:
        return "highnoise"
    if "odshift" in text:
        return "odshift"
    if "rushshift" in text:
        return "rushshift"
    if "terminal" in text:
        return "terminal"
    raise argparse.ArgumentTypeError(f"cannot infer domain from {config_name!r}")


def parse_base_configs(text: str) -> dict[str, str]:
    if not text:
        return dict(DEFAULT_DOMAINS)
    out: dict[str, str] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            domain, config = [piece.strip() for piece in item.split(":", 1)]
        else:
            config = item
            domain = domain_from_config(config)
        if domain not in DEFAULT_DOMAINS:
            raise argparse.ArgumentTypeError(f"unsupported domain {domain!r}")
        out[domain] = config.removesuffix(".yaml")
    if not out:
        raise argparse.ArgumentTypeError("no base configs parsed")
    return out


def config_prefix(base_config: str) -> str:
    return base_config.removesuffix("_hiro")


def write_yaml(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_config_text(
    base_config: str,
    name: str,
    delta: float,
    terminal_dispatch: bool,
    terminal_shift_min_s: float,
    terminal_shift_max_s: float,
    disable_value_selectors: bool,
    trip_dump_freq: int | None,
) -> list[str]:
    lines = [
        f"_extends: {base_config}.yaml",
        f"_name: {name}",
        "",
        "# Generated action-level counterfactual config.",
        "# The upper actor is replaced by a fixed action for matched-seed rollout labels.",
        "upper:",
        "  action_override:",
        "    enable: true",
        f"    delta_s: {float(delta):.6g}",
        f"    disable_value_selectors: {str(bool(disable_value_selectors)).lower()}",
        "  timetable_planner:",
    ]
    if terminal_dispatch:
        lines.extend([
            "    terminal_dispatch: true",
            f"    terminal_shift_min_s: {float(terminal_shift_min_s):.6g}",
            f"    terminal_shift_max_s: {float(terminal_shift_max_s):.6g}",
        ])
    lines.extend([
        "    headway_value_planner:",
        "      enable: false",
        "    terminal_value_selector:",
        "      enable: false",
    ])
    if trip_dump_freq is not None:
        lines.extend([
            "training:",
            f"  trip_dump_freq: {int(trip_dump_freq)}",
        ])
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-configs", type=parse_base_configs, default=dict(DEFAULT_DOMAINS))
    ap.add_argument("--deltas-s", type=parse_csv_floats, default=[-20.0, -10.0, 0.0, 10.0, 20.0])
    ap.add_argument(
        "--modes",
        default="target,terminalhold45",
        help="Comma-separated modes: target, terminalhold45, terminalrelease10.",
    )
    ap.add_argument("--out-config-dir", type=Path, default=Path("configs_freqduet/counterfactual_action"))
    ap.add_argument("--out-manifest", type=Path, default=Path("results_freqduet/action_counterfactual_config_manifest.json"))
    ap.add_argument("--name-prefix", default="cfaction")
    ap.add_argument("--disable-value-selectors", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--trip-dump-freq",
        type=int,
        default=None,
        help="Override training.trip_dump_freq in generated configs.",
    )
    args = ap.parse_args()

    mode_specs = []
    for raw_mode in str(args.modes).split(","):
        mode = raw_mode.strip().lower()
        if not mode:
            continue
        if mode == "target":
            mode_specs.append((mode, False, 0.0, 0.0))
        elif mode == "terminalhold45":
            mode_specs.append((mode, True, 0.0, 45.0))
        elif mode == "terminalrelease10":
            mode_specs.append((mode, True, -10.0, 45.0))
        else:
            raise SystemExit(f"unsupported mode {mode!r}")
    if not mode_specs:
        raise SystemExit("no modes requested")

    generated = []
    for domain, base_config in args.base_configs.items():
        for mode, terminal_dispatch, shift_min, shift_max in mode_specs:
            for delta in args.deltas_s:
                delta_token = token_for_delta(delta)
                name = (
                    f"{config_prefix(base_config)}_"
                    f"{args.name_prefix}_{mode}_d{delta_token}_hiro"
                )
                path = args.out_config_dir / f"{name}.yaml"
                write_yaml(
                    path,
                    build_config_text(
                        base_config=base_config,
                        name=name,
                        delta=float(delta),
                        terminal_dispatch=terminal_dispatch,
                        terminal_shift_min_s=shift_min,
                        terminal_shift_max_s=shift_max,
                        disable_value_selectors=args.disable_value_selectors,
                        trip_dump_freq=args.trip_dump_freq,
                    ),
                )
                generated.append({
                    "domain": domain,
                    "base_config": base_config,
                    "mode": mode,
                    "delta_s": float(delta),
                    "config": name,
                    "path": str(path),
                    "terminal_dispatch": bool(terminal_dispatch),
                    "terminal_shift_min_s": float(shift_min),
                    "terminal_shift_max_s": float(shift_max),
                })

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(
        json.dumps({
            "base_configs": args.base_configs,
            "deltas_s": [float(x) for x in args.deltas_s],
            "modes": [spec[0] for spec in mode_specs],
            "generated": generated,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"generated {len(generated)} configs")
    print(args.out_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
