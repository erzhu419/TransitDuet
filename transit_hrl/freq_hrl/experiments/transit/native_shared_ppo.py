"""Native TransitDuet bridge for the shared Freq-HRL PPO core.

This module does not modify the copied TransitDuet runner.  It provides a
small adapter that instantiates the native runner, reads its real state/action
contract, and maps the domain-agnostic Gaussian PPO actions into native
timetable and holding actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.rl import DualActorCriticPPO, DualPPOConfig


TRANSIT_HRL_ROOT = Path(__file__).resolve().parents[3]
TRANSIT_DUET_ROOT = TRANSIT_HRL_ROOT / "freq_transitduet"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _array(value: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


@dataclass
class NativeTransitContract:
    upper_state_dim: int
    lower_state_dim: int
    upper_action_dim: int
    lower_action_dim: int
    upper_action_low: list[float]
    upper_action_high: list[float]
    lower_action_range_s: float
    lower_action_bins: list[float]
    frequency_method: str
    timetable_planner: bool
    terminal_dispatch: bool
    promotion_replan: bool
    shared_core: str = "freq_hrl.rl.DualActorCriticPPO"

    def as_dict(self) -> dict[str, Any]:
        return {
            "upper_state_dim": int(self.upper_state_dim),
            "lower_state_dim": int(self.lower_state_dim),
            "upper_action_dim": int(self.upper_action_dim),
            "lower_action_dim": int(self.lower_action_dim),
            "upper_action_low": list(self.upper_action_low),
            "upper_action_high": list(self.upper_action_high),
            "lower_action_range_s": float(self.lower_action_range_s),
            "lower_action_bins": list(self.lower_action_bins),
            "frequency_method": str(self.frequency_method),
            "timetable_planner": bool(self.timetable_planner),
            "terminal_dispatch": bool(self.terminal_dispatch),
            "promotion_replan": bool(self.promotion_replan),
            "shared_core": self.shared_core,
        }


class NativeTransitPPOBridge:
    """Map shared PPO actions into native TransitDuet action spaces."""

    def __init__(
        self,
        contract: NativeTransitContract,
        model: DualActorCriticPPO | None = None,
        *,
        hidden_dim: int = 0,
        init_log_std: float = -2.0,
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.contract = contract
        self.upper_action_low = _array(contract.upper_action_low, dtype=np.float64)
        self.upper_action_high = _array(contract.upper_action_high, dtype=np.float64)
        if self.upper_action_low.size != int(contract.upper_action_dim):
            raise ValueError("upper_action_low must match upper_action_dim")
        if self.upper_action_high.size != int(contract.upper_action_dim):
            raise ValueError("upper_action_high must match upper_action_dim")
        self.lower_action_bins = _array(contract.lower_action_bins, dtype=np.float64)
        self.model = model or DualActorCriticPPO(DualPPOConfig(
            upper_state_dim=int(contract.upper_state_dim),
            lower_state_dim=int(contract.lower_state_dim),
            upper_action_dim=int(contract.upper_action_dim),
            lower_action_dim=int(contract.lower_action_dim),
            hidden_dim=int(hidden_dim),
            init_log_std=float(init_log_std),
            learning_rate=float(learning_rate),
            device=str(device),
        ))

    @classmethod
    def from_runner(
        cls,
        runner: Any,
        *,
        hidden_dim: int = 0,
        init_log_std: float = -2.0,
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ) -> "NativeTransitPPOBridge":
        cfg = getattr(runner, "cfg", {})
        lower_cfg = cfg.get("lower", {}) if isinstance(cfg, dict) else {}
        freq_cfg = cfg.get("frequency", {}) if isinstance(cfg, dict) else {}
        planner_cfg = cfg.get("upper", {}).get("timetable_planner", {}) if isinstance(cfg, dict) else {}
        lower_action_range = float(lower_cfg.get("action_range", 60.0))
        lower_bins = getattr(runner, "lower_action_bins", None)
        contract = NativeTransitContract(
            upper_state_dim=int(runner.upper_state_dim),
            lower_state_dim=int(runner.lower_state_dim),
            upper_action_dim=int(runner.upper_action_dim),
            lower_action_dim=1,
            upper_action_low=_array(runner.upper_action_low).astype(float).tolist(),
            upper_action_high=_array(runner.upper_action_high).astype(float).tolist(),
            lower_action_range_s=lower_action_range,
            lower_action_bins=(
                _array(lower_bins).astype(float).tolist()
                if lower_bins is not None else []
            ),
            frequency_method=str(freq_cfg.get("method", "unknown")),
            timetable_planner=bool(getattr(runner, "timetable_planner", None) is not None),
            terminal_dispatch=bool(getattr(runner, "timetable_terminal_dispatch", False)),
            promotion_replan=bool(planner_cfg.get(
                "promotion_replan",
                getattr(runner, "timetable_promotion_replan", False),
            )),
        )
        return cls(
            contract,
            hidden_dim=hidden_dim,
            init_log_std=init_log_std,
            learning_rate=learning_rate,
            device=device,
        )

    def upper_latent_to_native(self, latent_action: Any) -> np.ndarray:
        latent = _array(latent_action, dtype=np.float64)
        if latent.size != int(self.contract.upper_action_dim):
            raise ValueError("upper latent action has the wrong dimension")
        weight = 0.5 * (np.tanh(latent) + 1.0)
        return (
            self.upper_action_low
            + weight * (self.upper_action_high - self.upper_action_low)
        ).astype(np.float32)

    def lower_latent_to_native(self, latent_action: Any) -> np.ndarray:
        latent = _array(latent_action, dtype=np.float64)
        if latent.size < 1:
            raise ValueError("lower latent action must have at least one dimension")
        value = float(self.contract.lower_action_range_s) * float(_sigmoid(latent[:1])[0])
        if self.lower_action_bins.size:
            idx = int(np.argmin(np.abs(self.lower_action_bins - value)))
            value = float(self.lower_action_bins[idx])
        value = float(np.clip(value, 0.0, float(self.contract.lower_action_range_s)))
        return np.asarray([value], dtype=np.float32)

    def act_upper_native(self, upper_state: Any, sample: bool = False) -> dict[str, Any]:
        out = self.model.act_upper(_array(upper_state), sample=sample)
        latent = _array(out["action"], dtype=np.float64)
        native = self.upper_latent_to_native(latent)
        return {
            "native_action": native,
            "latent_action": latent.astype(np.float32),
            "logp": float(out["logp"]),
            "value": float(out["value"]),
        }

    def act_lower_native(self, lower_state: Any, sample: bool = False) -> dict[str, Any]:
        out = self.model.act_lower(_array(lower_state), sample=sample)
        latent = _array(out["action"], dtype=np.float64)
        native = self.lower_latent_to_native(latent)
        return {
            "native_action": native,
            "latent_action": latent.astype(np.float32),
            "logp": float(out["logp"]),
            "value": float(out["value"]),
        }

    def contract_dict(self) -> dict[str, Any]:
        return self.contract.as_dict()


def load_native_runner(
    config_path: Path,
    *,
    seed: int,
    logs_dir: Path | None,
    device: str = "cpu",
) -> Any:
    if str(TRANSIT_HRL_ROOT) not in sys.path:
        sys.path.insert(0, str(TRANSIT_HRL_ROOT))
    if str(TRANSIT_DUET_ROOT) not in sys.path:
        sys.path.insert(0, str(TRANSIT_DUET_ROOT))
    from freq_transitduet.runner_v3 import TransitDuetV2Runner, load_config

    cfg = load_config(str(config_path))
    cfg["seed"] = int(seed)
    if logs_dir is not None:
        cfg.setdefault("logging", {})["logs_dir"] = str(logs_dir)
    return TransitDuetV2Runner(cfg, device=device)


def run_native_shared_ppo_audit(
    output_dir: Path,
    config_path: Path,
    *,
    seed: int = 7,
    device: str = "cpu",
    keep_native_log_dir: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    native_logs = output_dir / "native_logs"
    runner = load_native_runner(
        config_path,
        seed=int(seed),
        logs_dir=native_logs,
        device=str(device),
    )
    bridge = NativeTransitPPOBridge.from_runner(runner, device=device)
    upper_state = np.zeros(bridge.contract.upper_state_dim, dtype=np.float32)
    lower_state = np.zeros(bridge.contract.lower_state_dim, dtype=np.float32)
    upper = bridge.act_upper_native(upper_state, sample=False)
    lower = bridge.act_lower_native(lower_state, sample=False)
    contract = bridge.contract_dict()
    checks = {
        "native_runner_instantiated": True,
        "uses_shared_core": isinstance(bridge.model, DualActorCriticPPO),
        "upper_action_dim_matches_native": (
            int(upper["native_action"].size) == int(contract["upper_action_dim"])
        ),
        "upper_action_in_bounds": bool(np.all(
            upper["native_action"] >= bridge.upper_action_low - 1e-6
        ) and np.all(upper["native_action"] <= bridge.upper_action_high + 1e-6)),
        "lower_action_in_bounds": bool(
            0.0 <= float(lower["native_action"][0]) <= float(contract["lower_action_range_s"])
        ),
        "native_timetable_terminal_dispatch": bool(contract["terminal_dispatch"]),
        "native_promotion_replan": bool(contract["promotion_replan"]),
    }
    summary = {
        "config_path": str(config_path),
        "seed": int(seed),
        "contract": contract,
        "smoke_actions": {
            "upper_native_action": _array(upper["native_action"]).astype(float).tolist(),
            "lower_native_action": _array(lower["native_action"]).astype(float).tolist(),
            "upper_logp": float(upper["logp"]),
            "lower_logp": float(lower["logp"]),
        },
        "checks": checks,
        "status": "supported_interface" if all(checks.values()) else "failed_interface",
    }
    write_outputs(output_dir, summary)
    if native_logs.exists() and not keep_native_log_dir:
        shutil.rmtree(native_logs)
    return summary


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    row = {
        "status": summary["status"],
        **summary["contract"],
        **summary["checks"],
    }
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    lines = [
        "# Native Transit Shared-PPO Interface Audit",
        "",
        f"- status: {summary['status']}",
        f"- config: `{summary['config_path']}`",
        f"- shared core: `{summary['contract']['shared_core']}`",
        f"- upper contract: state={summary['contract']['upper_state_dim']} action={summary['contract']['upper_action_dim']}",
        f"- lower contract: state={summary['contract']['lower_state_dim']} action={summary['contract']['lower_action_dim']}",
        f"- terminal dispatch: {summary['contract']['terminal_dispatch']}",
        f"- promotion replan: {summary['contract']['promotion_replan']}",
        "",
        "| check | value |",
        "|---|---:|",
    ]
    for key, value in summary["checks"].items():
        lines.append(f"| {key} | {value} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=TRANSIT_DUET_ROOT / "configs_freqduet" / "T_freqhrl_native_full.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/transit_native_shared_ppo_audit"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--keep-native-log-dir", action="store_true")
    args = parser.parse_args()
    summary = run_native_shared_ppo_audit(
        output_dir=args.output_dir,
        config_path=args.config,
        seed=int(args.seed),
        device=str(args.device),
        keep_native_log_dir=bool(args.keep_native_log_dir),
    )
    print(f"wrote {args.output_dir}")
    print(
        "native_shared_ppo "
        f"status={summary['status']} "
        f"upper_dim={summary['contract']['upper_state_dim']}x{summary['contract']['upper_action_dim']} "
        f"lower_dim={summary['contract']['lower_state_dim']}x{summary['contract']['lower_action_dim']}"
    )


if __name__ == "__main__":
    main()
