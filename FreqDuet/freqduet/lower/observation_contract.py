"""Auditable sensing contract for the protocol-v4 lower controller."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


_PRIVILEGED_CONTEXT = {
    "bwd_headway_norm",
    "headway_balance",
    "hold_value_proxy",
    "prev_queue",
    "next_queue",
}

_CONTEXT_LEDGER = {
    "load": ("APC onboard load after current stop service", "current stop event"),
    "capacity": ("vehicle specification", "static"),
    "queue": ("APC/dispatch estimate of passengers left behind", "current stop event"),
    "speed_residual": ("AVL segment speed", "latest causal AVL sample"),
    "shock_age": ("causal APC boarding residual filter", "latest completed filter bin"),
    "schedule_slack": ("executable upper plan and exact forward headway", "current decision"),
    "causal_hold_limit": (
        "matched predecessor departure and executable target headway",
        "current pre-action departure event",
    ),
    "fwd_headway_norm": ("same-stop AVL arrival recorder", "current arrival event"),
    "departure_gap_norm": (
        "matched predecessor departure and current operations clock",
        "current pre-action departure event",
    ),
    "departure_gap_valid": (
        "matched predecessor departure availability",
        "current pre-action departure event",
    ),
    "avl_follower_gap_norm": (
        "same-time follower AVL position and causal speed estimate",
        "current pre-action AVL snapshot",
    ),
    "avl_follower_gap_valid": (
        "same-time physical follower AVL availability",
        "current pre-action AVL snapshot",
    ),
    "route_progress": ("AVL vehicle location", "current decision"),
    "station_phase": ("static stop sequence and AVL stop id", "current stop event"),
    "prev_launch_gap": ("actual terminal launch log", "latest prior launch"),
    "next_launch_gap": ("published executable dispatch plan", "current plan"),
    "time_sin": ("operations clock", "current decision"),
    "time_cos": ("operations clock", "current decision"),
}


@dataclass(frozen=True)
class LowerObservationContract:
    mode: str
    input_schema: str
    reward_mode: str
    unobserved_action_mode: str
    frequency_enabled: bool
    frequency_source: str
    context_features: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        input_schema: str,
        reward_mode: str,
        unobserved_action_mode: str,
        frequency_enabled: bool,
        frequency_source: str,
        context_features: Iterable[str],
    ) -> "LowerObservationContract":
        contract = cls(
            mode=str(mode).strip().lower(),
            input_schema=str(input_schema).strip().lower(),
            reward_mode=str(reward_mode).strip().lower(),
            unobserved_action_mode=str(unobserved_action_mode).strip().lower(),
            frequency_enabled=bool(frequency_enabled),
            frequency_source=str(frequency_source).strip().lower(),
            context_features=tuple(str(item) for item in context_features),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.mode not in {"latent_oracle_legacy", "deployable_apc_avl_v4"}:
            raise ValueError("unknown lower observation contract")
        if self.mode != "deployable_apc_avl_v4":
            return
        if self.input_schema != "causal_forward_v4":
            raise ValueError(
                "deployable_apc_avl_v4 requires causal_forward_v4 input")
        if self.reward_mode != "forward_event_only":
            raise ValueError(
                "deployable_apc_avl_v4 requires forward_event_only reward")
        if self.unobserved_action_mode != "zero":
            raise ValueError(
                "deployable_apc_avl_v4 requires unobserved_action_mode=zero")
        if self.frequency_enabled and self.frequency_source != "apc_boardings":
            raise ValueError(
                "deployable_apc_avl_v4 requires APC frequency observations")
        privileged = _PRIVILEGED_CONTEXT.intersection(self.context_features)
        if privileged:
            raise ValueError(
                "deployable observation contract rejects context features: "
                + ", ".join(sorted(privileged)))
        unknown = set(self.context_features).difference(_CONTEXT_LEDGER)
        if unknown:
            raise ValueError(
                "deployable observation ledger has no source for: "
                + ", ".join(sorted(unknown)))

    def ledger(self) -> list[dict[str, object]]:
        deployable = self.mode == "deployable_apc_avl_v4"
        rows = [
            {
                "feature": "forward_headway",
                "source": "same-stop AVL arrival recorder",
                "timestamp": "current arrival minus previous causal arrival",
                "deployable": deployable,
            },
            {
                "feature": "forward_headway_valid",
                "source": "same-stop AVL event availability",
                "timestamp": "current decision",
                "deployable": deployable,
            },
            {
                "feature": "service_dwell",
                "source": "APC boarding/alighting service event",
                "timestamp": "current stop event",
                "deployable": deployable,
            },
            {
                "feature": "target_headway",
                "source": "executable upper dispatch plan",
                "timestamp": "active plan version",
                "deployable": True,
            },
            {
                "feature": "route_speed_profile",
                "source": "latest causal AVL speed samples",
                "timestamp": "current decision",
                "deployable": deployable,
            },
        ]
        for feature in self.context_features:
            source, timestamp = _CONTEXT_LEDGER.get(
                feature, ("simulator latent state", "current simulation time"))
            rows.append({
                "feature": feature,
                "source": source,
                "timestamp": timestamp,
                "deployable": deployable and feature in _CONTEXT_LEDGER,
            })
        if self.frequency_enabled:
            rows.append({
                "feature": "frequency_features",
                "source": (
                    "causal APC boarding filter"
                    if self.frequency_source == "apc_boardings"
                    else "simulator latent passenger arrivals"
                ),
                "timestamp": "latest completed causal frequency bin",
                "deployable": (
                    deployable and self.frequency_source == "apc_boardings"),
            })
        return rows

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.ledger(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
