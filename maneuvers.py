"""Deterministic open-loop inputs shared by both F-16 adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ActionFn = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class Maneuver:
    name: str
    duration_s: float
    action: ActionFn
    description: str


def _segments(*items: tuple[float, tuple[float, float, float, float]]) -> ActionFn:
    """Build an action function from (segment end time, action) entries."""
    def action(t: float) -> np.ndarray:
        for end, value in items:
            if t < end:
                return np.asarray(value, dtype=np.float32)
        return np.asarray(items[-1][1], dtype=np.float32)
    return action


MANEUVERS: dict[str, Maneuver] = {
    "trim": Maneuver(
        "trim", 12.0,
        _segments((12.0, (0.0, 0.0, 0.0, 0.65))),
        "Neutral controls at constant throttle.",
    ),
    "aileron_step": Maneuver(
        "aileron_step", 12.0,
        _segments((2.0, (0.0, 0.0, 0.0, 0.65)),
                  (5.0, (0.45, 0.0, 0.0, 0.65)),
                  (8.0, (-0.45, 0.0, 0.0, 0.65)),
                  (12.0, (0.0, 0.0, 0.0, 0.65))),
        "Positive then negative roll command.",
    ),
    "pull_up": Maneuver(
        "pull_up", 12.0,
        _segments((2.0, (0.0, 0.0, 0.0, 0.75)),
                  (6.0, (0.0, 0.35, 0.0, 0.85)),
                  (12.0, (0.0, 0.0, 0.0, 0.75))),
        "Moderate positive pitch pulse.",
    ),
    "rudder_step": Maneuver(
        "rudder_step", 10.0,
        _segments((2.0, (0.0, 0.0, 0.0, 0.65)),
                  (5.0, (0.0, 0.0, 0.35, 0.65)),
                  (10.0, (0.0, 0.0, 0.0, 0.65))),
        "Rudder pulse for lateral-directional response.",
    ),
    "throttle_step": Maneuver(
        "throttle_step", 18.0,
        _segments((4.0, (0.0, 0.0, 0.0, 0.35)),
                  (11.0, (0.0, 0.0, 0.0, 1.0)),
                  (18.0, (0.0, 0.0, 0.0, 0.55))),
        "Low/high/medium throttle response.",
    ),
    "banked_pull": Maneuver(
        "banked_pull", 16.0,
        _segments((2.0, (0.0, 0.0, 0.0, 0.8)),
                  (6.0, (0.45, 0.0, 0.0, 0.85)),
                  (12.0, (0.12, 0.28, 0.04, 0.9)),
                  (16.0, (0.0, 0.0, 0.0, 0.75))),
        "Roll-in followed by a banked pull.",
    ),
}


def select(names: list[str] | None) -> list[Maneuver]:
    if not names or names == ["all"]:
        return list(MANEUVERS.values())
    unknown = sorted(set(names) - set(MANEUVERS))
    if unknown:
        raise ValueError(f"Unknown maneuvers: {', '.join(unknown)}")
    return [MANEUVERS[name] for name in names]

