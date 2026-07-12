import numpy as np


class Maneuver:
    """Open-loop maneuver descriptor.

    Args:
        name: Stable maneuver key.
        duration_s: Maneuver duration in seconds.
        action: Function returning control action for a timestamp.
        description: Short human-readable description.
    """

    __slots__ = ("name", "duration_s", "action", "description")

    def __init__(self, name, duration_s, action, description):
        self.name = name
        self.duration_s = duration_s
        self.action = action
        self.description = description


def _segments(*items):
    """Build an action function from (segment end time, action) entries."""

    def action(t):
        for end, value in items:
            if t < end:
                return np.asarray(value, dtype=np.float32)

        return np.asarray(items[-1][1], dtype=np.float32)

    return action


MANEUVERS = {
    "trim": Maneuver(
        "trim",
        12.0,
        _segments((12.0, (0.0, 0.0, 0.0, 0.65))),
        "Neutral controls at constant throttle.",
    ),
    "aileron_step": Maneuver(
        "aileron_step",
        12.0,
        _segments(
            (2.0, (0.0, 0.0, 0.0, 0.65)),
            (5.0, (0.45, 0.0, 0.0, 0.65)),
            (8.0, (-0.45, 0.0, 0.0, 0.65)),
            (12.0, (0.0, 0.0, 0.0, 0.65)),
        ),
        "Positive then negative roll command.",
    ),
    "pull_up": Maneuver(
        "pull_up",
        12.0,
        _segments(
            (2.0, (0.0, 0.0, 0.0, 0.75)),
            (6.0, (0.0, 0.35, 0.0, 0.85)),
            (12.0, (0.0, 0.0, 0.0, 0.75)),
        ),
        "Moderate positive pitch pulse.",
    ),
    "rudder_step": Maneuver(
        "rudder_step",
        10.0,
        _segments(
            (2.0, (0.0, 0.0, 0.0, 0.65)),
            (5.0, (0.0, 0.0, 0.35, 0.65)),
            (10.0, (0.0, 0.0, 0.0, 0.65)),
        ),
        "Rudder pulse for lateral-directional response.",
    ),
    "throttle_step": Maneuver(
        "throttle_step",
        18.0,
        _segments(
            (4.0, (0.0, 0.0, 0.0, 0.35)),
            (11.0, (0.0, 0.0, 0.0, 1.0)),
            (18.0, (0.0, 0.0, 0.0, 0.55)),
        ),
        "Low/high/medium throttle response.",
    ),
    "banked_pull": Maneuver(
        "banked_pull",
        16.0,
        _segments(
            (2.0, (0.0, 0.0, 0.0, 0.8)),
            (6.0, (0.45, 0.0, 0.0, 0.85)),
            (12.0, (0.12, 0.28, 0.04, 0.9)),
            (16.0, (0.0, 0.0, 0.0, 0.75)),
        ),
        "Roll-in followed by a banked pull.",
    ),
}


def select(names):
    if not names or names == ["all"]:
        return list(MANEUVERS.values())

    unknown = sorted(set(names) - set(MANEUVERS))

    if unknown:
        raise ValueError(f"Unknown maneuvers: {', '.join(unknown)}")

    return [MANEUVERS[name] for name in names]
