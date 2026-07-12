"""Convenience entrypoint for the tight-WEZ gun curriculum PPO trainer.

It reuses ``train_fast_aip_ppo.py`` unchanged, but defaults to:

* ``--stage-schedule gun_curriculum``
* ``--target-maneuver gun_curriculum``
* ``--output models``
* ``--variant ppo_lstm``
* ``--stop-stage 19``
"""
from __future__ import annotations

import sys

from train_fast_aip_ppo import main


def _has(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def _append_default(flag: str, value: str) -> None:
    if not _has(flag):
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _append_default("--variant", "ppo_lstm")
    _append_default("--stage-schedule", "gun_curriculum")
    _append_default("--target-maneuver", "gun_curriculum")
    _append_default("--output", "models")
    _append_default("--stop-stage", "19")
    raise SystemExit(main())
