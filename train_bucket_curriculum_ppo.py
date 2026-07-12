"""Run PPO on the bucketized gun curriculum.

Wrapper defaults:

* ``--stage-schedule gun_bucket_curriculum``
* ``--target-maneuver gun_curriculum``
"""
from __future__ import annotations

import sys

from train_fast_aip_ppo import main as ppo_main


def _append_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _append_default("--stage-schedule", "gun_bucket_curriculum")
    _append_default("--target-maneuver", "gun_curriculum")
    raise SystemExit(ppo_main())
