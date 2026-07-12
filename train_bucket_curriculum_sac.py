"""Run SAC on the bucketized gun curriculum.

Wrapper defaults:

* ``--stage-schedule gun_bucket_curriculum``
* ``--target-maneuver gun_curriculum``
* ``--reset-replay-on-stage``
"""
from __future__ import annotations

import sys

from train_fast_aip_sac import main as sac_main


def _append_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def _append_flag_default(flag: str) -> None:
    if flag not in sys.argv:
        sys.argv.append(flag)


if __name__ == "__main__":
    _append_default("--stage-schedule", "gun_bucket_curriculum")
    _append_default("--target-maneuver", "gun_curriculum")
    _append_flag_default("--reset-replay-on-stage")
    raise SystemExit(sac_main())
