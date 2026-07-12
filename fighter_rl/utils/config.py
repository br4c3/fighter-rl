import json
import os
from pathlib import Path
from types import SimpleNamespace

PATH_KEYS = {"resume", "output"}


def load_training_config(default_path):
    path = Path(os.environ.get("CONFIG") or os.environ.get("TRAINING_CONFIG") or default_path)

    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a JSON object: {path}")

    values = {"config_path": path}

    for key, value in payload.items():
        if key == "comment":
            continue

        if value is not None and key in PATH_KEYS:
            value = Path(value)
        values[key] = value
    return SimpleNamespace(**values)


def config_dict(config):
    return dict(vars(config))
