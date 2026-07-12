import importlib
import json
import os
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[2]


def config_path(default_config):
    root = project_root()
    raw_path = os.environ.get("CONFIG") or os.environ.get("TRAINING_CONFIG") or default_config
    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    os.environ["CONFIG"] = str(path)

    return path


def config_device(path):
    data = json.loads(path.read_text())

    if not isinstance(data, dict):
        raise ValueError(f"Training config must be a JSON object: {path}")

    return str(data.get("device", "cuda"))


def check_torch_device(device):
    import torch

    if device == "cpu":
        print(f"torch: {torch.__version__}", flush=True)
        print("device: cpu", flush=True)

        return

    if not device.startswith("cuda"):
        raise ValueError(f"Unsupported training device for launcher: {device}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    index = torch.device(device).index
    index = 0 if index is None else index

    print(f"torch: {torch.__version__}", flush=True)
    print(f"gpu{index}: {torch.cuda.get_device_name(index)}", flush=True)


def launch(module_name, default_config):
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    path = config_path(default_config)
    device = config_device(path)

    print(f"config: {path}", flush=True)
    check_torch_device(device)

    module = importlib.import_module(module_name)

    return module.main()
