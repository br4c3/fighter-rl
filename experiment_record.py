"""Reproducibility helpers for fast AIP/NeuralPlane experiments."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
    except Exception:
        pass
    return value


def sha256_file(path: Path | str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {"path": ""}
    p = Path(path)
    out: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if p.is_file():
        out["size_bytes"] = p.stat().st_size
        out["sha256"] = sha256_file(p)
    return out


def stage_record(stage) -> dict[str, Any]:
    return {
        "index": int(stage.index),
        "name": stage.name,
        "decision_limit": int(stage.decision_limit),
        "max_engage_time": float(stage.max_engage_time),
        "step_ratio": int(stage.step_ratio),
        "source": str(stage.source),
        "source_sha256": sha256_file(stage.source),
        "ownship_randomization": _jsonable(stage.ownship_randomization),
        "target_randomization": _jsonable(stage.target_randomization),
        "wez": _jsonable(stage.wez),
        "reward": _jsonable(stage.reward),
        "advance_conditions": _jsonable(stage.advance_conditions),
    }


def selected_environment() -> dict[str, str]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "STAGE_SCHEDULE",
        "LOITER_STAGE_SCHEDULE",
        "VARIANT",
        "TARGET_MANEUVER",
        "NUM_ENVS",
        "ADVANCE_WINDOW",
        "ADVANCE_PATIENCE",
        "RESIDUAL",
        "RESIDUAL_GAIN",
        "RESIDUAL_RAMP_SECONDS",
        "RESUME",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def torch_record() -> dict[str, Any]:
    try:
        import torch

        out: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "initial_seed": int(torch.initial_seed()),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
            "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        }
        if torch.cuda.is_available():
            out["device_count"] = int(torch.cuda.device_count())
            out["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return out
    except Exception as exc:
        return {"error": repr(exc)}


def code_records(extra_files: list[Path | str] | None = None) -> list[dict[str, Any]]:
    files = [
        ROOT / "competition_loiter_env.py",
        ROOT / "competition_neuralplane" / "env.py",
        ROOT / "loiter_gpu_stages.py",
        ROOT / "fast_aip_policy.py",
        ROOT / "fast_aip_sac.py",
        ROOT / "bt_policy.py",
        ROOT / "experiment_record.py",
    ]
    if extra_files:
        files.extend(Path(p) for p in extra_files)
    seen: set[str] = set()
    out = []
    for path in files:
        key = str(Path(path).resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(file_record(path))
    return out


def write_experiment_manifest(
    run_dir: Path,
    *,
    trainer: str,
    args,
    profile,
    stages,
    extra_code_files: list[Path | str] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "trainer": trainer,
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "torch": torch_record(),
        "environment": selected_environment(),
        "args": _jsonable(vars(args)),
        "profile": _jsonable(profile.as_metadata()),
        "artifacts": {
            "resume": file_record(getattr(args, "resume", None)),
            "residual": file_record(getattr(args, "residual", None)),
        },
        "code": code_records(extra_code_files),
        "stage_count": len(stages),
        "stage_indices": [int(s.index) for s in stages],
    }
    stage_snapshot = [stage_record(stage) for stage in stages]
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "stage_snapshot.json").write_text(
        json.dumps(stage_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True))
        f.write("\n")
