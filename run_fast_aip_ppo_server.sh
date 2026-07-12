#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
export CONFIG="${CONFIG:-$ROOT/configs/ppo_lstm.json}"

if [[ ! -x "$PY" ]]; then
  echo "Python not found: $PY. Create .venv or set PYTHON_BIN." >&2
  exit 2
fi

"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('torch:', torch.__version__); print('gpu0:', torch.cuda.get_device_name(0))"
"$PY" -m fighter_rl.training.ppo
