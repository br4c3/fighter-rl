#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIP_NO_CACHE_DIR=1

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

RESET_VENV="${RESET_VENV:-0}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_JUPYTER_KERNEL="${INSTALL_JUPYTER_KERNEL:-1}"
KERNEL_NAME="${KERNEL_NAME:-aip-neuralplane}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-AIP NeuralPlane (.venv)}"

if [[ -f "$ROOT/.venv/pyvenv.cfg" ]] && grep -qiE 'C:\\|Windows|pythoncore-[0-9]' "$ROOT/.venv/pyvenv.cfg"; then
  if [[ "$RESET_VENV" != "1" ]]; then
    echo "Existing .venv looks like a Windows/local virtualenv and cannot run on Linux:" >&2
    sed -n '1,8p' "$ROOT/.venv/pyvenv.cfg" >&2 || true
    echo "Re-run with RESET_VENV=1 bash $ROOT/setup_linux.sh" >&2
    exit 2
  fi
fi

if [[ "$RESET_VENV" == "1" ]]; then
  rm -rf "$ROOT/.venv"
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi

"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel

# requirements.txt keeps "torch>=2.0" for generic local installs.  On the
# server we filter it out and install PyTorch explicitly below.  Otherwise pip
# can pick a wheel whose CUDA runtime is newer than the server driver, making
# torch.cuda.is_available() false even when GPUs exist.
REQ_NO_TORCH="$(mktemp)"
trap 'rm -f "$REQ_NO_TORCH"' EXIT
grep -viE '^[[:space:]]*torch([<=>[:space:]]|$)' "$ROOT/requirements.txt" > "$REQ_NO_TORCH"
"$ROOT/.venv/bin/python" -m pip install -r "$REQ_NO_TORCH"

"$ROOT/.venv/bin/python" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
if [[ "$INSTALL_JUPYTER_KERNEL" == "1" ]]; then
  "$ROOT/.venv/bin/python" -m pip install ipykernel
  "$ROOT/.venv/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"
fi
"$ROOT/.venv/bin/python" -m py_compile \
  "$ROOT/competition_neuralplane/env.py" \
  "$ROOT/aip_neuralplane_rllib_env.py" \
  "$ROOT/train_aip_neuralplane_bundle.py"

if [[ "${BUILD_EXACT_CORE:-0}" == "1" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python" bash "$ROOT/build_exact_core_linux.sh"
else
  echo "Exact/JSBSim teacher validation needs lib/libExactF16Core.so."
  echo "Build it when needed with:"
  echo "  PYTHON_BIN=$ROOT/.venv/bin/python bash $ROOT/build_exact_core_linux.sh"
fi

"$ROOT/.venv/bin/python" "$ROOT/check_server_environment.py" --require-cuda
echo "Bundle pipeline:"
echo "  VARIANT=ppo_mlp NUM_ENVS=4096 PYTHON_BIN=$ROOT/.venv/bin/python bash $ROOT/run_aip_neuralplane_bundle_server.sh"
