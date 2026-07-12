#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

VARIANT="${VARIANT:-sac_lstm}"
SEED="${SEED:-7}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-$SEED}"
TARGET_MANEUVER="${TARGET_MANEUVER:-random_loiter}"
NUM_ENVS="${NUM_ENVS:-2048}"
if [[ "$VARIANT" == "sac_lstm" ]]; then
  HORIZON="${HORIZON:-16}"
  BATCH_SEQUENCES="${BATCH_SEQUENCES:-512}"
else
  HORIZON="${HORIZON:-32}"
  BATCH_SEQUENCES="${BATCH_SEQUENCES:-2048}"
fi
UPDATES_PER_ROLLOUT="${UPDATES_PER_ROLLOUT:-16}"
LEARNING_STARTS="${LEARNING_STARTS:-65536}"
REPLAY_CHUNKS="${REPLAY_CHUNKS:-64}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
ALPHA_LR="${ALPHA_LR:-1e-4}"
MIN_VALID_FRACTION="${MIN_VALID_FRACTION:-0.05}"
MAX_UPDATES="${MAX_UPDATES:-20000}"
ADVANCE_WINDOW="${ADVANCE_WINDOW:-8}"
ADVANCE_PATIENCE="${ADVANCE_PATIENCE:-3}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
START_STAGE="${START_STAGE:-0}"
STAGE_SCHEDULE="${STAGE_SCHEDULE:-aip}"
if [[ -z "${STOP_STAGE:-}" ]]; then
  case "$STAGE_SCHEDULE" in
    kill_bridge|stage10_11_bridge|final_kill_bridge) STOP_STAGE="14" ;;
    gun_curriculum|gun|shooting|tight_wez) STOP_STAGE="19" ;;
    gun_bucket_curriculum|bucket_gun|axis_bucket|bucketized_gun) STOP_STAGE="14" ;;
    *) STOP_STAGE="11" ;;
  esac
fi
OUTPUT="${OUTPUT:-$ROOT/fast_aip_sac_runs}"
RESIDUAL="${RESIDUAL:-}"
RESIDUAL_GAIN="${RESIDUAL_GAIN:-0.3}"
RESIDUAL_RAMP_SECONDS="${RESIDUAL_RAMP_SECONDS:-5}"
RESUME="${RESUME:-}"
ACTION_MEAN_L2_COEF="${ACTION_MEAN_L2_COEF:-1e-4}"
RESET_REPLAY_ON_STAGE="${RESET_REPLAY_ON_STAGE:-0}"
RESUME_WEIGHTS_ONLY="${RESUME_WEIGHTS_ONLY:-0}"
RESET_ALPHA_ON_RESUME="${RESET_ALPHA_ON_RESUME:-0}"

if [[ ! -x "$PY" ]]; then
  echo "Python not found: $PY. Set PYTHON_BIN or recreate .venv." >&2
  exit 2
fi

"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('torch:', torch.__version__); print('gpu0:', torch.cuda.get_device_name(0))"
"$PY" "$ROOT/validate_fast_parallel_contract.py" \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --device cuda \
  --num-envs 64 \
  --stage-index "$START_STAGE" \
  --stage-schedule "$STAGE_SCHEDULE" \
  --target-maneuver "$TARGET_MANEUVER"

EXTRA=()
if [[ -n "$RESIDUAL" ]]; then
  EXTRA+=(--residual "$RESIDUAL" --residual-gain "$RESIDUAL_GAIN" --residual-ramp-seconds "$RESIDUAL_RAMP_SECONDS")
fi
if [[ -n "$RESUME" ]]; then
  EXTRA+=(--resume "$RESUME")
fi
case "${RESET_REPLAY_ON_STAGE,,}" in
  1|true|yes|on) EXTRA+=(--reset-replay-on-stage) ;;
esac
case "${RESUME_WEIGHTS_ONLY,,}" in
  1|true|yes|on) EXTRA+=(--resume-weights-only) ;;
esac
case "${RESET_ALPHA_ON_RESUME,,}" in
  1|true|yes|on) EXTRA+=(--reset-alpha-on-resume) ;;
esac

"$PY" "$ROOT/train_fast_aip_sac.py" \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --device cuda \
  --num-envs "$NUM_ENVS" \
  --horizon "$HORIZON" \
  --batch-sequences "$BATCH_SEQUENCES" \
  --updates-per-rollout "$UPDATES_PER_ROLLOUT" \
  --learning-starts "$LEARNING_STARTS" \
  --replay-chunks "$REPLAY_CHUNKS" \
  --actor-lr "$ACTOR_LR" \
  --critic-lr "$CRITIC_LR" \
  --alpha-lr "$ALPHA_LR" \
  --min-valid-fraction "$MIN_VALID_FRACTION" \
  --max-updates-per-stage "$MAX_UPDATES" \
  --advance-window "$ADVANCE_WINDOW" \
  --advance-patience "$ADVANCE_PATIENCE" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --start-stage "$START_STAGE" \
  --stop-stage "$STOP_STAGE" \
  --stage-schedule "$STAGE_SCHEDULE" \
  --action-mean-l2-coef "$ACTION_MEAN_L2_COEF" \
  --target-maneuver "$TARGET_MANEUVER" \
  --output "$OUTPUT" \
  "${EXTRA[@]}"
