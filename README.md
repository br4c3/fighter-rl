# Fighter RL Training

Minimal fast GPU training setup for the AIP dogfight policy.

## What Is Here

- `train_fast_aip_ppo.py` - PPO trainer.
- `train_fast_aip_sac.py` - SAC trainer.
- `run_fast_aip_ppo_server.sh` - PPO launch wrapper.
- `run_fast_aip_sac_server.sh` - SAC launch wrapper.
- `fast_aip_policy.py` - PPO MLP/LSTM policy profiles.
- `fast_aip_sac.py` - SAC MLP/LSTM actor/critic profiles.
- `competition_loiter_env.py` - batched gun curriculum environment.
- `competition_neuralplane/` - GPU-batched F-16 surrogate.
- `loiter_gpu_stages.py` - curriculum stages and gates.
- `stock_runtime/` - aircraft/engine XML used by the trainer.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## PPO LSTM

```bash
CUDA_VISIBLE_DEVICES=0 \
VARIANT=ppo_lstm \
STAGE_SCHEDULE=gun_bucket_curriculum \
TARGET_MANEUVER=gun_curriculum \
OUTPUT=fast_aip_ppo_runs/bucket_gun_ppo_lstm \
bash run_fast_aip_ppo_server.sh
```

## SAC LSTM

```bash
CUDA_VISIBLE_DEVICES=0 \
VARIANT=sac_lstm \
STAGE_SCHEDULE=gun_bucket_curriculum \
TARGET_MANEUVER=gun_curriculum \
OUTPUT=fast_aip_sac_runs/bucket_gun_sac_lstm \
bash run_fast_aip_sac_server.sh
```

If two GPUs are available, run PPO with `CUDA_VISIBLE_DEVICES=0` and SAC with
`CUDA_VISIBLE_DEVICES=1`.

## Variants

Supported trainer variants:

- PPO: `ppo_lstm`, `ppo_mlp`
- SAC: `sac_lstm`, `sac_mlp`

Current default choice is to start with `ppo_lstm`, then compare against
`sac_lstm` on curriculum progress and damage metrics.
