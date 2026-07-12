# Fighter RL Training

Minimal fast GPU training setup for the AIP dogfight policy.

학습 코드 파일별 역할은 `TRAINING_CODE_MAP.md`에 정리되어 있다.

## What Is Here

- `fighter_rl/training/ppo.py` - PPO trainer.
- `fighter_rl/training/sac.py` - SAC trainer.
- `run_fast_aip_ppo_server.sh` - PPO launch wrapper.
- `run_fast_aip_sac_server.sh` - SAC launch wrapper.
- `configs/ppo_lstm.json` - PPO training config.
- `configs/sac_lstm.json` - SAC training config.
- `fighter_rl/models/ppo.py` - PPO MLP/LSTM policy profiles.
- `fighter_rl/models/sac.py` - SAC MLP/LSTM actor/critic profiles.
- `fighter_rl/envs/loiter.py` - batched gun curriculum environment.
- `fighter_rl/sim/neuralplane/` - GPU-batched F-16 surrogate.
- `fighter_rl/training/stages.py` - curriculum stages and gates.
- `stock_runtime/` - aircraft/engine XML used by the trainer.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## PPO LSTM

```bash
bash run_fast_aip_ppo_server.sh
```

Edit `configs/ppo_lstm.json` to change PPO settings.

## SAC LSTM

```bash
bash run_fast_aip_sac_server.sh
```

Edit `configs/sac_lstm.json` to change SAC settings.

If two GPUs are available, run PPO with `CUDA_VISIBLE_DEVICES=0` and SAC with
`CUDA_VISIBLE_DEVICES=1`.

```bash
CUDA_VISIBLE_DEVICES=0 bash run_fast_aip_ppo_server.sh
CUDA_VISIBLE_DEVICES=1 bash run_fast_aip_sac_server.sh
```

To use another config file:

```bash
CONFIG=configs/ppo_lstm.json bash run_fast_aip_ppo_server.sh
```

## Variants

Supported trainer variants:

- PPO: `ppo_lstm`, `ppo_mlp`
- SAC: `sac_lstm`, `sac_mlp`

Current default choice is to start with `ppo_lstm`, then compare against
`sac_lstm` on curriculum progress and damage metrics.
