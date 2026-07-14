# Fighter RL Training

Minimal fast GPU training setup for the AIP dogfight policy.

학습 코드 파일별 역할은 `TRAINING_CODE_MAP.md`에 정리되어 있다.
현재 reward/curriculum 구성은 `REWARD_GUIDE.md`에 정리되어 있다.

## What Is Here

- `fighter_rl/training/ppo.py` - PPO trainer.
- `fighter_rl/training/sac.py` - SAC trainer.
- `run_fast_aip_ppo_server.ipynb` - PPO Jupyter launcher.
- `run_fast_aip_sac_server.ipynb` - SAC Jupyter launcher.
- `configs/ppo_lstm.json` - PPO training config.
- `configs/sac_lstm.json` - SAC training config.
- `configs/sac_lstm_micro.json` - 29-stage fine-grained SAC gun curriculum.
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

Edit `configs/ppo_lstm.json` to change PPO settings.

To run the trainer interactively in Jupyter, open
`run_fast_aip_ppo_server.ipynb` with the project's Python environment and run
the cells from top to bottom. The training cell streams logs and can be stopped
with **Interrupt Kernel**.

```bash
.venv/bin/python -m jupyter lab run_fast_aip_ppo_server.ipynb
```

## SAC LSTM

Edit `configs/sac_lstm.json` to change SAC settings.

To run SAC interactively in Jupyter, open `run_fast_aip_sac_server.ipynb` and
run the cells from top to bottom.

```bash
.venv/bin/python -m jupyter lab run_fast_aip_sac_server.ipynb
```

If two GPUs are available, run PPO with `CUDA_VISIBLE_DEVICES=0` and SAC with
`CUDA_VISIBLE_DEVICES=1`.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m jupyter lab run_fast_aip_ppo_server.ipynb
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m jupyter lab run_fast_aip_sac_server.ipynb
```

To use another config file, change `CONFIG_PATH` in the notebook setup cell.

For the fine-grained gun curriculum, point `CONFIG_PATH` at
`configs/sac_lstm_micro.json`. It preserves replay within each A/G/E/B block
and resets replay only when the reward regime changes.

## Variants

Supported trainer variants:

- PPO: `ppo_lstm`, `ppo_mlp`
- SAC: `sac_lstm`, `sac_mlp`

Current default choice is to start with `ppo_lstm`, then compare against
`sac_lstm` on curriculum progress and damage metrics.
