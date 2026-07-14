# 학습 코드 구성도

Python 코드는 `fighter_rl/` 패키지 아래에 역할별로 정리되어 있다. 루트에는
실행 스크립트, JSON 설정, 문서, 최소 runtime XML만 둔다.

## 전체 흐름

```text
configs/*.json
  -> run_fast_aip_*_server.ipynb
  -> fighter_rl.training.ppo 또는 fighter_rl.training.sac
  -> fighter_rl.training.stages
  -> fighter_rl.envs.loiter
  -> fighter_rl.sim.neuralplane
  -> fighter_rl.models.ppo 또는 fighter_rl.models.sac
  -> fighter_rl.utils.experiment_record
```

## 폴더 구조

| 경로 | 역할 |
| --- | --- |
| `configs/` | 학습 설정 JSON. 코드 수정 없이 여기서 batch, LR, stage, output 등을 바꾼다. |
| `fighter_rl/training/` | PPO/SAC trainer, curriculum stage, preflight check. |
| `fighter_rl/envs/` | RL 환경, dogfight geometry, target/opponent policy. |
| `fighter_rl/models/` | PPO/SAC network 정의. |
| `fighter_rl/sim/neuralplane/` | GPU batch 비행 surrogate. |
| `fighter_rl/utils/` | config loading, experiment manifest/log 기록. |
| `stock_runtime/` | surrogate가 읽는 F-16/F100 XML runtime 데이터. |
| `REWARD_GUIDE.md` | 현재 reward 구성과 stage별 보상 의도. |

## 실행 파일

| 경로 | 역할 |
| --- | --- |
| `run_fast_aip_ppo_server.ipynb` | Jupyter에서 config/device를 확인하고 `fighter_rl.training.ppo`를 실행한다. |
| `run_fast_aip_sac_server.ipynb` | Jupyter에서 config/device를 확인하고 `fighter_rl.training.sac`를 실행한다. |
| `fighter_rl/training/launcher.py` | 노트북 launcher 공통 로직. config 경로, device, CUDA 확인을 담당한다. |
| `configs/ppo_lstm.json` | PPO LSTM 기본 설정. |
| `configs/sac_lstm.json` | SAC LSTM 기본 설정. |
| `configs/sac_lstm_micro.json` | 29-stage micro gun curriculum과 이름 기반 block reset 설정. |
| `fighter_rl/utils/config.py` | JSON config를 읽어 학습 설정 객체를 만든다. |

## 학습 본체

| 경로 | 역할 |
| --- | --- |
| `fighter_rl/training/ppo.py` | PPO 학습 루프. rollout, GAE, PPO update, checkpoint, stage advance를 담당한다. |
| `fighter_rl/training/sac.py` | SAC 학습 루프. rollout, sequence replay, actor/critic update, checkpoint, stage advance를 담당한다. |
| `fighter_rl/training/stages.py` | curriculum stage 정의. 초기 조건 bucket, target 행동 mix, reward gate, stage 통과 조건이 여기 있다. |
| `fighter_rl/training/preflight.py` | config와 모델/env shape이 맞는지 확인하는 가벼운 사전 점검 코드. |

## 환경 코드

| 경로 | 역할 |
| --- | --- |
| `fighter_rl/envs/loiter.py` | RL 환경 본체. ownship/target aircraft를 step하고 reward, done, episode summary를 만든다. |
| `fighter_rl/envs/batch.py` | batch dogfight geometry helper. 상대 거리, 각도, WEZ 관련 값, observation 구성 요소를 계산한다. |
| `fighter_rl/envs/bt_policy.py` | target aircraft가 쓸 rule/behavior-tree 스타일 기동 정책. |
| `fighter_rl/envs/maneuvers.py` | 기동 함수 registry. 현재 기본 학습 경로에서는 보조 유틸에 가깝다. |

## 모델 코드

| 경로 | 역할 | variant |
| --- | --- | --- |
| `fighter_rl/models/ppo.py` | PPO policy/value network와 PPO action distribution helper. | `ppo_lstm`, `ppo_mlp` |
| `fighter_rl/models/sac.py` | SAC actor/critic network, action sampling, target-network update helper. | `sac_lstm`, `sac_mlp` |

LSTM variant는 다음과 같다.

| Variant | Observation | Recurrent 구조 |
| --- | --- | --- |
| `ppo_lstm` | 20차원: tactical 16(closure 포함) + previous action 4 | Encoder `128,128` -> LSTM `64` -> policy/value head |
| `sac_lstm` | 20차원: tactical 16(closure 포함) + previous action 4 | Actor LSTM `64`, critic LSTM `64` |

MLP variant인 `ppo_mlp`, `sac_mlp`는 4 frame stack을 써서 observation이 80차원이다.

PPO 로그는 total loss 외에도 policy/value loss, entropy, approximate KL,
clip fraction, explained variance, gradient norm과 실제 actuator command의 평균,
표준편차, saturation rate, delta action을 기록한다.

## 비행 Surrogate

| 경로 | 역할 |
| --- | --- |
| `fighter_rl/sim/neuralplane/env.py` | aircraft 1대의 batched simulation wrapper. |
| `fighter_rl/sim/neuralplane/dynamics.py` | rigid-body dynamics 적분. |
| `fighter_rl/sim/neuralplane/fcs.py` | F-16 flight-control-system 로직. |
| `fighter_rl/sim/neuralplane/engine.py` | F100 엔진 table과 spool 모델. |
| `fighter_rl/sim/neuralplane/xml_aero.py` | F-16 XML 공력 table을 읽고 aerodynamic coefficient를 계산한다. |
| `fighter_rl/sim/neuralplane/atmosphere.py` | atmosphere, airspeed, gravity 유틸. |
| `fighter_rl/sim/neuralplane/eci.py` | 좌표 변환과 Earth-frame 계산. |

## 기록/로그

| 경로 | 역할 |
| --- | --- |
| `fighter_rl/utils/experiment_record.py` | 실행 manifest와 metrics metadata를 저장한다. checkpoint가 어떤 code/config/stage에서 나왔는지 추적하는 용도다. |

## 자주 수정할 파일

| 하고 싶은 일 | 수정할 파일 |
| --- | --- |
| PPO batch size, learning rate, output, resume path 변경 | `configs/ppo_lstm.json` |
| SAC replay/update 설정, learning rate, output, resume path 변경 | `configs/sac_lstm.json` |
| curriculum gate나 stage 초기 조건 변경 | `fighter_rl/training/stages.py` |
| reward 또는 done 조건 변경 | `fighter_rl/envs/loiter.py` |
| PPO network 구조 변경 | `fighter_rl/models/ppo.py` |
| SAC actor/critic 구조 변경 | `fighter_rl/models/sac.py` |
