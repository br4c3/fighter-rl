# Training HTML Report

학습 trainer와 독립된 `tools/render_training_report.py`로 PPO/SAC 실행 결과를
단일 HTML 대시보드로 만들 수 있다. 외부 Python 또는 JavaScript 패키지는
필요하지 않다.

## 완료된 run 리포트

```bash
.venv/bin/python tools/render_training_report.py \
  fast_aip_ppo_runs/bucket_gun_ppo_lstm_YYYYMMDD_HHMMSS
```

결과는 해당 run의 `training_report.html`에 저장된다. 상위 output 폴더를
넘기면 그 아래에서 가장 최근에 수정된 `metrics.jsonl` run을 자동 선택한다.

```bash
.venv/bin/python tools/render_training_report.py fast_aip_ppo_runs
```

## 학습 중 자동 갱신

```bash
.venv/bin/python tools/render_training_report.py fast_aip_ppo_runs --watch 10
```

브라우저에서 생성된 `training_report.html`을 열어두면 10초마다 파일과 화면이
갱신된다. 종료는 `Ctrl-C`를 사용한다.

## 출력 경로와 데이터 크기 조정

```bash
.venv/bin/python tools/render_training_report.py RUN_DIR \
  --output /tmp/ppo_report.html \
  --max-points 10000
```

리포트는 다음 파일을 읽는다.

- 필수: `metrics.jsonl`
- 선택: `config.json`, `curriculum_state.json`, `stage_snapshot.json`

PPO에서는 optimizer 세부 지표와 actuator action을 표시한다. SAC에서는 Q loss,
actor loss, alpha뿐 아니라 entropy proxy와 policy log standard deviation을 함께
표시한다. 새로 생성한 run은 에피소드 마지막 25%의 track/ATA/WEZ/reward와
damage, dwell, aim, track, closure, action-rate 보상 성분도 별도 차트로 보여준다.
이전 run에는 저장되지 않은 항목이므로 해당 차트가 `No data`로 표시되는 것이
정상이다. `--demo`를 사용하면 실제 run 없이 화면 구성을 확인할 수 있다.

```bash
.venv/bin/python tools/render_training_report.py --demo --output /tmp/fighter_rl_demo.html
```
