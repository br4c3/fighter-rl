# Reward Guide

이 문서는 현재 기본 학습 설정인 `configs/ppo_lstm.json`의
`stage_schedule = "gun_bucket_curriculum"` 기준이다.

리워드 계수는 `fighter_rl/training/stages.py`에서 stage별로 정하고,
실제 보상 계산은 `fighter_rl/envs/loiter.py`의
`CompetitionLoiterCurriculumEnv.step()`에서 수행한다.

## 전체 구조

| 구간 | Stage | 목적 | Reward 성격 |
| --- | --- | --- | --- |
| Trail hold | 0-4 | 적 후방 가까운 위치를 안정적으로 유지 | 위치 추적 중심, damage 보상 없음 |
| Nose bridge | 5 | 후방 추적에서 nose-on/사격 자세로 연결 | 추적 + 조준 + 약한 damage 보상 |
| Gun WEZ | 6-9 | WEZ 안에서 조준/체류/피해 누적 | damage, dwell, phi 개선 중심 |
| Close BFM entry | 10-12 | 가까운 거리에서 각도/3-9 line 관리 | damage 중심, phi 보상은 점진 감소 |
| BT pressure | 13-14 | 가까운 거리에서 BT/defensive target 대응 | damage 중심, 복잡한 상대 기동 대응 |

현재 close-range curriculum 적용으로 모든 stage의 초기 거리는 최대
`1500m` 안쪽으로 제한된다. 후반부도 먼 BFM 시작이 아니라 가까운 교전
상태에서 시작한다.

## 보상 항목

| 항목 | 코드 key | 의미 |
| --- | --- | --- |
| Step penalty | `step_penalty` | 매 decision step마다 주는 작은 시간 비용 |
| Damage reward | `damage_scale` | 적에게 준 damage 보상 |
| Own damage penalty | `own_damage_scale` | 내가 받은 damage 패널티 |
| WEZ dwell | `dwell_scale` | WEZ 안에 연속으로 머무는 보상 |
| Phi shaping | `phi_scale` | 거리/ATA/AA 기반 gun quality 개선량 보상 |
| Aim shaping | `aim_scale` | 조준 각도와 목표 거리 근접도 보상 |
| Track reward | `track_scale` | 목표 trail 위치를 따라가는 보상 |
| Inner penalty | `inner_penalty_scale` | 너무 가까운 거리 진입 패널티 |
| Red WEZ penalty | `red_wez_penalty` | 적의 WEZ 안에 들어간 패널티 |
| Bad 3-9 penalty | `bad_3_9_penalty` | 좋지 않은 3-9 line crossing 패널티 |
| Low altitude penalty | `low_altitude_penalty` | 저고도 패널티 |
| Terminal reward | `win_reward`, `loss_reward`, `draw_reward` | 승패/무승부 종료 보상 |

## Stage별 현재 보상

| Stage | 이름 | 초기 거리 | 핵심 보상 |
| --- | --- | ---: | --- |
| 0 | `T0_anchor_trail_hold` | 780-940m | `track_scale=0.130`, `track_trail_m=850`, damage 없음 |
| 1 | `T1_range_closure_trail` | 730-1050m | `track_scale=0.125`, closure/range 변화 적응 |
| 2 | `T2_angular_trail` | 820-950m | `track_scale=0.120`, `phi_scale=0.006`, 작은 ATA/AA 교정 |
| 3 | `T3_weak_turn_trail` | 780-1150m | `track_scale=0.115`, weak turn target 추적 |
| 4 | `T4_maneuver_trail` | 850-1400m | `track_scale=0.110`, jink/turn target 추적 |
| 5 | `A0_nose_on_trail_bridge` | 500-930m | `track_scale=0.070`, `aim_scale=0.055`, `damage_scale=6` |
| 6 | `G0_in_wez_hold` | 320-860m | `damage_scale=12`, `dwell_scale=0.03`, `phi_scale=0.110`, `aim_scale=0.025` |
| 7 | `G1_wez_boundary_hold` | 260-940m | WEZ edge 유지, `phi_scale=0.090`, `aim_scale=0.015` |
| 8 | `G2_small_reacquire` | 450-1100m | 작은 재획득, `phi_scale=0.075`, `aim_scale=0.008` |
| 9 | `G3_moving_gun_track` | 450-1300m | 움직이는 target gun track, `phi_scale=0.058` |
| 10 | `E0_outer_rear_entry` | 800-1500m | 가까운 rear entry, `phi_scale=0.040` |
| 11 | `E1_side_rear_entry` | 900-1500m | side/rear 각도 관리, `phi_scale=0.028` |
| 12 | `E2_three_nine_management` | 900-1500m | 3-9 line 관리, `phi_scale=0.018` |
| 13 | `B0_bt_intro` | 1000-1500m | BT target intro, `phi_scale=0.010` |
| 14 | `B1_short_bfm` | 1100-1500m | short BFM, `phi_scale=0.006` |

공통적으로 stage 6 이후 기본 gun reward는 다음 값을 쓴다.

Stage 5에서 gun reward로 갑자기 전환할 때 조준 자세가 무너지는 것을
막기 위해 stage 6-8에는 작은 aim shaping을 남기고 점진적으로 제거한다.
Stage 9부터는 `aim_scale=0`인 기본 gun reward만 사용한다.

| Key | Value |
| --- | ---: |
| `step_penalty` | `-0.002` |
| `damage_scale` | `12.0` |
| `own_damage_scale` | `18.0` |
| `dwell_scale` | `0.03` |
| `inner_penalty_scale` | `0.65` |
| `red_wez_penalty` | `0.08` |
| `low_altitude_penalty` | `2.0` |
| `win_reward` | `8.0` |
| `loss_reward` | `-10.0` |
| `draw_reward` | `-2.0` |

## 수정 위치

| 바꾸고 싶은 것 | 수정 위치 |
| --- | --- |
| stage별 reward 계수 | `fighter_rl/training/stages.py` |
| stage별 거리/각도/bucket | `fighter_rl/training/stages.py` |
| reward 계산식 자체 | `fighter_rl/envs/loiter.py` |
| 로그에 남는 metric | `fighter_rl/envs/loiter.py` |
| curriculum 통과 조건 | `fighter_rl/training/stages.py` |

계수만 바꿀 때는 `stages.py`를 수정하는 편이 낫다. 예를 들어 damage를
더 강하게 주려면 `_gun_reward()`의 `damage_scale` 또는 특정 stage의
`reward_override`를 바꾼다.

보상 항목을 새로 추가하려면 `loiter.py`에서 `reward += ...` 또는
`reward -= ...` 계산을 추가하고, episode summary에 필요한 metric을
같이 기록한다.
