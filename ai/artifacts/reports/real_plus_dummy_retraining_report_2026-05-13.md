# 실데이터 + 더미 통합 재학습 리포트

- 작성일: 2026-05-13
- 브랜치: `dain_lightgbm`
- 목적: 실데이터 + 더미 데이터를 합쳐 LightGBM 모델을 처음부터 재학습하고 성능을 검증

## 1. 데이터 구성

원본:

- 실데이터: `ai/artifacts/datasets/lgbm_training_dataset.csv` (898 rows)
- 더미데이터: `ai/artifacts/datasets/dummy_2000/lgbm_training_dataset_dummy_2000.csv` (2,000 rows)

통합 산출물:

- `ai/artifacts/datasets/real_plus_dummy/lgbm_training_dataset_real_plus_dummy.csv`

통합 분포:

- 총 rows: **2,898**
- source 비중:
  - dummy: 2,000
  - real: 898
- `is_meaningful_label`:
  - 0: 1,079
  - 1: 1,819
- positive event_type:
  - daily_noise: 664
  - repeated_vibration: 594
  - impact_noise: 561

## 2. 검증 분할 전략

통합 데이터는 timestamp 기준으로 정렬해 시계열 분할을 사용했다.

분할 진단:

- `train_ratio=0.8`: valid가 dummy만 포함되어 평가 왜곡 위험
- `train_ratio=0.7`: valid real 비중 매우 낮음(29 rows)
- `train_ratio=0.6`: valid에 real 310 + dummy 850 포함

최종 선택:

- **train_ratio=0.6**
- 이유: 검증셋에 실데이터를 충분히 포함하기 위함

## 3. 학습 설정

공통:

- 모델: `lightgbm.LGBMClassifier`
- `n_estimators=500`
- `learning_rate=0.05`
- `num_leaves=31`
- `class_weight=balanced`
- `early_stopping_rounds=50`
- `seed=42`

과업:

1. `is_meaningful` binary
2. `event_type` multiclass (`daily_noise,repeated_vibration,impact_noise`)

## 4. 통합 검증 결과 (train_ratio=0.6)

### 4.1 is_meaningful

- accuracy: **0.6948**
- precision: **0.6325**
- recall: **0.9715**
- f1: **0.7662**
- average_precision: **0.6086**
- roc_auc: **0.6226**

confusion matrix (`[0,1]`):

- `[[226, 337], [17, 580]]`

해석:

- positive 재현율은 높으나(`recall`), negative를 positive로 과다 예측하는 경향이 남아 있음
- 운영 관점에서 과탐 부담을 추가로 관리해야 함

### 4.2 event_type

- accuracy: **0.9715**
- macro_f1: **0.9688**
- weighted_f1: **0.9715**
- roc_auc_ovr: **0.9975**

class별:

- daily_noise: precision 0.9588 / recall 0.9422 / f1 0.9504
- repeated_vibration: precision 0.9487 / recall 0.9635 / f1 0.9561
- impact_noise: precision 1.0000 / recall 1.0000 / f1 1.0000

confusion matrix (`[daily_noise, repeated_vibration, impact_noise]`):

- `[[163, 10, 0], [7, 185, 0], [0, 0, 232]]`

해석:

- event_type 분류는 전반적으로 안정적
- impact_noise는 더미 라벨 증강 효과로 분리 성능이 크게 개선됨

## 5. 실데이터 전용 평가 (참고)

혼합 학습 모델을 실데이터셋(898 rows) 전체에 적용한 참고 지표:

### 5.1 is_meaningful (real-only)

- accuracy: **0.6570**
- precision: **0.5763**
- recall: **1.0000**
- f1: **0.7312**
- roc_auc: **0.6567**

해석:

- 실데이터에서도 recall 우선 성향이 유지됨
- precision 개선을 위한 threshold/negative 케이스 보강이 필요

### 5.2 event_type (real-only)

- accuracy: **1.0000** (419 rows)
- macro_f1: **1.0000**

주의:

- real-only event_type는 분포가 단순하며(`impact_noise` 1건), 본 수치가 일반화 성능을 완전히 보장하지는 않음

## 6. 결론

1. 요청한 "실데이터 + 더미 통합 재학습"은 완료되었고 모델 산출물 생성도 정상이다.
2. `event_type` 성능은 크게 개선되었으며 impact 클래스 분류 가능 상태를 확보했다.
3. `is_meaningful`는 여전히 high-recall/low-precision 특성이 있어 운영 적용 시 과탐 제어가 필요하다.

## 7. 권장 후속 작업

1. `AI_LGBM_MIN_CONFIDENCE` 임계치 스윕(예: 0.5~0.8)으로 precision/recall 균형점 선정
2. hard-negative 실측 라벨 수집 확대
3. `hybrid + shadow` 운영으로 rule vs lgbm 불일치 사례 축적
4. 주기적 재학습 시 real:dummy 비율 실험(예: 90:10, 80:20)

## 8. 산출물 경로

- 통합 데이터셋:
  - `ai/artifacts/datasets/real_plus_dummy/lgbm_training_dataset_real_plus_dummy.csv`
  - `ai/artifacts/datasets/real_plus_dummy/merge_summary.json`
- 모델:
  - `ai/artifacts/models_real_plus_dummy/is_meaningful_model.txt`
  - `ai/artifacts/models_real_plus_dummy/event_type_model.txt`
  - `ai/artifacts/models_real_plus_dummy/event_type_label_map.json`
- 지표:
  - `ai/artifacts/models_real_plus_dummy/is_meaningful_metrics.json`
  - `ai/artifacts/models_real_plus_dummy/event_type_metrics.json`
  - `ai/artifacts/models_real_plus_dummy/evaluation_metrics.json`
  - `ai/artifacts/models_real_plus_dummy/confusion_matrices.json`
  - `ai/artifacts/models_real_plus_dummy/real_only_evaluation.json`
