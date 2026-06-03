# LightGBM 학습 가이드

이 디렉터리는 쿵로그 AI 분류기를 LightGBM으로 학습/평가하기 위한 스크립트를 제공합니다.

## 준비물

- Python 패키지 설치

```powershell
pip install -r requirements.txt
pip install -r requirements-train.txt
```

- 입력 CSV 2개
1. 센서 원본: `raw_sensor_readings` (`all_sensor_readings.csv`)
2. 이벤트 라벨: `noise_events` (`all_noise_events.csv`)

## 1) 학습 데이터셋 생성

`build_lgbm_dataset.py`는 센서 행을 기준으로 라벨을 붙여 학습용 CSV를 만듭니다.

조인 키:

- 센서: `sensor_id + sensor_timestamp`
- 이벤트: `sensor_id + started_at`

실행:

```powershell
python -m ai.training.build_lgbm_dataset `
  --sensor-csv ".\data_exports\all_sensor_readings.csv" `
  --noise-csv ".\data_exports\all_noise_events.csv" `
  --out-csv ".\ai\artifacts\datasets\lgbm_training_dataset.csv" `
  --out-summary ".\ai\artifacts\datasets\lgbm_training_summary.json"
```

생성 컬럼:

- `is_meaningful_label` (`0/1`)
- `event_type_label`
- `severity_label`
- `noise_event_id`
- `noise_event_started_at`

## 2) 의미 이벤트 분류 모델 학습 (`is_meaningful`)

```powershell
python -m ai.training.train_is_meaningful_lgbm `
  --dataset ".\ai\artifacts\datasets\lgbm_training_dataset.csv" `
  --out-dir ".\ai\artifacts\models"
```

주요 산출물:

- `is_meaningful_model.txt`
- `is_meaningful_metrics.json`
- `feature_spec.json`

## 3) 이벤트 타입 모델 학습 (`event_type`)

기본 클래스는 `daily_noise,repeated_vibration` 입니다.

```powershell
python -m ai.training.train_event_type_lgbm `
  --dataset ".\ai\artifacts\datasets\lgbm_training_dataset.csv" `
  --out-dir ".\ai\artifacts\models" `
  --classes "daily_noise,repeated_vibration"
```

주요 산출물:

- `event_type_model.txt`
- `event_type_metrics.json`
- `event_type_label_map.json`
- `feature_spec.json`

## 4) 통합 평가

```powershell
python -m ai.training.evaluate_lgbm `
  --dataset ".\ai\artifacts\datasets\lgbm_training_dataset.csv" `
  --model-dir ".\ai\artifacts\models" `
  --out-json ".\ai\artifacts\models\evaluation_metrics.json"
```

## 무결성 체크 포인트 (`lgbm_training_summary.json`)

- `output_equals_sensor_rows`: 출력 행 수가 센서 행 수와 같은지
- `positive_equals_matched`: positive 라벨 수가 조인 매칭 수와 같은지
- `orphan_noise_event_count`: 센서에 매칭되지 않은 noise event 수

## 자주 발생하는 오류

- `not enough meaningful labeled rows`: 라벨 데이터 부족 (의미 이벤트 행 수 확대 필요)
- `train split is missing one or more classes`: 시간 분할 후 특정 클래스 부재
- `missing LightGBM artifacts`: 모델 파일/레이블맵 경로 확인 필요
