# LightGBM Dataset Builder

`build_lgbm_dataset.py` joins exported tables for model training:

- sensor table: `raw_sensor_readings` (`all_sensor_readings.csv`)
- label table: `noise_events` (`all_noise_events.csv`)

Join key:

- sensor: `sensor_id + sensor_timestamp`
- noise: `sensor_id + started_at`

Each output row keeps one sensor record and appends labels:

- `is_meaningful_label` (`0/1`)
- `event_type_label`
- `severity_label`
- `noise_event_id`
- `noise_event_started_at`

## Usage

```powershell
python -m ai.training.build_lgbm_dataset `
  --sensor-csv ".\data_exports\all_sensor_readings.csv" `
  --noise-csv ".\data_exports\all_noise_events.csv" `
  --out-csv ".\ai\artifacts\datasets\lgbm_training_dataset.csv" `
  --out-summary ".\ai\artifacts\datasets\lgbm_training_summary.json"
```

## Train: is_meaningful

```powershell
python -m ai.training.train_is_meaningful_lgbm `
  --dataset ".\ai\artifacts\datasets\lgbm_training_dataset.csv" `
  --out-dir ".\ai\artifacts\models"
```

## Train: event_type

```powershell
python -m ai.training.train_event_type_lgbm `
  --dataset ".\ai\artifacts\datasets\lgbm_training_dataset.csv" `
  --out-dir ".\ai\artifacts\models" `
  --classes "daily_noise,repeated_vibration"
```

## Evaluate

```powershell
python -m ai.training.evaluate_lgbm `
  --dataset ".\ai\artifacts\datasets\lgbm_training_dataset.csv" `
  --model-dir ".\ai\artifacts\models" `
  --out-json ".\ai\artifacts\models\evaluation_metrics.json"
```

## Integrity checks in summary JSON

- `output_equals_sensor_rows`: output row count equals sensor row count
- `positive_equals_matched`: number of positive labels equals joined event count
- `orphan_noise_event_count`: noise events that did not match any sensor row
