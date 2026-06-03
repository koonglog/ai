from __future__ import annotations

import csv
from pathlib import Path
from uuid import uuid4

from ai.ml_features import FEATURE_COLUMNS, FEATURE_SPEC_VERSION
from ai.training.train_noise_risk_lgbm import (
    FORBIDDEN_FEATURE_COLUMNS,
    TARGET_COLUMN,
    _split_rows_time_based,
    load_noise_risk_artifact,
    predict_noise_risk_from_rows,
    train_noise_risk_model,
)


def _case_dir() -> Path:
    tmp_root = Path(__file__).resolve().parent / ".tmp"
    tmp_root.mkdir(exist_ok=True)
    case_dir = tmp_root / f"noise_risk_train_{uuid4().hex}"
    case_dir.mkdir()
    return case_dir


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row(index: int, label: str) -> dict[str, str]:
    label_to_raw = {
        "normal": "8.0",
        "caution": "19.3",
        "warning": "120.0",
        "high": "1007.0",
    }
    label_to_category = {
        "normal": "normal",
        "caution": "caution",
        "warning": "warning",
        "high": "impact_risk",
    }
    label_to_count = {
        "normal": "0",
        "caution": "3",
        "warning": "6",
        "high": "6",
    }
    raw = label_to_raw[label]
    timestamp = f"2026-05-{11 + (index // 96):02d}T{index % 24:02d}:00:00"
    return {
        "id": str(index),
        "sensor_id": f"SENSOR-{index % 3}",
        "household_id": str((index % 3) + 1),
        "sound_level": "45.0",
        "vibration_value": raw,
        "duration_ms": "2000",
        "received_at": timestamp,
        "sensor_timestamp": timestamp,
        "vibration_raw": raw,
        "vibration_acc_mps2": "0.005",
        "vibration_dbv": "53.98",
        "is_daytime": "0",
        "is_nighttime": "1",
        "time_period": "nighttime",
        "vibration_level_category": label_to_category[label],
        "impact_count_in_window": label_to_count[label],
        "repeated_impact_flag": "1" if int(label_to_count[label]) >= 3 else "0",
        TARGET_COLUMN: label,
        "feature_spec_version": FEATURE_SPEC_VERSION,
    }


def _pattern_row(index: int) -> dict[str, str]:
    timestamp = f"2026-05-{11 + (index // 96):02d}T{index % 24:02d}:00:00"
    return {
        "sensor_id": f"SENSOR-{index % 3}",
        "household_id": str((index % 3) + 1),
        "event_type": "impact_noise",
        "severity": "critical",
        "severity_score": "8",
        "confidence": "0.9",
        "is_night": "1",
        "is_meaningful": "1",
        "pattern_label": "night_repeated_impact",
        "avg_sound_level": "45.0",
        "max_sound_level": "45.0",
        "avg_vibration": "1007.0",
        "duration_ms": "2000",
        "sample_count": "1",
        "started_at": timestamp,
        "ended_at": timestamp,
        "status": "new",
        "recent_count_10min": "1",
    }


def test_noise_risk_training_artifact_load_and_sample_inference() -> None:
    case_dir = _case_dir()
    dataset = case_dir / "training_dataset_feature_spec_2_0_0.csv"
    pattern_events = case_dir / "pattern_events.csv"
    rows = [_row(i, ["normal", "caution", "warning", "high"][i % 4]) for i in range(160)]
    _write_csv(dataset, rows)
    _write_csv(pattern_events, [_pattern_row(i) for i in range(0, 160, 8)])

    result = train_noise_risk_model(
        dataset=dataset,
        pattern_events=pattern_events,
        artifact_dir=case_dir / "artifacts",
        report_dir=case_dir / "reports",
        n_estimators=30,
        early_stopping_rounds=5,
    )

    artifact_path = Path(result["metadata"]["artifacts"]["model_pickle"])
    artifact = load_noise_risk_artifact(artifact_path)
    predictions = predict_noise_risk_from_rows(artifact, rows[:3])

    assert artifact["feature_spec_version"] == FEATURE_SPEC_VERSION
    assert artifact["feature_names"] == FEATURE_COLUMNS
    assert len(predictions) == 3
    assert all("model_prediction" in row for row in predictions)


def test_noise_risk_training_forbidden_columns_exclude_pattern_and_target() -> None:
    assert TARGET_COLUMN in FORBIDDEN_FEATURE_COLUMNS
    assert "feature_spec_version" in FORBIDDEN_FEATURE_COLUMNS
    assert "event_type" in FORBIDDEN_FEATURE_COLUMNS
    assert "severity" in FORBIDDEN_FEATURE_COLUMNS
    assert "pattern_label" in FORBIDDEN_FEATURE_COLUMNS
    assert "recent_count_10min" in FORBIDDEN_FEATURE_COLUMNS
    assert set(FEATURE_COLUMNS).isdisjoint(FORBIDDEN_FEATURE_COLUMNS)


def test_time_based_split_creates_three_ordered_partitions() -> None:
    rows = [_row(i, ["normal", "caution", "warning", "high"][i % 4]) for i in range(100)]
    split = _split_rows_time_based(rows)

    assert len(split.train) == 70
    assert len(split.valid) == 15
    assert len(split.test) == 15
    assert split.train[-1]["sensor_timestamp"] <= split.valid[0]["sensor_timestamp"]
    assert split.valid[-1]["sensor_timestamp"] <= split.test[0]["sensor_timestamp"]
