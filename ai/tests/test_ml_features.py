from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ai.ml_features import (
    FEATURE_COLUMNS,
    build_feature_spec,
    event_features_from_sensor_row,
    feature_dict_from_event,
    feature_dict_from_payload,
    feature_vector_from_event,
    write_feature_spec,
)
from ai.schemas import EventFeatures
from ai.vibration import VIBRATION_LEVEL_CATEGORY_CODES, vibration_raw_to_acc_mps2


def _case_dir() -> Path:
    tmp_root = Path(__file__).resolve().parent / ".tmp"
    tmp_root.mkdir(exist_ok=True)
    case_dir = tmp_root / f"ml_features_{uuid4().hex}"
    case_dir.mkdir()
    return case_dir


def test_feature_vector_order_is_stable() -> None:
    event = EventFeatures(
        device_id="SENSOR-A101-01",
        source="test",
        sound_level=55.0,
        vibration_value=420,
        duration_ms=6000,
        accel_delta=0.15,
        timestamp=datetime.fromisoformat("2026-05-13T14:20:00+09:00"),
        recent_count_10min=3,
    )

    feature_dict = feature_dict_from_event(event)
    feature_vector = feature_vector_from_event(event)

    assert len(feature_vector) == len(FEATURE_COLUMNS)
    assert set(feature_dict.keys()) == set(FEATURE_COLUMNS)
    for index, name in enumerate(FEATURE_COLUMNS):
        assert feature_vector[index] == feature_dict[name]


def test_feature_values_respect_thresholds_and_night_window() -> None:
    event = EventFeatures(
        device_id="SENSOR-A101-01",
        source="test",
        sound_level=41.0,
        vibration_value=360,
        duration_ms=9000,
        accel_delta=0.02,
        timestamp=datetime.fromisoformat("2026-05-13T23:10:00+09:00"),
        recent_count_10min=2,
    )
    features = feature_dict_from_event(event)

    assert features["is_nighttime"] == 1.0
    assert features["is_daytime"] == 0.0
    assert features["hour_of_day"] == 23.0
    assert features["sound_over_airborne_leq"] == 1.0
    assert features["sound_over_impact_lmax"] == 0.0
    assert features["vibration_raw"] == 360.0
    assert features["vibration_acc_mps2"] == vibration_raw_to_acc_mps2(360)
    assert features["vibration_level_category"] == float(
        VIBRATION_LEVEL_CATEGORY_CODES["impact_risk"]
    )
    assert features["impact_count_in_window"] == 1.0
    assert features["repeated_impact_flag"] == 0.0
    assert features["duration_over_medium_ms"] == 1000.0


def test_sensor_row_parsing_handles_missing_and_nan_values() -> None:
    row = {
        "sensor_id": "SENSOR-A101-01",
        "sound_level": "nan",
        "vibration_value": "",
        "duration_ms": None,
        "acceleration_x": "",
        "acceleration_y": "",
        "acceleration_z": "",
        "sensor_timestamp": "2026-05-13 12:01:02.000000",
        "recent_count_10min": "-5",
    }

    event = event_features_from_sensor_row(row)
    features = feature_dict_from_event(event)

    assert event.sound_level == 0.0
    assert event.vibration_value == 0
    assert event.duration_ms == 0
    assert event.accel_delta == 0.0
    assert event.recent_count_10min == 0
    assert event.timestamp == datetime.fromisoformat("2026-05-13T12:01:02+09:00")
    assert event.timestamp_source == "sensor_timestamp"
    assert features["sound_level"] == 0.0
    assert features["vibration_raw"] == 0.0
    assert features["vibration_acc_mps2"] == 0.005
    assert features["duration_ms"] == 0.0


def test_payload_and_event_conversion_match() -> None:
    payload = {
        "sensor_id": "SENSOR-A101-01",
        "sound_level": 58.2,
        "vibration_value": 640,
        "duration_ms": 4200,
        "accel_delta": 0.16,
        "timestamp": "2026-05-13T10:00:00+09:00",
        "recent_count_10min": 2,
    }
    event = event_features_from_sensor_row(payload)

    from_payload = feature_dict_from_payload(payload)
    from_event = feature_dict_from_event(event)

    assert from_payload == from_event


def test_sensor_row_timestamp_priority_and_conflict_metadata() -> None:
    event = event_features_from_sensor_row(
        {
            "sensor_id": "SENSOR-A101-01",
            "sound_level": 45.2,
            "vibration_value": 1007,
            "duration_ms": 2000,
            "sensor_timestamp": "2026-05-11T22:00:00",
            "timestamp": "2026-05-11T21:59:00",
        }
    )

    assert event.timestamp.isoformat() == "2026-05-11T22:00:00+09:00"
    assert event.timestamp_source == "sensor_timestamp"
    assert event.timestamp_conflict is True


def test_sensor_row_preserves_fractional_vibration_raw() -> None:
    event = event_features_from_sensor_row(
        {
            "sensor_id": "SENSOR-A101-01",
            "sound_level": 45.2,
            "vibration_value": "10.8",
            "duration_ms": 2000,
            "sensor_timestamp": "2026-05-11T22:00:00",
        }
    )
    features = feature_dict_from_event(event)

    assert event.vibration_value == 10.8
    assert features["vibration_raw"] == 10.8


def test_feature_spec_json_output() -> None:
    case_dir = _case_dir()
    out_path = case_dir / "feature_spec.json"

    spec = write_feature_spec(out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))

    assert spec["feature_columns"] == FEATURE_COLUMNS
    assert loaded["feature_columns"] == FEATURE_COLUMNS
    assert loaded["feature_spec_version"] == build_feature_spec()["feature_spec_version"]
