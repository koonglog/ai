from __future__ import annotations

import csv
import json
from pathlib import Path
from uuid import uuid4

import pytest

from ai.training.build_lgbm_dataset import build_lgbm_dataset


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _case_dir() -> Path:
    tmp_root = Path(__file__).resolve().parent / ".tmp"
    tmp_root.mkdir(exist_ok=True)
    case_dir = tmp_root / f"lgbm_dataset_{uuid4().hex}"
    case_dir.mkdir()
    return case_dir


def test_build_lgbm_dataset_labels_and_summary() -> None:
    case_dir = _case_dir()
    sensor_csv = case_dir / "all_sensor_readings.csv"
    noise_csv = case_dir / "all_noise_events.csv"
    out_csv = case_dir / "lgbm_training_dataset.csv"
    out_summary = case_dir / "lgbm_training_summary.json"

    sensor_rows = [
        {
            "id": "1",
            "sensor_id": "SENSOR-A101-01",
            "household_id": "1",
            "sound_level": "55.0",
            "vibration_value": "700.0",
            "duration_ms": "3000",
            "acceleration_x": "0.1",
            "acceleration_y": "0.0",
            "acceleration_z": "1.0",
            "received_at": "2026-05-11 12:07:49.702419",
            "sensor_timestamp": "2026-05-11 12:00:00.000000",
        },
        {
            "id": "2",
            "sensor_id": "SENSOR-A101-01",
            "household_id": "1",
            "sound_level": "57.3",
            "vibration_value": "7.0",
            "duration_ms": "2000",
            "acceleration_x": "",
            "acceleration_y": "",
            "acceleration_z": "",
            "received_at": "2026-05-11 12:27:04.047238",
            "sensor_timestamp": "2026-05-11 12:27:04.047238",
        },
        {
            "id": "3",
            "sensor_id": "SENSOR-A101-01",
            "household_id": "1",
            "sound_level": "41.9",
            "vibration_value": "8.0",
            "duration_ms": "2000",
            "acceleration_x": "",
            "acceleration_y": "",
            "acceleration_z": "",
            "received_at": "2026-05-11 12:27:07.880298",
            "sensor_timestamp": "2026-05-11 12:27:07.880298",
        },
        {
            "id": "4",
            "sensor_id": "SENSOR-A101-01",
            "household_id": "1",
            "sound_level": "72.5",
            "vibration_value": "8.0",
            "duration_ms": "2000",
            "acceleration_x": "",
            "acceleration_y": "",
            "acceleration_z": "",
            "received_at": "2026-05-11 12:27:10.898634",
            "sensor_timestamp": "2026-05-11 12:27:10.898634",
        },
    ]
    noise_rows = [
        {
            "id": "11",
            "sensor_id": "SENSOR-A101-01",
            "household_id": "1",
            "event_type": "daily_noise",
            "severity": "medium",
            "severity_score": "3.0",
            "confidence": "0.74",
            "is_night": "0",
            "is_meaningful": "1",
            "pattern_label": "no_pattern",
            "avg_sound_level": "55.0",
            "max_sound_level": "55.0",
            "avg_vibration": "700.0",
            "duration_ms": "3000",
            "sample_count": "1",
            "started_at": "2026-05-11 12:00:00.000000",
            "ended_at": "",
            "status": "new",
        },
        {
            "id": "12",
            "sensor_id": "SENSOR-A101-01",
            "household_id": "1",
            "event_type": "daily_noise",
            "severity": "low",
            "severity_score": "2.0",
            "confidence": "0.74",
            "is_night": "0",
            "is_meaningful": "1",
            "pattern_label": "no_pattern",
            "avg_sound_level": "57.3",
            "max_sound_level": "57.3",
            "avg_vibration": "7.0",
            "duration_ms": "2000",
            "sample_count": "1",
            "started_at": "2026-05-11 12:27:04.047238",
            "ended_at": "",
            "status": "new",
        },
        # Orphan (not present in sensor rows)
        {
            "id": "13",
            "sensor_id": "SENSOR-A101-01",
            "household_id": "1",
            "event_type": "repeated_vibration",
            "severity": "high",
            "severity_score": "5.0",
            "confidence": "0.82",
            "is_night": "0",
            "is_meaningful": "1",
            "pattern_label": "vibration_cluster",
            "avg_sound_level": "61.1",
            "max_sound_level": "62.0",
            "avg_vibration": "400.0",
            "duration_ms": "2000",
            "sample_count": "1",
            "started_at": "2026-05-11 12:55:00.000000",
            "ended_at": "",
            "status": "new",
        },
    ]

    _write_csv(sensor_csv, sensor_rows, list(sensor_rows[0].keys()))
    _write_csv(noise_csv, noise_rows, list(noise_rows[0].keys()))

    summary = build_lgbm_dataset(
        sensor_csv=sensor_csv,
        noise_csv=noise_csv,
        out_csv=out_csv,
        out_summary=out_summary,
    )

    assert out_csv.exists()
    assert out_summary.exists()

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        out_rows = list(csv.DictReader(f))

    assert len(out_rows) == 4
    assert sum(int(row["is_meaningful_label"]) for row in out_rows) == 2
    assert out_rows[2]["is_meaningful_label"] == "0"
    assert out_rows[2]["event_type_label"] == ""

    assert summary["sensor_row_count"] == 4
    assert summary["noise_row_count"] == 3
    assert summary["output_row_count"] == 4
    assert summary["matched_noise_event_count"] == 2
    assert summary["positive_label_count"] == 2
    assert summary["orphan_noise_event_count"] == 1
    assert summary["integrity_checks"]["output_equals_sensor_rows"] is True
    assert summary["integrity_checks"]["positive_equals_matched"] is True

    disk_summary = json.loads(out_summary.read_text(encoding="utf-8"))
    assert disk_summary["sensor_row_count"] == 4
    assert disk_summary["paths"]["output_csv"] == str(out_csv)


def test_build_lgbm_dataset_duplicate_noise_key_raises() -> None:
    case_dir = _case_dir()
    sensor_csv = case_dir / "all_sensor_readings.csv"
    noise_csv = case_dir / "all_noise_events.csv"
    out_csv = case_dir / "lgbm_training_dataset.csv"

    sensor_rows = [
        {
            "id": "1",
            "sensor_id": "SENSOR-A101-01",
            "household_id": "1",
            "sound_level": "55.0",
            "vibration_value": "700.0",
            "duration_ms": "3000",
            "acceleration_x": "0.1",
            "acceleration_y": "0.0",
            "acceleration_z": "1.0",
            "received_at": "2026-05-11 12:07:49.702419",
            "sensor_timestamp": "2026-05-11 12:00:00.000000",
        }
    ]
    noise_rows = [
        {
            "id": "11",
            "sensor_id": "SENSOR-A101-01",
            "household_id": "1",
            "event_type": "daily_noise",
            "severity": "medium",
            "severity_score": "3.0",
            "confidence": "0.74",
            "is_night": "0",
            "is_meaningful": "1",
            "pattern_label": "no_pattern",
            "avg_sound_level": "55.0",
            "max_sound_level": "55.0",
            "avg_vibration": "700.0",
            "duration_ms": "3000",
            "sample_count": "1",
            "started_at": "2026-05-11 12:00:00.000000",
            "ended_at": "",
            "status": "new",
        },
        {
            "id": "12",
            "sensor_id": "SENSOR-A101-01",
            "household_id": "1",
            "event_type": "daily_noise",
            "severity": "high",
            "severity_score": "4.0",
            "confidence": "0.80",
            "is_night": "0",
            "is_meaningful": "1",
            "pattern_label": "no_pattern",
            "avg_sound_level": "56.0",
            "max_sound_level": "56.0",
            "avg_vibration": "710.0",
            "duration_ms": "3000",
            "sample_count": "1",
            "started_at": "2026-05-11 12:00:00.000000",
            "ended_at": "",
            "status": "new",
        },
    ]

    _write_csv(sensor_csv, sensor_rows, list(sensor_rows[0].keys()))
    _write_csv(noise_csv, noise_rows, list(noise_rows[0].keys()))

    with pytest.raises(ValueError, match="duplicate noise join key"):
        build_lgbm_dataset(
            sensor_csv=sensor_csv,
            noise_csv=noise_csv,
            out_csv=out_csv,
        )
