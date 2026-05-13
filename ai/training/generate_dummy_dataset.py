from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ai.training.build_lgbm_dataset import build_lgbm_dataset


@dataclass(frozen=True)
class Scenario:
    name: str
    weight: float
    event_type: str | None
    time_slot: str  # day | night | any
    sound_range: tuple[float, float]
    vibration_range: tuple[int, int]
    duration_range_ms: tuple[int, int]
    accel_delta_range: tuple[float, float]


SCENARIOS: list[Scenario] = [
    Scenario(
        name="impact_peak_night",
        weight=0.12,
        event_type="impact_noise",
        time_slot="night",
        sound_range=(60.0, 80.0),
        vibration_range=(680, 1100),
        duration_range_ms=(5000, 18000),
        accel_delta_range=(0.14, 0.35),
    ),
    Scenario(
        name="impact_peak_day",
        weight=0.08,
        event_type="impact_noise",
        time_slot="day",
        sound_range=(64.0, 84.0),
        vibration_range=(680, 1050),
        duration_range_ms=(4000, 16000),
        accel_delta_range=(0.13, 0.30),
    ),
    Scenario(
        name="impact_borderline_night",
        weight=0.08,
        event_type="impact_noise",
        time_slot="night",
        sound_range=(52.0, 62.0),
        vibration_range=(520, 800),
        duration_range_ms=(2500, 9000),
        accel_delta_range=(0.11, 0.20),
    ),
    Scenario(
        name="repeated_cluster_night",
        weight=0.12,
        event_type="repeated_vibration",
        time_slot="night",
        sound_range=(45.0, 58.0),
        vibration_range=(360, 900),
        duration_range_ms=(2500, 10000),
        accel_delta_range=(0.01, 0.10),
    ),
    Scenario(
        name="repeated_cluster_day",
        weight=0.10,
        event_type="repeated_vibration",
        time_slot="day",
        sound_range=(45.0, 60.0),
        vibration_range=(350, 880),
        duration_range_ms=(2500, 9000),
        accel_delta_range=(0.01, 0.10),
    ),
    Scenario(
        name="daily_loud_day",
        weight=0.12,
        event_type="daily_noise",
        time_slot="day",
        sound_range=(46.0, 63.0),
        vibration_range=(90, 420),
        duration_range_ms=(2500, 12000),
        accel_delta_range=(0.00, 0.08),
    ),
    Scenario(
        name="daily_loud_night",
        weight=0.08,
        event_type="daily_noise",
        time_slot="night",
        sound_range=(41.0, 56.0),
        vibration_range=(80, 380),
        duration_range_ms=(2500, 10000),
        accel_delta_range=(0.00, 0.08),
    ),
    Scenario(
        name="quiet_background_day",
        weight=0.10,
        event_type=None,
        time_slot="day",
        sound_range=(28.0, 42.0),
        vibration_range=(20, 160),
        duration_range_ms=(500, 4500),
        accel_delta_range=(0.00, 0.05),
    ),
    Scenario(
        name="quiet_background_night",
        weight=0.08,
        event_type=None,
        time_slot="night",
        sound_range=(25.0, 39.0),
        vibration_range=(20, 150),
        duration_range_ms=(500, 4000),
        accel_delta_range=(0.00, 0.05),
    ),
    Scenario(
        name="ambiguous_vibration_no_impact",
        weight=0.05,
        event_type=None,
        time_slot="any",
        sound_range=(40.0, 54.0),
        vibration_range=(250, 520),
        duration_range_ms=(1000, 6000),
        accel_delta_range=(0.00, 0.09),
    ),
    Scenario(
        name="short_transient_spike",
        weight=0.04,
        event_type=None,
        time_slot="any",
        sound_range=(45.0, 70.0),
        vibration_range=(100, 430),
        duration_range_ms=(200, 1200),
        accel_delta_range=(0.04, 0.12),
    ),
    Scenario(
        name="hard_negative_impact_like",
        weight=0.03,
        event_type=None,
        time_slot="any",
        sound_range=(56.0, 66.0),
        vibration_range=(600, 900),
        duration_range_ms=(500, 2900),
        accel_delta_range=(0.00, 0.08),
    ),
]


SENSORS: list[tuple[str, int]] = [
    ("SENSOR-A101-01", 1),
    ("SENSOR-A201-01", 2),
    ("SENSOR-B102-01", 3),
    ("SENSOR-B202-01", 4),
    ("SENSOR-C101-01", 5),
    ("SENSOR-C201-01", 6),
    ("SENSOR-D102-01", 7),
    ("SENSOR-D202-01", 8),
]


def _timestamp_for_slot(rng: random.Random, base_date: datetime, slot: str) -> datetime:
    day_offset = rng.randint(0, 20)
    day = base_date + timedelta(days=day_offset)
    if slot == "day":
        hour = rng.randint(7, 21)
    elif slot == "night":
        hour = rng.choice([22, 23, 0, 1, 2, 3, 4, 5, 6])
    else:
        hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    microsecond = rng.randint(0, 999999)
    return day.replace(
        hour=hour,
        minute=minute,
        second=second,
        microsecond=microsecond,
    )


def _format_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def _severity_for(
    *,
    event_type: str,
    sound: float,
    vibration: int,
    duration_ms: int,
    is_night: bool,
) -> tuple[str, float, float]:
    if event_type == "impact_noise":
        if sound >= 72.0 or vibration >= 900 or duration_ms >= 14000:
            return "critical", 9.5, 0.94
        if sound >= 62.0 or vibration >= 700:
            return "high", 7.5, 0.90
        return "medium", 5.0, 0.83

    if event_type == "repeated_vibration":
        if vibration >= 700 or (is_night and sound >= 52.0):
            return "high", 6.8, 0.88
        if vibration >= 450:
            return "medium", 4.8, 0.82
        return "low", 3.0, 0.76

    # daily_noise
    if is_night and sound >= 50.0:
        return "high", 6.3, 0.84
    if sound >= 52.0:
        return "medium", 4.5, 0.78
    return "low", 2.5, 0.72


def _pattern_label(event_type: str) -> str:
    if event_type == "impact_noise":
        return "impact_pattern"
    if event_type == "repeated_vibration":
        return "vibration_cluster"
    if event_type == "daily_noise":
        return "daily_pattern"
    return "no_pattern"


def _acc_vector_for_delta(rng: random.Random, accel_delta: float) -> tuple[float, float, float]:
    # Keep gravity-like vector around z-axis while controlling magnitude offset.
    sign = rng.choice([-1.0, 1.0])
    target_mag = max(0.2, 1.0 + sign * accel_delta)
    ax = rng.uniform(-0.20, 0.20)
    ay = rng.uniform(-0.20, 0.20)
    remain = max(0.0, target_mag * target_mag - (ax * ax) - (ay * ay))
    az = math.sqrt(remain)
    return (round(ax, 4), round(ay, 4), round(az, 4))


def _is_night(dt: datetime) -> bool:
    return dt.hour >= 22 or dt.hour < 7


def _scenario_counts(total_rows: int) -> list[tuple[Scenario, int]]:
    # Stable rounded allocation with remainder fill.
    raw = [(s, int(total_rows * s.weight)) for s in SCENARIOS]
    assigned = sum(count for _, count in raw)
    remainder = total_rows - assigned
    idx = 0
    while remainder > 0:
        s, count = raw[idx % len(raw)]
        raw[idx % len(raw)] = (s, count + 1)
        idx += 1
        remainder -= 1
    return raw


def generate_dummy_exports(
    *,
    total_rows: int,
    seed: int,
    sensor_out_csv: Path,
    noise_out_csv: Path,
) -> dict[str, Any]:
    rng = random.Random(seed)
    base_date = datetime(2026, 5, 1, 0, 0, 0)
    used_join_keys: set[tuple[str, str]] = set()

    sensor_rows: list[dict[str, str]] = []
    noise_rows: list[dict[str, str]] = []
    noise_id = 1

    for scenario, count in _scenario_counts(total_rows):
        for _ in range(count):
            sensor_id, household_id = rng.choice(SENSORS)

            # Ensure unique join key.
            for _attempt in range(100):
                ts = _timestamp_for_slot(rng, base_date, scenario.time_slot)
                ts_str = _format_ts(ts)
                join_key = (sensor_id, ts_str)
                if join_key not in used_join_keys:
                    used_join_keys.add(join_key)
                    break
            else:
                raise RuntimeError("failed to generate unique sensor timestamp key")

            sound = round(rng.uniform(*scenario.sound_range), 3)
            vibration = int(rng.randint(*scenario.vibration_range))
            duration_ms = int(rng.randint(*scenario.duration_range_ms))
            accel_delta = rng.uniform(*scenario.accel_delta_range)
            ax, ay, az = _acc_vector_for_delta(rng, accel_delta)

            received_at = ts + timedelta(milliseconds=rng.randint(50, 800))
            is_night = _is_night(ts)

            sensor_rows.append(
                {
                    "id": "0",  # reassigned after sort
                    "sensor_id": sensor_id,
                    "household_id": str(household_id),
                    "sound_level": f"{sound:.3f}",
                    "vibration_value": f"{float(vibration):.1f}",
                    "duration_ms": str(duration_ms),
                    "acceleration_x": f"{ax:.4f}",
                    "acceleration_y": f"{ay:.4f}",
                    "acceleration_z": f"{az:.4f}",
                    "received_at": _format_ts(received_at),
                    "sensor_timestamp": ts_str,
                }
            )

            if scenario.event_type is not None:
                severity, severity_score, confidence = _severity_for(
                    event_type=scenario.event_type,
                    sound=sound,
                    vibration=vibration,
                    duration_ms=duration_ms,
                    is_night=is_night,
                )
                ended_at = ts + timedelta(milliseconds=duration_ms)
                noise_rows.append(
                    {
                        "id": str(noise_id),
                        "sensor_id": sensor_id,
                        "household_id": str(household_id),
                        "event_type": scenario.event_type,
                        "severity": severity,
                        "severity_score": f"{severity_score:.1f}",
                        "confidence": f"{confidence:.2f}",
                        "is_night": "1" if is_night else "0",
                        "is_meaningful": "1",
                        "pattern_label": _pattern_label(scenario.event_type),
                        "avg_sound_level": f"{sound:.3f}",
                        "max_sound_level": f"{sound:.3f}",
                        "avg_vibration": f"{float(vibration):.1f}",
                        "duration_ms": str(duration_ms),
                        "sample_count": str(rng.randint(1, 4)),
                        "started_at": ts_str,
                        "ended_at": _format_ts(ended_at),
                        "status": "new",
                    }
                )
                noise_id += 1

    sensor_rows.sort(key=lambda row: row["sensor_timestamp"])
    for idx, row in enumerate(sensor_rows, start=1):
        row["id"] = str(idx)

    sensor_fields = [
        "id",
        "sensor_id",
        "household_id",
        "sound_level",
        "vibration_value",
        "duration_ms",
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
        "received_at",
        "sensor_timestamp",
    ]
    noise_fields = [
        "id",
        "sensor_id",
        "household_id",
        "event_type",
        "severity",
        "severity_score",
        "confidence",
        "is_night",
        "is_meaningful",
        "pattern_label",
        "avg_sound_level",
        "max_sound_level",
        "avg_vibration",
        "duration_ms",
        "sample_count",
        "started_at",
        "ended_at",
        "status",
    ]

    sensor_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with sensor_out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sensor_fields)
        writer.writeheader()
        writer.writerows(sensor_rows)

    noise_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with noise_out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=noise_fields)
        writer.writeheader()
        writer.writerows(noise_rows)

    by_type: dict[str, int] = {"none": 0}
    by_scenario: dict[str, int] = {}
    for scenario, count in _scenario_counts(total_rows):
        by_scenario[scenario.name] = count
        key = scenario.event_type if scenario.event_type is not None else "none"
        by_type[key] = by_type.get(key, 0) + count

    return {
        "total_sensor_rows": len(sensor_rows),
        "total_noise_rows": len(noise_rows),
        "label_positive_ratio": round(len(noise_rows) / len(sensor_rows), 4),
        "event_type_mix_in_sensor_rows": by_type,
        "scenario_counts": by_scenario,
        "paths": {
            "sensor_csv": str(sensor_out_csv),
            "noise_csv": str(noise_out_csv),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate diverse dummy sensor/noise exports and optionally "
            "build LightGBM training dataset."
        )
    )
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--build-training-dataset",
        action="store_true",
        help="also build joined lgbm training dataset csv",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sensor_csv = out_dir / "all_sensor_readings_dummy_2000.csv"
    noise_csv = out_dir / "all_noise_events_dummy_2000.csv"

    summary = generate_dummy_exports(
        total_rows=args.rows,
        seed=args.seed,
        sensor_out_csv=sensor_csv,
        noise_out_csv=noise_csv,
    )

    if args.build_training_dataset:
        training_csv = out_dir / "lgbm_training_dataset_dummy_2000.csv"
        training_summary = out_dir / "lgbm_training_summary_dummy_2000.json"
        join_summary = build_lgbm_dataset(
            sensor_csv=sensor_csv,
            noise_csv=noise_csv,
            out_csv=training_csv,
            out_summary=training_summary,
        )
        summary["joined_training_dataset"] = join_summary

    summary_path = out_dir / "dummy_generation_summary_2000.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
