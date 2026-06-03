from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.config import get_settings
from ai.ml_features import FEATURE_SPEC_VERSION
from ai.time_utils import resolve_event_timestamp
from ai.vibration import (
    IMPACT_WINDOW_SECONDS,
    build_vibration_risk_profile,
    category_is_caution_or_higher,
)

REQUIRED_SENSOR_FIELDS = [
    "sensor_id",
    "household_id",
    "sound_level",
    "vibration_value",
    "duration_ms",
    "received_at",
    "sensor_timestamp",
]

DERIVED_FEATURE_FIELDS = [
    "vibration_raw",
    "vibration_acc_mps2",
    "vibration_dbv",
    "is_daytime",
    "is_nighttime",
    "time_period",
    "vibration_level_category",
    "impact_count_in_window",
    "repeated_impact_flag",
    "noise_risk_level",
    "feature_spec_version",
]


def _normalize_timestamp(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    candidate = raw.replace("T", " ").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return raw
    return dt.isoformat(sep=" ")


def _build_join_key(sensor_id: str, ts: str) -> tuple[str, str]:
    return (sensor_id.strip(), _normalize_timestamp(ts))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    return int(_safe_float(value, float(default)))


def _format_float(value: float) -> str:
    return f"{float(value):.10g}"


def _pseudo_noise_risk_level(
    *,
    vibration_level_category: str,
    impact_count_in_window: int,
    sound_level: float,
    is_nighttime: bool,
) -> str:
    # Pseudo-label for internal alert/model bootstrap only; this is not a
    # legal judgment label or externally verified ground truth.
    settings = get_settings()
    high_sound = sound_level >= settings.thresholds.impact_lmax(is_nighttime)
    if vibration_level_category == "impact_risk" and (
        impact_count_in_window >= 3 or high_sound
    ):
        return "high"
    if vibration_level_category == "impact_risk":
        return "warning"
    if vibration_level_category == "warning" or impact_count_in_window >= 6:
        return "warning"
    if vibration_level_category == "caution" or impact_count_in_window >= 3:
        return "caution"
    return "normal"


def _add_vibration_preprocessing_columns(rows: list[dict[str, str]]) -> None:
    settings = get_settings()
    ordered: list[tuple[int, str, datetime]] = []
    for index, row in enumerate(rows):
        timestamp_resolution = resolve_event_timestamp(
            row,
            fallback_to_now=False,
        )
        if timestamp_resolution is None:
            continue
        ts = timestamp_resolution.resolved_timestamp
        key = (row.get("sensor_id") or row.get("household_id") or "unknown").strip()
        ordered.append((index, key, ts))

    windows: dict[str, deque[datetime]] = defaultdict(deque)
    for index, key, ts in sorted(ordered, key=lambda item: (item[1], item[2])):
        row = rows[index]
        raw = max(0.0, _safe_float(row.get("vibration_value", row.get("vibration_raw")), 0.0))
        profile_without_count = build_vibration_risk_profile(
            vibration_value=raw,
            timestamp=ts,
            settings=settings,
            impact_count_in_window=0,
        )

        window = windows[key]
        while window and (ts - window[0]).total_seconds() > IMPACT_WINDOW_SECONDS:
            window.popleft()
        if category_is_caution_or_higher(profile_without_count.vibration_level_category):
            window.append(ts)

        profile = build_vibration_risk_profile(
            vibration_value=raw,
            timestamp=ts,
            settings=settings,
            impact_count_in_window=len(window),
        )
        sound_level = _safe_float(row.get("sound_level"), 0.0)
        row["vibration_raw"] = _format_float(profile.vibration_raw)
        row["vibration_acc_mps2"] = f"{profile.vibration_acc_mps2:.6f}"
        row["vibration_dbv"] = f"{profile.vibration_dbv:.2f}"
        row["is_daytime"] = "1" if profile.is_daytime else "0"
        row["is_nighttime"] = "1" if profile.is_nighttime else "0"
        row["time_period"] = profile.time_period
        row["vibration_level_category"] = profile.vibration_level_category
        row["impact_count_in_window"] = str(profile.impact_count)
        row["repeated_impact_flag"] = "1" if profile.repeated_impact_flag else "0"
        row["noise_risk_level"] = _pseudo_noise_risk_level(
            vibration_level_category=profile.vibration_level_category,
            impact_count_in_window=profile.impact_count,
            sound_level=sound_level,
            is_nighttime=profile.is_nighttime,
        )
        row["feature_spec_version"] = FEATURE_SPEC_VERSION


def build_lgbm_dataset(
    *,
    sensor_csv: Path,
    noise_csv: Path,
    out_csv: Path,
    out_summary: Path | None = None,
) -> dict[str, Any]:
    if not sensor_csv.exists():
        raise FileNotFoundError(f"sensor csv not found: {sensor_csv}")
    if not noise_csv.exists():
        raise FileNotFoundError(f"noise csv not found: {noise_csv}")

    noise_index: dict[tuple[str, str], dict[str, str]] = {}
    noise_row_count = 0
    noise_rows_missing_key = 0

    with noise_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            noise_row_count += 1
            sensor_id = (row.get("sensor_id") or "").strip()
            started_at = row.get("started_at")
            key = _build_join_key(sensor_id, started_at or "")
            if not key[0] or not key[1]:
                noise_rows_missing_key += 1
                continue
            if key in noise_index:
                raise ValueError(
                    "duplicate noise join key detected: "
                    f"sensor_id={key[0]}, started_at={key[1]}"
                )
            noise_index[key] = row

    sensor_row_count = 0
    matched_noise_event_count = 0
    output_rows: list[dict[str, str]] = []

    with sensor_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        sensor_fields = list(reader.fieldnames or [])
        if not sensor_fields:
            raise ValueError("sensor csv has no header")

        for row in reader:
            sensor_row_count += 1
            sensor_id = (row.get("sensor_id") or "").strip()
            sensor_timestamp = row.get("sensor_timestamp") or ""
            key = _build_join_key(sensor_id, sensor_timestamp)
            matched = noise_index.get(key)
            if matched is not None:
                matched_noise_event_count += 1

            output = dict(row)
            output["is_meaningful_label"] = "1" if matched is not None else "0"
            output["event_type_label"] = (matched.get("event_type") if matched else "") or ""
            output["severity_label"] = (matched.get("severity") if matched else "") or ""
            output["noise_event_id"] = (matched.get("id") if matched else "") or ""
            output["noise_event_started_at"] = (
                (matched.get("started_at") if matched else "") or ""
            )
            output_rows.append(output)

    _add_vibration_preprocessing_columns(output_rows)

    orphan_noise_event_count = len(noise_index) - matched_noise_event_count
    output_row_count = len(output_rows)
    positive_label_count = sum(
        1 for row in output_rows if row.get("is_meaningful_label") == "1"
    )

    derived_fields = DERIVED_FEATURE_FIELDS
    out_fields = sensor_fields + [
        field for field in derived_fields if field not in sensor_fields
    ] + [
        "is_meaningful_label",
        "event_type_label",
        "severity_label",
        "noise_event_id",
        "noise_event_started_at",
    ]

    _ensure_parent(out_csv)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "join_key": {
            "sensor_table": ["sensor_id", "sensor_timestamp"],
            "noise_table": ["sensor_id", "started_at"],
        },
        "sensor_row_count": sensor_row_count,
        "noise_row_count": noise_row_count,
        "noise_rows_missing_key": noise_rows_missing_key,
        "output_row_count": output_row_count,
        "matched_noise_event_count": matched_noise_event_count,
        "positive_label_count": positive_label_count,
        "orphan_noise_event_count": orphan_noise_event_count,
        "integrity_checks": {
            "output_equals_sensor_rows": output_row_count == sensor_row_count,
            "positive_equals_matched": positive_label_count == matched_noise_event_count,
        },
        "paths": {
            "sensor_csv": str(sensor_csv),
            "noise_csv": str(noise_csv),
            "output_csv": str(out_csv),
        },
    }

    summary_path = out_summary
    if summary_path is not None:
        _ensure_parent(summary_path)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        summary["paths"]["summary_json"] = str(summary_path)

    return summary


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError("sensor csv has no header")
        return fields, [dict(row) for row in reader]


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float((ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight))


def _numeric_stats(rows: list[dict[str, str]], field: str) -> dict[str, float | int | None]:
    values: list[float] = []
    for row in rows:
        parsed = _safe_float(row.get(field), default=float("nan"))
        if math.isfinite(parsed):
            values.append(parsed)
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(values) if values else None,
    }


def _value_counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(field, ""))] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _nan_inf_count(rows: list[dict[str, str]], fields: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in fields:
        count = 0
        for row in rows:
            parsed = _safe_float(row.get(field), default=float("nan"))
            if not math.isfinite(parsed):
                count += 1
        result[field] = count
    return result


def _sensor_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return _value_counts(rows, "sensor_id")


def _raw_fractional_count(rows: list[dict[str, str]]) -> int:
    count = 0
    for row in rows:
        raw = _safe_float(row.get("vibration_raw"), 0.0)
        if abs(raw - math.floor(raw)) > 1e-9:
            count += 1
    return count


def _category_changes_if_int_truncated(rows: list[dict[str, str]]) -> dict[str, Any]:
    settings = get_settings()
    changed = 0
    by_change: dict[str, int] = defaultdict(int)
    for row in rows:
        timestamp_resolution = resolve_event_timestamp(row, fallback_to_now=False)
        if timestamp_resolution is None:
            continue
        raw = _safe_float(row.get("vibration_raw", row.get("vibration_value")), 0.0)
        float_profile = build_vibration_risk_profile(
            vibration_value=raw,
            timestamp=timestamp_resolution.resolved_timestamp,
            settings=settings,
            impact_count_in_window=0,
        )
        int_profile = build_vibration_risk_profile(
            vibration_value=int(raw),
            timestamp=timestamp_resolution.resolved_timestamp,
            settings=settings,
            impact_count_in_window=0,
        )
        if float_profile.vibration_level_category != int_profile.vibration_level_category:
            changed += 1
            key = (
                f"{float_profile.vibration_level_category}"
                f"->{int_profile.vibration_level_category}"
            )
            by_change[key] += 1
    return {
        "changed_count": changed,
        "by_change": dict(sorted(by_change.items(), key=lambda item: (-item[1], item[0]))),
    }


def _metadata_for_feature_dataset(
    *,
    sensor_csv: Path,
    out_csv: Path,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    required_presence = {
        field: field in rows[0] if rows else False
        for field in REQUIRED_SENSOR_FIELDS
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(sensor_csv),
        "output_file": str(out_csv),
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "row_count": len(rows),
        "required_columns": REQUIRED_SENSOR_FIELDS,
        "required_columns_present": required_presence,
        "vibration_value_processing": "float raw ADC-like SEN0209 value; never int-truncated",
        "vibration_conversion_formula": {
            "baseline_raw": 8.0,
            "reference_raw": 1007.0,
            "min_acceleration_mps2": 0.005,
            "max_reference_acceleration_mps2": 0.45,
            "dbv_reference_acceleration": 1e-5,
        },
        "time_policy": {
            "sensor_timestamp": "KST sensor measurement wall time when naive",
            "daytime": "06:00 <= time < 22:00",
            "nighttime": "22:00 <= time or time < 06:00",
        },
        "rolling_window": {
            "group_by": "sensor_id",
            "window_seconds": IMPACT_WINDOW_SECONDS,
            "counted_categories": ["caution", "warning", "impact_risk"],
            "repeated_impact_flag_threshold": 3,
            "warning_count_threshold": 6,
        },
        "pseudo_label": {
            "column": "noise_risk_level",
            "is_ground_truth": False,
            "description": (
                "Internal alert/bootstrap pseudo-label, not a legal judgment "
                "and not externally verified inter-floor-noise ground truth."
            ),
        },
        "vibration_acc_mps2_stats": _numeric_stats(rows, "vibration_acc_mps2"),
        "vibration_dbv_stats": _numeric_stats(rows, "vibration_dbv"),
        "vibration_level_category_distribution": _value_counts(rows, "vibration_level_category"),
        "impact_count_in_window_stats": _numeric_stats(rows, "impact_count_in_window"),
        "repeated_impact_flag_distribution": _value_counts(rows, "repeated_impact_flag"),
        "noise_risk_level_distribution": _value_counts(rows, "noise_risk_level"),
        "sensor_id_row_counts": _sensor_counts(rows),
        "vibration_raw_fractional_count": _raw_fractional_count(rows),
        "nan_or_inf_counts": _nan_inf_count(
            rows,
            ["vibration_raw", "vibration_acc_mps2", "vibration_dbv", "impact_count_in_window"],
        ),
        "int_truncation_category_change_check": _category_changes_if_int_truncated(rows),
    }


def build_sensor_feature_dataset(
    *,
    sensor_csv: Path,
    out_csv: Path,
    out_metadata: Path | None = None,
) -> dict[str, Any]:
    if not sensor_csv.exists():
        raise FileNotFoundError(f"sensor csv not found: {sensor_csv}")

    sensor_fields, output_rows = _read_csv_rows(sensor_csv)
    missing_required = [field for field in REQUIRED_SENSOR_FIELDS if field not in sensor_fields]
    if missing_required:
        raise ValueError(f"missing required sensor columns: {', '.join(missing_required)}")

    _add_vibration_preprocessing_columns(output_rows)

    out_fields = sensor_fields + [
        field for field in DERIVED_FEATURE_FIELDS if field not in sensor_fields
    ]
    _ensure_parent(out_csv)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    metadata = _metadata_for_feature_dataset(
        sensor_csv=sensor_csv,
        out_csv=out_csv,
        rows=output_rows,
    )
    metadata_path = out_metadata
    if metadata_path is not None:
        _ensure_parent(metadata_path)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        metadata["metadata_file"] = str(metadata_path)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LightGBM training dataset by joining "
            "raw sensor readings and noise events."
        )
    )
    parser.add_argument("--sensor-csv", required=True, type=Path)
    parser.add_argument("--noise-csv", required=False, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-summary", required=False, type=Path)
    parser.add_argument(
        "--feature-only",
        action="store_true",
        help="Build a feature dataset with pseudo-labels from raw sensor CSV only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.feature_only:
        summary = build_sensor_feature_dataset(
            sensor_csv=args.sensor_csv,
            out_csv=args.out_csv,
            out_metadata=args.out_summary,
        )
    else:
        if args.noise_csv is None:
            raise ValueError("--noise-csv is required unless --feature-only is set")
        summary = build_lgbm_dataset(
            sensor_csv=args.sensor_csv,
            noise_csv=args.noise_csv,
            out_csv=args.out_csv,
            out_summary=args.out_summary,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
