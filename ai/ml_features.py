from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import AISettings, get_settings
from .schemas import EventFeatures
from .time_utils import as_kst, resolve_event_timestamp
from .vibration import (
    VIBRATION_LEVEL_CATEGORY_CODES,
    build_vibration_risk_profile,
    category_is_caution_or_higher,
)

FEATURE_SPEC_VERSION = "2.0.0"
FEATURE_COLUMNS: list[str] = [
    "sound_level",
    "vibration_raw",
    "vibration_acc_mps2",
    "vibration_dbv",
    "duration_ms",
    "hour_of_day",
    "is_daytime",
    "is_nighttime",
    "vibration_level_category",
    "impact_count_in_window",
    "repeated_impact_flag",
    "sound_over_impact_leq",
    "sound_over_impact_lmax",
    "sound_over_airborne_leq",
    "duration_over_short_ms",
    "duration_over_medium_ms",
    "duration_over_long_ms",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return default
    return parsed


def _is_night(ts: datetime, settings: AISettings) -> bool:
    start = settings.nighttime.start_hour
    end = settings.nighttime.end_hour
    hour = as_kst(ts).hour
    return hour >= start or hour < end


def _accel_delta_from_xyz(x: Any, y: Any, z: Any) -> float:
    fx = _safe_float(x, 0.0)
    fy = _safe_float(y, 0.0)
    fz = _safe_float(z, 1.0)
    magnitude = math.sqrt((fx * fx) + (fy * fy) + (fz * fz))
    return abs(magnitude - 1.0)


def event_features_from_sensor_row(
    row: Mapping[str, Any],
    *,
    default_source: str = "training_dataset",
    default_timestamp: datetime | None = None,
) -> EventFeatures:
    device_id = str(
        row.get("device_id")
        or row.get("sensor_id")
        or row.get("id")
        or "unknown"
    )
    source = str(row.get("source") or default_source)

    timestamp_resolution = resolve_event_timestamp(
        row,
        fallback_timestamp=default_timestamp,
    )
    assert timestamp_resolution is not None
    parsed_ts = timestamp_resolution.resolved_timestamp

    accel_delta = _safe_float(row.get("accel_delta"), default=float("nan"))
    if not math.isfinite(accel_delta):
        accel_delta = _accel_delta_from_xyz(
            row.get("acceleration_x"),
            row.get("acceleration_y"),
            row.get("acceleration_z"),
        )

    recent_count_10min = max(0, _safe_int(row.get("recent_count_10min"), 0))
    vibration_raw = max(0.0, _safe_float(row.get("vibration_value", row.get("vibration_raw")), 0.0))
    provided_impact_count = max(0, _safe_int(row.get("impact_count_in_window"), 0))
    current_profile = build_vibration_risk_profile(
        vibration_value=vibration_raw,
        timestamp=parsed_ts,
        impact_count_in_window=provided_impact_count,
    )
    current_impact_count = (
        max(1, provided_impact_count)
        if category_is_caution_or_higher(current_profile.vibration_level_category)
        else provided_impact_count
    )

    return EventFeatures(
        device_id=device_id,
        source=source,
        sound_level=_safe_float(row.get("sound_level"), 0.0),
        vibration_value=vibration_raw,
        duration_ms=max(0, _safe_int(row.get("duration_ms"), 0)),
        accel_delta=max(0.0, accel_delta),
        timestamp=parsed_ts,
        recent_count_10min=recent_count_10min,
        impact_count_in_window=current_impact_count,
        timestamp_source=timestamp_resolution.timestamp_source,
        timestamp_conflict=timestamp_resolution.timestamp_conflict,
    )


def feature_dict_from_event(
    event: EventFeatures,
    *,
    settings: AISettings | None = None,
) -> dict[str, float]:
    cfg = settings or get_settings()
    thresholds = cfg.thresholds

    sound_level = _safe_float(event.sound_level, 0.0)
    vibration_raw = max(0.0, _safe_float(event.vibration_value, 0.0))
    duration_ms = max(0, _safe_int(event.duration_ms, 0))
    recent_count_10min = max(0, _safe_int(event.recent_count_10min, 0))
    impact_count_in_window = max(0, _safe_int(event.impact_count_in_window, 0))

    ts = as_kst(event.timestamp)
    hour_of_day = float(ts.hour)
    night = _is_night(ts, cfg)
    profile = build_vibration_risk_profile(
        vibration_value=vibration_raw,
        timestamp=ts,
        settings=cfg,
        impact_count_in_window=impact_count_in_window,
    )

    impact_leq = thresholds.impact_leq(night)
    impact_lmax = thresholds.impact_lmax(night)
    airborne_leq = thresholds.airborne_leq(night)

    return {
        "sound_level": sound_level,
        "vibration_raw": float(vibration_raw),
        "vibration_acc_mps2": float(profile.vibration_acc_mps2),
        "vibration_dbv": float(profile.vibration_dbv),
        "duration_ms": float(duration_ms),
        "hour_of_day": hour_of_day,
        "is_daytime": 0.0 if night else 1.0,
        "is_nighttime": 1.0 if night else 0.0,
        "vibration_level_category": float(
            VIBRATION_LEVEL_CATEGORY_CODES[profile.vibration_level_category]
        ),
        "impact_count_in_window": float(profile.impact_count),
        "repeated_impact_flag": 1.0 if profile.repeated_impact_flag else 0.0,
        "sound_over_impact_leq": max(0.0, sound_level - impact_leq),
        "sound_over_impact_lmax": max(0.0, sound_level - impact_lmax),
        "sound_over_airborne_leq": max(0.0, sound_level - airborne_leq),
        "duration_over_short_ms": float(max(0, duration_ms - thresholds.duration_short_ms)),
        "duration_over_medium_ms": float(max(0, duration_ms - thresholds.duration_medium_ms)),
        "duration_over_long_ms": float(max(0, duration_ms - thresholds.duration_long_ms)),
    }


def feature_vector_from_event(
    event: EventFeatures,
    *,
    settings: AISettings | None = None,
) -> list[float]:
    feature_dict = feature_dict_from_event(event, settings=settings)
    return [float(feature_dict[name]) for name in FEATURE_COLUMNS]


def feature_dict_from_payload(
    payload: Mapping[str, Any],
    *,
    settings: AISettings | None = None,
) -> dict[str, float]:
    event = event_features_from_sensor_row(payload)
    return feature_dict_from_event(event, settings=settings)


def feature_matrix_from_events(
    events: Sequence[EventFeatures],
    *,
    settings: AISettings | None = None,
) -> list[list[float]]:
    return [feature_vector_from_event(event, settings=settings) for event in events]


def build_feature_spec(*, settings: AISettings | None = None) -> dict[str, Any]:
    cfg = settings or get_settings()
    return {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "nighttime_window": {
            "start_hour": cfg.nighttime.start_hour,
            "end_hour": cfg.nighttime.end_hour,
        },
        "notes": {
            "missing_numeric_default": 0.0,
            "recent_count_10min_policy": (
                "excluded from feature columns for noise_risk feature_spec 2.0.0; "
                "pattern/event context must not leak into this base model"
            ),
            "impact_count_window_seconds": 10,
            "impact_count_in_window_min": 0,
            "vibration_value_contract": "SEN0209 raw ADC count, not m/s^2 or dB(V)",
            "vibration_level_category_encoding": VIBRATION_LEVEL_CATEGORY_CODES,
            "timestamp_policy": (
                "AI resolves timestamps by priority: sensor_timestamp, timestamp, "
                "event_feature.sensor_timestamp, event_feature.timestamp, received_at, "
                "event_feature.received_at, fallback_server_time. Naive datetimes are "
                "interpreted as KST sensor wall time; aware datetimes are converted to KST."
            ),
        },
    }


def write_feature_spec(
    path: Path,
    *,
    settings: AISettings | None = None,
) -> dict[str, Any]:
    spec = build_feature_spec(settings=settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    return spec
