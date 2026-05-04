from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping, Sequence

from .config import AISettings, get_settings
from .schemas import EventClassificationResult, EventType, SensorReading, Severity


def is_night(dt: datetime, settings: AISettings) -> bool:
    """Return True if timestamp is inside configured night-time hours."""
    start = settings.nighttime.start_hour
    end = settings.nighttime.end_hour
    hour = dt.hour
    return hour >= start or hour < end


def _accel_delta_g(reading: SensorReading) -> float:
    """Return acceleration magnitude delta from 1g baseline."""
    magnitude = math.sqrt(
        reading.acceleration.x**2
        + reading.acceleration.y**2
        + reading.acceleration.z**2
    )
    return abs(magnitude - 1.0)


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _count_recent_events_10min(
    recent_events_10min: Sequence[Mapping[str, Any]] | None,
    now: datetime,
) -> int:
    if not recent_events_10min:
        return 0
    count = 0
    for row in recent_events_10min:
        dt = _parse_dt(row.get("detected_at") or row.get("timestamp"))
        if not dt:
            continue
        seconds = (now - dt).total_seconds()
        if 0 <= seconds <= 600:
            count += 1
    return count


def _score_severity(
    reading: SensorReading,
    night: bool,
    recent_count_10min: int,
    event_type: str,
    settings: AISettings,
) -> tuple[str, int]:
    thresholds = settings.thresholds
    score = 0

    # dB score
    if reading.sound_level >= 65:
        score += 4
    elif reading.sound_level >= 58:
        score += 3
    elif reading.sound_level >= thresholds.db_impact:
        score += 2
    elif reading.sound_level >= thresholds.db_daily:
        score += 1

    # vibration score
    if reading.vibration_value >= 800:
        score += 3
    elif reading.vibration_value >= thresholds.vibration_high:
        score += 2
    elif reading.vibration_value >= thresholds.vibration_mid:
        score += 1

    # duration score
    if reading.duration_ms >= thresholds.duration_long_ms:
        score += 2
    elif reading.duration_ms >= thresholds.duration_medium_ms:
        score += 1

    # context bonus
    if night and event_type != EventType.BACKGROUND_NOISE.value:
        score += 1
    if recent_count_10min >= 5:
        score += 2
    elif recent_count_10min >= 3:
        score += 1
    if event_type == EventType.REPEATED_VIBRATION.value:
        score += 1

    if score >= 9:
        return Severity.CRITICAL.value, score
    if score >= 6:
        return Severity.HIGH.value, score
    if score >= 3:
        return Severity.MEDIUM.value, score
    return Severity.LOW.value, score


def classify_event(
    reading: SensorReading,
    recent_events_10min: Sequence[Mapping[str, Any]] | None = None,
    settings: AISettings | None = None,
) -> EventClassificationResult:
    """
    Classify one sensor reading into an event type and severity.

    Rule priority:
    1) background_noise
    2) repeated_vibration
    3) impact_noise
    4) daily_noise
    5) unknown
    """
    cfg = settings or get_settings()
    thresholds = cfg.thresholds
    night = is_night(reading.timestamp, cfg)
    accel_delta = _accel_delta_g(reading)
    recent_count_10min = _count_recent_events_10min(recent_events_10min, reading.timestamp)

    event_type = EventType.UNKNOWN.value
    severity = Severity.LOW.value
    score = 0
    confidence = 0.5
    rule_hits: list[str] = []

    # 1) background noise
    if reading.sound_level < thresholds.db_background or (
        reading.sound_level < thresholds.db_daily
        and reading.vibration_value < thresholds.vibration_low
        and reading.duration_ms < thresholds.duration_short_ms
    ):
        event_type = EventType.BACKGROUND_NOISE.value
        confidence = 0.95
        rule_hits.append("background_rule")

    # 2) repeated vibration
    elif reading.vibration_value >= thresholds.vibration_mid and recent_count_10min >= 3:
        event_type = EventType.REPEATED_VIBRATION.value
        confidence = 0.82
        rule_hits.append("repeated_vibration_rule")

    # 3) impact noise
    elif (
        reading.sound_level >= thresholds.db_impact
        and reading.vibration_value >= thresholds.vibration_high
        and accel_delta >= 0.12
    ):
        event_type = EventType.IMPACT_NOISE.value
        confidence = 0.87
        rule_hits.extend(["impact_db_rule", "impact_vibration_rule", "impact_accel_rule"])

    # 4) daily noise
    elif reading.sound_level >= thresholds.db_daily:
        event_type = EventType.DAILY_NOISE.value
        confidence = 0.74
        rule_hits.append("daily_noise_rule")

    else:
        event_type = EventType.UNKNOWN.value
        confidence = 0.55
        rule_hits.append("unknown_rule")

    if event_type == EventType.BACKGROUND_NOISE.value:
        severity = Severity.LOW.value
        score = 0
    else:
        severity, score = _score_severity(
            reading=reading,
            night=night,
            recent_count_10min=recent_count_10min,
            event_type=event_type,
            settings=cfg,
        )

    return EventClassificationResult(
        event_type=event_type,
        severity=severity,
        severity_score=score,
        confidence=confidence,
        is_night=night,
        rule_hits=rule_hits,
        features={
            "sound_level": reading.sound_level,
            "vibration_value": reading.vibration_value,
            "duration_ms": reading.duration_ms,
            "accel_delta": round(accel_delta, 4),
            "recent_10min_count": recent_count_10min,
        },
    )
