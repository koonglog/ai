from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .time_utils import resolve_event_timestamp


class EventType(str, Enum):
    IMPACT_NOISE = "impact_noise"
    REPEATED_VIBRATION = "repeated_vibration"
    DAILY_NOISE = "daily_noise"
    BACKGROUND_NOISE = "background_noise"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class Acceleration:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class EventFeatures:
    device_id: str
    source: str
    sound_level: float
    vibration_value: float
    duration_ms: int
    accel_delta: float
    timestamp: datetime
    # Contract: count of meaningful events in the last 10 minutes.
    # Raw sample/log counts are not allowed here.
    recent_count_10min: int = 0
    # Count of caution-or-higher vibration impacts in the rolling 10-second
    # window, including the current reading when applicable.
    impact_count_in_window: int = 0
    timestamp_source: str = "timestamp"
    timestamp_conflict: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventFeatures":
        acc = payload.get("acceleration") or {}
        accel_delta = payload.get("accel_delta")
        if accel_delta is None:
            try:
                x = float(acc.get("x", 0.0))
                y = float(acc.get("y", 0.0))
                z = float(acc.get("z", 1.0))
                magnitude = math.sqrt(x * x + y * y + z * z)
                accel_delta = abs(magnitude - 1.0)
            except (TypeError, ValueError):
                accel_delta = 0.0
        resolved_ts = resolve_event_timestamp(payload)
        assert resolved_ts is not None
        return cls(
            device_id=str(payload.get("device_id") or payload.get("sensor_id") or "unknown"),
            source=str(payload.get("source", "unknown")),
            sound_level=float(payload["sound_level"]),
            vibration_value=float(payload.get("vibration_value", payload.get("vibration_raw"))),
            duration_ms=int(payload["duration_ms"]),
            accel_delta=float(accel_delta),
            timestamp=resolved_ts.resolved_timestamp,
            recent_count_10min=int(payload.get("recent_count_10min") or 0),
            impact_count_in_window=int(payload.get("impact_count_in_window") or 0),
            timestamp_source=resolved_ts.timestamp_source,
            timestamp_conflict=resolved_ts.timestamp_conflict,
        )


@dataclass(slots=True)
class EventClassificationResult:
    event_type: str
    severity: str
    severity_score: int
    confidence: float
    is_night: bool
    is_meaningful: bool
    noise_risk_level: str = "normal"
    vibration_raw: float = 0.0
    vibration_acc_mps2: float = 0.0
    vibration_dbv: float = 0.0
    reason: str = ""
    time_period: str = "daytime"
    impact_count: int = 0
    rule_hits: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PatternAnalysisResult:
    household_id: int
    analysis_period_days: int
    total_events: int
    night_events: int
    most_frequent_time_slot: str | None
    repeated_days: int
    max_events_in_10min: int
    pattern_label: str
    needs_mediation: bool
    needs_escalation: bool
    summary: str
    post_mediation_recurrence: bool


@dataclass(slots=True)
class AIMessageResult:
    event_summary: str
    resident_message: str
    admin_summary: str
    recommended_action: str
    tone_check: dict[str, bool]
    generation_method: str
