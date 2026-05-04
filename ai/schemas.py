from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


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
class SensorReading:
    device_id: str
    source: str
    sound_level: float
    vibration_value: int
    acceleration: Acceleration
    duration_ms: int
    timestamp: datetime

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SensorReading":
        acc = payload.get("acceleration") or {}
        return cls(
            device_id=str(payload["device_id"]),
            source=str(payload.get("source", "unknown")),
            sound_level=float(payload["sound_level"]),
            vibration_value=int(payload["vibration_value"]),
            acceleration=Acceleration(
                x=float(acc.get("x", 0.0)),
                y=float(acc.get("y", 0.0)),
                z=float(acc.get("z", 1.0)),
            ),
            duration_ms=int(payload["duration_ms"]),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
        )


@dataclass(slots=True)
class EventClassificationResult:
    event_type: str
    severity: str
    severity_score: int
    confidence: float
    is_night: bool
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
