from datetime import datetime

from ai.event_classifier import classify_event
from ai.schemas import Acceleration, SensorReading


def _reading(db: float, vib: int, dur: int, ts: str, z: float = 1.0) -> SensorReading:
    return SensorReading(
        device_id="SENSOR-A101-01",
        source="simulator",
        sound_level=db,
        vibration_value=vib,
        acceleration=Acceleration(x=0.01, y=0.01, z=z),
        duration_ms=dur,
        timestamp=datetime.fromisoformat(ts),
    )


def test_background_noise_classification() -> None:
    reading = _reading(38.0, 80, 1200, "2026-05-04T14:10:00+09:00")
    result = classify_event(reading)
    assert result.event_type == "background_noise"
    assert result.severity == "low"


def test_daily_noise_classification() -> None:
    reading = _reading(52.0, 120, 6000, "2026-05-04T20:10:00+09:00")
    result = classify_event(reading)
    assert result.event_type == "daily_noise"
    assert result.severity in {"low", "medium"}


def test_impact_noise_high_severity() -> None:
    reading = _reading(63.0, 720, 9000, "2026-05-04T23:10:00+09:00", z=1.2)
    result = classify_event(reading)
    assert result.event_type == "impact_noise"
    assert result.severity in {"high", "critical"}


def test_repeated_vibration_classification() -> None:
    reading = _reading(48.0, 420, 3000, "2026-05-04T23:20:00+09:00", z=1.02)
    recent = [
        {"detected_at": "2026-05-04T23:11:00+09:00"},
        {"detected_at": "2026-05-04T23:14:00+09:00"},
        {"detected_at": "2026-05-04T23:18:30+09:00"},
    ]
    result = classify_event(reading, recent_events_10min=recent)
    assert result.event_type == "repeated_vibration"
    assert result.severity in {"medium", "high"}
