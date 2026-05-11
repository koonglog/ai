from datetime import datetime

from ai.event_classifier import classify_event
from ai.schemas import EventFeatures


def _event(
    db: float,
    vib: int,
    dur: int,
    ts: str,
    accel_delta: float = 0.0,
    recent_count_10min: int = 0,
) -> EventFeatures:
    return EventFeatures(
        device_id="SENSOR-A101-01",
        source="simulator",
        sound_level=db,
        vibration_value=vib,
        duration_ms=dur,
        accel_delta=accel_delta,
        timestamp=datetime.fromisoformat(ts),
        recent_count_10min=recent_count_10min,
    )


def test_background_noise_classification() -> None:
    event = _event(38.0, 80, 1200, "2026-05-04T14:10:00+09:00")
    result = classify_event(event)
    assert result.event_type == "background_noise"
    assert result.severity == "low"
    assert result.is_meaningful is False


def test_daily_noise_classification() -> None:
    event = _event(52.0, 120, 6000, "2026-05-04T20:10:00+09:00")
    result = classify_event(event)
    assert result.event_type == "daily_noise"
    assert result.severity in {"low", "medium"}
    assert result.is_meaningful is True


def test_impact_noise_high_severity() -> None:
    event = _event(63.0, 720, 9000, "2026-05-04T23:10:00+09:00", accel_delta=0.2)
    result = classify_event(event)
    assert result.event_type == "impact_noise"
    assert result.severity in {"high", "critical"}
    assert result.is_meaningful is True


def test_repeated_vibration_classification_with_meaningful_count() -> None:
    event = _event(
        48.0,
        420,
        3000,
        "2026-05-04T23:20:00+09:00",
        accel_delta=0.02,
        recent_count_10min=3,
    )
    result = classify_event(event)
    assert result.event_type == "repeated_vibration"
    assert result.severity in {"medium", "high"}
    assert result.is_meaningful is True


def test_repeated_vibration_not_triggered_by_raw_influx_without_meaningful_count() -> None:
    # Even if many raw samples exist upstream, classifier only accepts meaningful
    # recent_count_10min contract. Zero means no repeated-vibration trigger.
    event = _event(
        48.0,
        420,
        3000,
        "2026-05-04T23:20:00+09:00",
        accel_delta=0.02,
        recent_count_10min=0,
    )
    result = classify_event(event)
    assert result.event_type != "repeated_vibration"


def test_airborne_threshold_differs_between_day_and_night() -> None:
    # 41dB is below day airborne(45) but above night airborne(40).
    day_event = _event(41.0, 100, 1200, "2026-05-04T21:30:00+09:00")
    night_event = _event(41.0, 100, 3200, "2026-05-04T23:30:00+09:00")

    day_result = classify_event(day_event)
    night_result = classify_event(night_event)

    assert day_result.event_type == "background_noise"
    assert night_result.event_type == "daily_noise"


def test_impact_lmax_threshold_differs_between_day_and_night() -> None:
    # 53dB + strong vibration: night(>=52) can be impact, day(<57) should not.
    day_event = _event(53.0, 700, 6000, "2026-05-04T14:30:00+09:00", accel_delta=0.2)
    night_event = _event(53.0, 700, 6000, "2026-05-04T23:30:00+09:00", accel_delta=0.2)

    day_result = classify_event(day_event)
    night_result = classify_event(night_event)

    assert day_result.event_type != "impact_noise"
    assert night_result.event_type == "impact_noise"
