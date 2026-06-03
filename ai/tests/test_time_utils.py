from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ai.time_utils import as_kst, resolve_event_timestamp, to_naive_utc


def test_resolve_event_timestamp_prefers_sensor_timestamp() -> None:
    resolved = resolve_event_timestamp(
        {
            "sensor_timestamp": "2026-05-11T22:00:00",
            "timestamp": "2026-05-11T21:59:00",
        }
    )

    assert resolved is not None
    assert resolved.resolved_timestamp.isoformat() == "2026-05-11T22:00:00+09:00"
    assert resolved.timestamp_source == "sensor_timestamp"
    assert resolved.timestamp_conflict is True
    assert resolved.time_period == "nighttime"


def test_resolve_event_timestamp_uses_timestamp_when_sensor_timestamp_missing() -> None:
    resolved = resolve_event_timestamp({"timestamp": "2026-05-11T21:59:00"})

    assert resolved is not None
    assert resolved.resolved_timestamp.isoformat() == "2026-05-11T21:59:00+09:00"
    assert resolved.timestamp_source == "timestamp"
    assert resolved.time_period == "daytime"


def test_resolve_event_timestamp_uses_event_feature_timestamp() -> None:
    resolved = resolve_event_timestamp(
        {"event_feature": {"timestamp": "2026-05-11T22:00:00"}}
    )

    assert resolved is not None
    assert resolved.timestamp_source == "event_feature.timestamp"
    assert resolved.time_period == "nighttime"


def test_resolve_event_timestamp_uses_received_at_when_measurement_time_missing() -> None:
    resolved = resolve_event_timestamp({"received_at": "2026-05-11T06:00:00"})

    assert resolved is not None
    assert resolved.timestamp_source == "received_at"
    assert resolved.time_period == "daytime"


def test_resolve_event_timestamp_fallback_server_time() -> None:
    fallback = datetime(2026, 5, 11, 5, 59, tzinfo=timezone.utc)
    resolved = resolve_event_timestamp({}, fallback_timestamp=fallback)

    assert resolved is not None
    assert resolved.resolved_timestamp.isoformat() == "2026-05-11T14:59:00+09:00"
    assert resolved.timestamp_source == "fallback_server_time"


def test_kst_day_night_boundaries() -> None:
    cases = [
        ("2026-05-11T21:59:00", "daytime"),
        ("2026-05-11T22:00:00", "nighttime"),
        ("2026-05-11T05:59:00", "nighttime"),
        ("2026-05-11T06:00:00", "daytime"),
    ]

    for value, expected in cases:
        resolved = resolve_event_timestamp({"sensor_timestamp": value})
        assert resolved is not None
        assert resolved.time_period == expected


def test_timezone_aware_datetime_is_converted_to_kst() -> None:
    resolved = resolve_event_timestamp({"sensor_timestamp": "2026-05-11T13:00:00Z"})

    assert resolved is not None
    assert resolved.resolved_timestamp.isoformat() == "2026-05-11T22:00:00+09:00"
    assert resolved.time_period == "nighttime"
    assert to_naive_utc(resolved.resolved_timestamp) == datetime(2026, 5, 11, 13, 0)
    assert as_kst(datetime(2026, 5, 11, 22, 0)).tzinfo == timezone(timedelta(hours=9))
