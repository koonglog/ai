from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

KST = timezone(timedelta(hours=9))

MEASUREMENT_TIMESTAMP_SOURCES = {
    "sensor_timestamp",
    "timestamp",
    "event_feature.sensor_timestamp",
    "event_feature.timestamp",
}


@dataclass(frozen=True, slots=True)
class ResolvedEventTimestamp:
    resolved_timestamp: datetime
    timestamp_source: str
    timestamp_conflict: bool = False

    @property
    def is_nighttime(self) -> bool:
        hour = self.resolved_timestamp.hour
        return hour >= 22 or hour < 6

    @property
    def is_daytime(self) -> bool:
        return not self.is_nighttime

    @property
    def time_period(self) -> str:
        return "nighttime" if self.is_nighttime else "daytime"


def _get_value(payload: Any, field: str) -> Any:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return payload.get(field)
    return getattr(payload, field, None)


def parse_event_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def as_kst(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=KST)
    return value.astimezone(KST)


def to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=KST)
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _candidate_values(payload: Any) -> list[tuple[str, Any]]:
    event_feature = _get_value(payload, "event_feature")
    return [
        ("sensor_timestamp", _get_value(payload, "sensor_timestamp")),
        ("timestamp", _get_value(payload, "timestamp")),
        ("event_feature.sensor_timestamp", _get_value(event_feature, "sensor_timestamp")),
        ("event_feature.timestamp", _get_value(event_feature, "timestamp")),
        ("received_at", _get_value(payload, "received_at")),
        ("event_feature.received_at", _get_value(event_feature, "received_at")),
    ]


def resolve_event_timestamp(
    payload: Any,
    *,
    fallback_timestamp: datetime | None = None,
    fallback_to_now: bool = True,
) -> ResolvedEventTimestamp | None:
    parsed_candidates: list[tuple[str, datetime]] = []
    for source, raw_value in _candidate_values(payload):
        parsed = parse_event_datetime(raw_value)
        if parsed is not None:
            parsed_candidates.append((source, as_kst(parsed)))

    if parsed_candidates:
        source, resolved = parsed_candidates[0]
        measurement_values = [
            ts
            for candidate_source, ts in parsed_candidates
            if candidate_source in MEASUREMENT_TIMESTAMP_SOURCES
        ]
        timestamp_conflict = (
            len(measurement_values) > 1
            and any(ts != measurement_values[0] for ts in measurement_values[1:])
        )
        return ResolvedEventTimestamp(
            resolved_timestamp=resolved,
            timestamp_source=source,
            timestamp_conflict=timestamp_conflict,
        )

    if fallback_timestamp is not None:
        parsed_fallback = parse_event_datetime(fallback_timestamp)
        if parsed_fallback is not None:
            return ResolvedEventTimestamp(
                resolved_timestamp=as_kst(parsed_fallback),
                timestamp_source="fallback_server_time",
                timestamp_conflict=False,
            )

    if not fallback_to_now:
        return None

    return ResolvedEventTimestamp(
        resolved_timestamp=datetime.now(KST),
        timestamp_source="fallback_server_time",
        timestamp_conflict=False,
    )
