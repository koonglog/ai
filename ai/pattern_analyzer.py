from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from .config import AISettings, get_settings
from .schemas import PatternAnalysisResult


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _hour_slot(dt: datetime) -> str:
    next_hour = (dt.hour + 1) % 24
    return f"{dt.hour:02d}:00-{next_hour:02d}:00"


def _max_events_in_window(times: list[datetime], window_minutes: int) -> int:
    if not times:
        return 0
    if len(times) == 1:
        return 1

    sorted_times = sorted(times)
    window_seconds = window_minutes * 60
    left = 0
    max_count = 1

    for right in range(len(sorted_times)):
        while (sorted_times[right] - sorted_times[left]).total_seconds() > window_seconds:
            left += 1
        max_count = max(max_count, right - left + 1)

    return max_count


def _post_mediation_recurrence(
    event_times: list[datetime],
    mediation_messages: Sequence[Mapping[str, Any]] | None,
) -> bool:
    if not event_times or not mediation_messages:
        return False

    message_times: list[datetime] = []
    for row in mediation_messages:
        dt = _parse_dt(row.get("created_at"))
        if dt:
            message_times.append(dt)

    if not message_times:
        return False

    last_message_time = max(message_times)
    return any(dt > last_message_time for dt in event_times)


def analyze_patterns(
    household_id: int,
    noise_logs: Sequence[Mapping[str, Any]],
    mediation_messages: Sequence[Mapping[str, Any]] | None = None,
    analysis_period_days: int = 7,
    reference_time: datetime | None = None,
    settings: AISettings | None = None,
) -> PatternAnalysisResult:
    """
    Analyze repeated noise pattern for one household.

    Includes:
    - repeated days in period
    - night events
    - max cluster count in 10 minutes (configurable)
    - post-mediation recurrence
    - mediation/escalation 판단
    """
    cfg = settings or get_settings()
    pattern_cfg = cfg.pattern

    now = reference_time or datetime.now().astimezone()
    start = now - timedelta(days=analysis_period_days)

    in_window: list[dict[str, Any]] = []
    for row in noise_logs:
        dt = _parse_dt(row.get("detected_at"))
        if dt and start <= dt <= now:
            in_window.append(
                {
                    "detected_at": dt,
                    "event_type": str(row.get("event_type", "unknown")),
                    "severity": str(row.get("severity", "low")),
                }
            )

    total_events = len(in_window)
    if total_events == 0:
        return PatternAnalysisResult(
            household_id=household_id,
            analysis_period_days=analysis_period_days,
            total_events=0,
            night_events=0,
            most_frequent_time_slot=None,
            repeated_days=0,
            max_events_in_10min=0,
            pattern_label="no_pattern",
            needs_mediation=False,
            needs_escalation=False,
            summary="분석 기간 내 유의미한 소음 이벤트가 없습니다.",
            post_mediation_recurrence=False,
        )

    event_times = [row["detected_at"] for row in in_window]
    night_events = sum(1 for dt in event_times if dt.hour >= 22 or dt.hour < 7)

    day_counter = Counter(dt.date().isoformat() for dt in event_times)
    repeated_days = len(day_counter)

    slot_counter = Counter(_hour_slot(dt) for dt in event_times)
    most_slot = slot_counter.most_common(1)[0][0]

    max_10min = _max_events_in_window(event_times, pattern_cfg.cluster_window_minutes)
    post_recurrence = _post_mediation_recurrence(event_times, mediation_messages)

    impact_count = sum(1 for row in in_window if row["event_type"] == "impact_noise")
    repeated_vibration_count = sum(
        1 for row in in_window if row["event_type"] == "repeated_vibration"
    )
    high_or_critical_count = sum(
        1 for row in in_window if row["severity"] in {"high", "critical"}
    )

    needs_mediation = (
        total_events >= 3
        or night_events >= 2
        or max_10min >= 3
        or repeated_days >= pattern_cfg.repeated_days_for_warning
    )

    needs_escalation = (
        post_recurrence
        and repeated_days >= pattern_cfg.repeated_days_for_escalation
        and (high_or_critical_count >= 2 or max_10min >= 5)
    )

    if (
        night_events >= max(2, int(total_events * 0.4))
        and impact_count >= max(2, int(total_events * 0.3))
        and repeated_days >= pattern_cfg.repeated_days_for_warning
    ):
        pattern_label = "night_repeated_impact"
    elif repeated_vibration_count >= 3 and max_10min >= 3:
        pattern_label = "vibration_cluster"
    elif repeated_days >= pattern_cfg.repeated_days_for_warning:
        pattern_label = "recurring_noise"
    else:
        pattern_label = "sporadic_noise"

    return PatternAnalysisResult(
        household_id=household_id,
        analysis_period_days=analysis_period_days,
        total_events=total_events,
        night_events=night_events,
        most_frequent_time_slot=most_slot,
        repeated_days=repeated_days,
        max_events_in_10min=max_10min,
        pattern_label=pattern_label,
        needs_mediation=needs_mediation,
        needs_escalation=needs_escalation,
        summary=(
            f"최근 {analysis_period_days}일 중 {repeated_days}일 동안 "
            f"{most_slot} 시간대 이벤트가 반복 감지되었습니다."
        ),
        post_mediation_recurrence=post_recurrence,
    )
