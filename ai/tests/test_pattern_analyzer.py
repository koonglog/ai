from datetime import datetime, timedelta

from ai.pattern_analyzer import analyze_patterns


def test_pattern_analysis_returns_counts() -> None:
    ref = datetime.fromisoformat("2026-05-08T00:00:00+09:00")
    logs = [
        {
            "detected_at": (ref - timedelta(days=1)).replace(hour=23, minute=10).isoformat(),
            "event_type": "impact_noise",
            "severity": "high",
        },
        {
            "detected_at": (ref - timedelta(days=2)).replace(hour=23, minute=20).isoformat(),
            "event_type": "impact_noise",
            "severity": "high",
        },
        {
            "detected_at": (ref - timedelta(days=3)).replace(hour=21, minute=40).isoformat(),
            "event_type": "daily_noise",
            "severity": "medium",
        },
    ]
    result = analyze_patterns(household_id=1, noise_logs=logs, reference_time=ref)
    assert result.total_events == 3
    assert result.repeated_days == 3
    assert result.night_events == 2
    assert result.needs_mediation is True


def test_cluster_count_in_10min() -> None:
    ref = datetime.fromisoformat("2026-05-08T00:00:00+09:00")
    base = (ref - timedelta(days=1)).replace(hour=23, minute=0)
    logs = [
        {"detected_at": (base + timedelta(minutes=0)).isoformat(), "event_type": "repeated_vibration", "severity": "medium"},
        {"detected_at": (base + timedelta(minutes=2)).isoformat(), "event_type": "repeated_vibration", "severity": "medium"},
        {"detected_at": (base + timedelta(minutes=4)).isoformat(), "event_type": "repeated_vibration", "severity": "medium"},
        {"detected_at": (base + timedelta(minutes=8)).isoformat(), "event_type": "repeated_vibration", "severity": "medium"},
    ]
    result = analyze_patterns(household_id=1, noise_logs=logs, reference_time=ref)
    assert result.max_events_in_10min == 4
    assert result.pattern_label in {"vibration_cluster", "recurring_noise"}


def test_escalation_when_post_mediation_recurrence_and_high_pattern() -> None:
    ref = datetime.fromisoformat("2026-05-08T00:00:00+09:00")
    logs = []
    for day, minute in [(1, 10), (2, 12), (3, 14), (4, 16), (5, 18)]:
        logs.append(
            {
                "detected_at": (ref - timedelta(days=day)).replace(hour=23, minute=minute).isoformat(),
                "event_type": "impact_noise",
                "severity": "high",
            }
        )
    mediation_messages = [
        {"created_at": (ref - timedelta(days=6)).replace(hour=22, minute=0).isoformat()}
    ]

    result = analyze_patterns(
        household_id=1,
        noise_logs=logs,
        mediation_messages=mediation_messages,
        reference_time=ref,
    )
    assert result.post_mediation_recurrence is True
    assert result.needs_escalation is True


def test_no_events_case() -> None:
    ref = datetime.fromisoformat("2026-05-08T00:00:00+09:00")
    result = analyze_patterns(household_id=1, noise_logs=[], reference_time=ref)
    assert result.total_events == 0
    assert result.pattern_label == "no_pattern"
    assert result.needs_mediation is False
