from __future__ import annotations

from typing import Any, Mapping

from .schemas import AIMessageResult, PatternAnalysisResult


def build_fallback_message(
    event_context: Mapping[str, Any],
    pattern_result: PatternAnalysisResult | None = None,
) -> AIMessageResult:
    """Build deterministic neutral message when OpenAI is unavailable."""
    event_type = str(event_context.get("event_type", "unknown"))
    severity = str(event_context.get("severity", "low"))
    event_count = int(event_context.get("event_count", 1))
    time_range = str(event_context.get("time_range", "해당 시간대"))

    event_summary = f"{time_range} 동안 {event_type} 이벤트가 {event_count}회 감지되었습니다."
    resident_message = (
        f"{event_summary} 생활 불편 최소화를 위해 해당 시간대 활동 여부를 확인 부탁드립니다. "
        "필요하다면 조용한 시간대를 함께 조율할 수 있습니다."
    )
    admin_summary = (
        f"이벤트 유형: {event_type}, 심각도: {severity}. "
        f"{pattern_result.summary if pattern_result else '단건 이벤트'}"
    )

    recommended_action = "no_action"
    if severity in {"medium", "high"}:
        recommended_action = "quiet_time_request"
    if severity == "critical":
        recommended_action = "admin_review"
    if pattern_result and pattern_result.needs_escalation:
        recommended_action = "escalation_review"

    return AIMessageResult(
        event_summary=event_summary,
        resident_message=resident_message,
        admin_summary=admin_summary,
        recommended_action=recommended_action,
        tone_check={"is_neutral": True, "contains_blame": False, "contains_threat": False},
        generation_method="fallback_template",
    )
