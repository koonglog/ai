from __future__ import annotations

from typing import Any, Mapping

from .schemas import AIMessageResult, PatternAnalysisResult


def _build_manual_report_text(event_context: Mapping[str, Any]) -> str:
    manual_report = event_context.get("manual_report")
    if not isinstance(manual_report, Mapping):
        return ""

    noise_type = str(manual_report.get("noise_type", "")).strip()
    noise_time_slot = str(manual_report.get("noise_time_slot", "")).strip()
    noise_frequency = str(manual_report.get("noise_frequency", "")).strip()
    situation_description = str(manual_report.get("situation_description", "")).strip()

    parts: list[str] = []
    if noise_type:
        parts.append(f"신고 소음 유형: {noise_type}")
    if noise_time_slot:
        parts.append(f"신고 시간대: {noise_time_slot}")
    if noise_frequency:
        parts.append(f"신고 빈도: {noise_frequency}")
    if situation_description:
        parts.append(f"신고 상황: {situation_description}")
    return " / ".join(parts)


def build_fallback_message(
    event_context: Mapping[str, Any],
    pattern_result: PatternAnalysisResult | None = None,
) -> AIMessageResult:
    """Build deterministic neutral message when OpenAI is unavailable."""
    event_type = str(event_context.get("event_type", "unknown"))
    severity = str(event_context.get("severity", "low"))
    event_count = int(event_context.get("event_count", 1))
    time_range = str(event_context.get("time_range", "해당 시간대"))
    manual_report_text = _build_manual_report_text(event_context)

    event_summary = f"{time_range} 동안 {event_type} 이벤트가 {event_count}회 감지되었습니다."
    resident_message = (
        f"{event_summary} 생활 불편 최소화를 위해 해당 시간대 활동 여부를 확인 부탁드립니다. "
        "필요하다면 조용한 시간대를 함께 조율할 수 있습니다."
    )
    admin_summary = (
        f"이벤트 유형: {event_type}, 심각도: {severity}. "
        f"{pattern_result.summary if pattern_result else '단건 이벤트'}"
    )
    if manual_report_text:
        admin_summary = f"{admin_summary} {manual_report_text}"

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
