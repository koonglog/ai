from __future__ import annotations

from typing import Any, Mapping

from .config import AISettings, get_settings, severity_rank
from .fallback_templates import build_fallback_message
from .openai_client import OpenAIMessageClient
from .schemas import AIMessageResult, PatternAnalysisResult

BANNED_WORDS = ["민폐", "고의", "경고", "보복", "윗집", "아랫집", "신고하겠다"]


def contains_banned_expression(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in BANNED_WORDS)


def should_use_openai(
    event_context: Mapping[str, Any],
    pattern_result: PatternAnalysisResult | None,
    settings: AISettings,
) -> bool:
    min_rank = severity_rank(settings.openai.llm_min_severity)
    current_rank = severity_rank(str(event_context.get("severity", "low")))
    if current_rank < min_rank:
        return False
    if pattern_result and (pattern_result.needs_mediation or pattern_result.needs_escalation):
        return settings.openai.can_call
    return settings.openai.can_call


def _to_ai_message(payload: Mapping[str, Any]) -> AIMessageResult:
    return AIMessageResult(
        event_summary=str(payload.get("event_summary", "")),
        resident_message=str(payload.get("resident_message", "")),
        admin_summary=str(payload.get("admin_summary", "")),
        recommended_action=str(payload.get("recommended_action", "no_action")),
        tone_check=dict(payload.get("tone_check", {})),
        generation_method=str(payload.get("generation_method", "openai")),
    )


def _is_payload_safe(payload: Mapping[str, Any]) -> bool:
    resident = str(payload.get("resident_message", ""))
    admin = str(payload.get("admin_summary", ""))
    return not (contains_banned_expression(resident) or contains_banned_expression(admin))


def generate_mediation_message(
    event_context: dict[str, Any],
    pattern_result: PatternAnalysisResult | None = None,
    openai_client: OpenAIMessageClient | None = None,
    settings: AISettings | None = None,
) -> AIMessageResult:
    """
    Generate mediation message with selective OpenAI call.

    Fallback is always used when OpenAI is disabled, fails, or returns unsafe tone.
    """
    cfg = settings or get_settings()
    if not should_use_openai(event_context, pattern_result, cfg):
        return build_fallback_message(event_context, pattern_result)

    if openai_client is None:
        try:
            openai_client = OpenAIMessageClient(cfg)
        except Exception:
            return build_fallback_message(event_context, pattern_result)

    try:
        payload = openai_client.generate_message(event_context, pattern_result)
        if not _is_payload_safe(payload):
            return build_fallback_message(event_context, pattern_result)
        payload = dict(payload)
        payload["generation_method"] = "openai"
        return _to_ai_message(payload)
    except Exception:
        return build_fallback_message(event_context, pattern_result)
