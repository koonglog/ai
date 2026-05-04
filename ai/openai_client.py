from __future__ import annotations

import json
import time
from typing import Any, Mapping

from .config import AISettings, get_settings
from .schemas import PatternAnalysisResult

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event_summary": {"type": "string"},
        "resident_message": {"type": "string"},
        "admin_summary": {"type": "string"},
        "recommended_action": {
            "type": "string",
            "enum": [
                "quiet_time_request",
                "admin_review",
                "escalation_review",
                "no_action",
            ],
        },
        "tone_check": {
            "type": "object",
            "properties": {
                "is_neutral": {"type": "boolean"},
                "contains_blame": {"type": "boolean"},
                "contains_threat": {"type": "boolean"},
            },
            "required": ["is_neutral", "contains_blame", "contains_threat"],
            "additionalProperties": False,
        },
    },
    "required": [
        "event_summary",
        "resident_message",
        "admin_summary",
        "recommended_action",
        "tone_check",
    ],
    "additionalProperties": False,
}


def _build_system_prompt() -> str:
    return (
        "당신은 층간소음 중재 메시지 생성기다. 반드시 JSON 객체만 출력한다. "
        "한국어로 작성하고 비난, 추측, 위협, 단정 표현을 금지한다. "
        "피해 세대와 발생 세대를 단정하지 않는다. "
        "갈등 유발 표현(윗집, 아랫집)은 최소화한다."
    )


def _build_user_payload(
    event_context: Mapping[str, Any],
    pattern_result: PatternAnalysisResult | None = None,
) -> dict[str, Any]:
    return {
        "event_type": event_context.get("event_type"),
        "time_range": event_context.get("time_range"),
        "event_count": event_context.get("event_count"),
        "severity": event_context.get("severity"),
        "pattern_summary": (
            pattern_result.summary if pattern_result else event_context.get("pattern_summary")
        ),
        "constraints": {
            "neutral_tone": True,
            "no_blame": True,
            "no_threat": True,
            "korean_language": True,
        },
    }


class OpenAIMessageClient:
    """OpenAI wrapper for mediation message generation."""

    def __init__(self, settings: AISettings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        if client is not None:
            self.client = client
            return

        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        if not self.settings.openai.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        self.client = OpenAI(
            api_key=self.settings.openai.api_key,
            timeout=self.settings.openai.timeout_seconds,
        )

    def _extract_output_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output_items = getattr(response, "output", None)
        if not output_items:
            return ""

        texts: list[str] = []
        for item in output_items:
            if isinstance(item, dict):
                content = item.get("content") or []
            else:
                content = getattr(item, "content", None) or []
            for part in content:
                if isinstance(part, dict):
                    part_text = part.get("text")
                else:
                    part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    texts.append(part_text)
        return "\n".join(texts).strip()

    def _validate_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("OpenAI response JSON must be an object")

        required_keys = [
            "event_summary",
            "resident_message",
            "admin_summary",
            "recommended_action",
            "tone_check",
        ]
        for key in required_keys:
            if key not in payload:
                raise ValueError(f"OpenAI response missing key: {key}")

        return payload

    def generate_message(
        self,
        event_context: Mapping[str, Any],
        pattern_result: PatternAnalysisResult | None = None,
    ) -> dict[str, Any]:
        """
        Generate JSON message from OpenAI.

        This method currently contains an MVP-safe prompt and JSON mode.
        """
        payload = _build_user_payload(event_context, pattern_result)
        attempts = max(1, self.settings.openai.max_retries + 1)
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = self.client.responses.create(
                    model=self.settings.openai.model,
                    input=[
                        {"role": "system", "content": _build_system_prompt()},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    reasoning={"effort": self.settings.openai.reasoning_effort},
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "mediation_message",
                            "strict": True,
                            "schema": RESPONSE_JSON_SCHEMA,
                        }
                    },
                    max_output_tokens=self.settings.openai.max_output_tokens,
                )
                raw = self._extract_output_text(response)
                if not raw:
                    status = getattr(response, "status", None)
                    incomplete_details = getattr(response, "incomplete_details", None)
                    raise RuntimeError(
                        f"Empty output_text from OpenAI (status={status}, incomplete={incomplete_details})"
                    )
                parsed = json.loads(raw)
                return self._validate_payload(parsed)
            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(0.4 * (attempt + 1))
                continue

        raise RuntimeError(f"OpenAI generation failed after retries: {last_error}")
