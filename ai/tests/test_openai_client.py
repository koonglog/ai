import json

import pytest

from ai.config import (
    AISettings,
    NightTimeSettings,
    OpenAISettings,
    PatternSettings,
    ThresholdSettings,
)
from ai.openai_client import OpenAIMessageClient


class _FakeResponses:
    def __init__(self, outputs):
        self._outputs = outputs
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        output_text = self._outputs[self.calls - 1]

        class _Response:
            def __init__(self, text):
                self.output_text = text

        return _Response(output_text)


class _FakeClient:
    def __init__(self, outputs):
        self.responses = _FakeResponses(outputs)


class _FakeResponsesWithOutputItems:
    def __init__(self, output_items):
        self._output_items = output_items
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1

        class _Response:
            def __init__(self, items):
                self.output_text = ""
                self.output = items

        return _Response(self._output_items)


class _FakeClientWithOutputItems:
    def __init__(self, output_items):
        self.responses = _FakeResponsesWithOutputItems(output_items)


def _settings(max_retries: int = 1) -> AISettings:
    return AISettings(
        thresholds=ThresholdSettings(),
        nighttime=NightTimeSettings(),
        pattern=PatternSettings(),
        openai=OpenAISettings(
            enabled=True,
            api_key="dummy",
            model="gpt-5-mini",
            timeout_seconds=8.0,
            max_retries=max_retries,
            max_output_tokens=400,
            llm_min_severity="medium",
        ),
    )


def test_openai_client_valid_json_schema_payload() -> None:
    payload = {
        "event_summary": "요약",
        "resident_message": "확인 부탁드립니다.",
        "admin_summary": "관리자 요약",
        "recommended_action": "quiet_time_request",
        "tone_check": {
            "is_neutral": True,
            "contains_blame": False,
            "contains_threat": False,
        },
    }
    fake = _FakeClient([json.dumps(payload, ensure_ascii=False)])
    client = OpenAIMessageClient(settings=_settings(max_retries=0), client=fake)

    result = client.generate_message(
        event_context={
            "event_type": "impact_noise",
            "time_range": "23:10-23:20",
            "event_count": 3,
            "severity": "high",
        }
    )
    assert result["recommended_action"] == "quiet_time_request"
    assert fake.responses.calls == 1


def test_openai_client_retry_then_success(monkeypatch) -> None:
    monkeypatch.setattr("ai.openai_client.time.sleep", lambda _: None)
    bad = "{not-json"
    good = json.dumps(
        {
            "event_summary": "요약",
            "resident_message": "확인 부탁드립니다.",
            "admin_summary": "관리자 요약",
            "recommended_action": "admin_review",
            "tone_check": {
                "is_neutral": True,
                "contains_blame": False,
                "contains_threat": False,
            },
        },
        ensure_ascii=False,
    )
    fake = _FakeClient([bad, good])
    client = OpenAIMessageClient(settings=_settings(max_retries=1), client=fake)

    result = client.generate_message(
        event_context={
            "event_type": "impact_noise",
            "time_range": "23:10-23:20",
            "event_count": 4,
            "severity": "high",
        }
    )
    assert result["recommended_action"] == "admin_review"
    assert fake.responses.calls == 2


def test_openai_client_raise_after_retry_exhausted(monkeypatch) -> None:
    monkeypatch.setattr("ai.openai_client.time.sleep", lambda _: None)
    fake = _FakeClient(['{"event_summary":"x"}', '{"event_summary":"x"}'])
    client = OpenAIMessageClient(settings=_settings(max_retries=1), client=fake)

    with pytest.raises(RuntimeError):
        client.generate_message(
            event_context={
                "event_type": "impact_noise",
                "time_range": "23:10-23:20",
                "event_count": 2,
                "severity": "medium",
            }
        )


def test_openai_client_includes_manual_report_in_user_payload() -> None:
    payload = {
        "event_summary": "요약",
        "resident_message": "확인 부탁드립니다.",
        "admin_summary": "관리자 요약",
        "recommended_action": "quiet_time_request",
        "tone_check": {
            "is_neutral": True,
            "contains_blame": False,
            "contains_threat": False,
        },
    }
    fake = _FakeClient([json.dumps(payload, ensure_ascii=False)])
    client = OpenAIMessageClient(settings=_settings(max_retries=0), client=fake)

    client.generate_message(
        event_context={
            "event_type": "daily_noise",
            "time_range": "23:10-23:20",
            "event_count": 3,
            "severity": "medium",
            "manual_report": {
                "noise_type": "충격성 소리",
                "noise_time_slot": "주로 야간",
                "noise_frequency": "거의 매일",
                "situation_description": "늦은 밤에 반복되는 큰 소리",
            },
        }
    )

    assert fake.responses.last_kwargs is not None
    user_payload = fake.responses.last_kwargs["input"][1]["content"]
    parsed = json.loads(user_payload)
    assert parsed["manual_report"]["noise_type"] == "충격성 소리"


def test_openai_client_extract_output_items_dict_shape() -> None:
    payload = {
        "event_summary": "요약",
        "resident_message": "확인 부탁드립니다.",
        "admin_summary": "관리자 요약",
        "recommended_action": "quiet_time_request",
        "tone_check": {
            "is_neutral": True,
            "contains_blame": False,
            "contains_threat": False,
        },
    }
    output_items = [
        {
            "content": [
                {"type": "output_text", "text": json.dumps(payload, ensure_ascii=False)}
            ]
        }
    ]
    fake = _FakeClientWithOutputItems(output_items)
    client = OpenAIMessageClient(settings=_settings(max_retries=0), client=fake)
    result = client.generate_message(
        event_context={
            "event_type": "impact_noise",
            "time_range": "23:10-23:20",
            "event_count": 3,
            "severity": "high",
        }
    )
    assert result["resident_message"] == "확인 부탁드립니다."
