from ai.message_generator import contains_banned_expression, generate_mediation_message


def test_contains_banned_expression() -> None:
    assert contains_banned_expression("민폐 소음입니다.") is True
    assert contains_banned_expression("확인 부탁드립니다.") is False


def test_message_generator_uses_fallback_without_openai() -> None:
    context = {
        "event_type": "impact_noise",
        "severity": "medium",
        "event_count": 2,
        "time_range": "23:10-23:20",
    }
    result = generate_mediation_message(context)
    assert result.generation_method == "fallback_template"


def test_fallback_message_reflects_manual_report_context() -> None:
    context = {
        "event_type": "daily_noise",
        "severity": "medium",
        "event_count": 1,
        "time_range": "22:00-22:30",
        "manual_report": {
            "noise_type": "가구 끄는 소리",
            "noise_time_slot": "주로 야간",
            "noise_frequency": "주 4~5회",
            "situation_description": "밤 늦게 반복적으로 발생",
        },
    }
    result = generate_mediation_message(context)
    assert result.generation_method == "fallback_template"
    assert "신고 소음 유형: 가구 끄는 소리" in result.admin_summary
