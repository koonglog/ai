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
