from datetime import datetime

from ai.config import (
    AISettings,
    ClassifierSettings,
    NightTimeSettings,
    OpenAISettings,
    PatternSettings,
    ThresholdSettings,
)
from ai.event_classifier import classify_event
from ai.lgbm_runtime import LGBMInferenceResult
from ai.schemas import EventFeatures


def _event(
    db: float,
    vib: int,
    dur: int,
    ts: str,
    accel_delta: float = 0.0,
    recent_count_10min: int = 0,
) -> EventFeatures:
    return EventFeatures(
        device_id="SENSOR-A101-01",
        source="simulator",
        sound_level=db,
        vibration_value=vib,
        duration_ms=dur,
        accel_delta=accel_delta,
        timestamp=datetime.fromisoformat(ts),
        recent_count_10min=recent_count_10min,
    )


def _settings_with_backend(backend: str) -> AISettings:
    return AISettings(
        thresholds=ThresholdSettings(),
        nighttime=NightTimeSettings(),
        pattern=PatternSettings(),
        openai=OpenAISettings(enabled=False),
        classifier=ClassifierSettings(
            backend=backend,
            lgbm_model_dir="./ai/tests/.tmp/nonexistent_models",
            lgbm_min_confidence=0.7,
            lgbm_shadow_mode=False,
        ),
    )


class _FakeRuntime:
    def __init__(self, response: LGBMInferenceResult):
        self.response = response
        self.calls = 0

    def predict_event(
        self,
        event: EventFeatures,
        *,
        min_confidence: float | None = None,
    ) -> LGBMInferenceResult:
        _ = event
        _ = min_confidence
        self.calls += 1
        return self.response


def test_background_noise_classification() -> None:
    event = _event(38.0, 80, 1200, "2026-05-04T14:10:00+09:00")
    result = classify_event(event)
    assert result.event_type == "background_noise"
    assert result.severity == "low"
    assert result.is_meaningful is False


def test_daily_noise_classification() -> None:
    event = _event(52.0, 120, 6000, "2026-05-04T20:10:00+09:00")
    result = classify_event(event)
    assert result.event_type == "daily_noise"
    assert result.severity in {"low", "medium"}
    assert result.is_meaningful is False


def test_impact_noise_high_severity() -> None:
    event = _event(63.0, 720, 9000, "2026-05-04T23:10:00+09:00", accel_delta=0.2)
    result = classify_event(event)
    assert result.event_type == "impact_noise"
    assert result.severity in {"high", "critical"}
    assert result.is_meaningful is True


def test_repeated_vibration_classification_with_meaningful_count() -> None:
    event = _event(
        48.0,
        420,
        3000,
        "2026-05-04T23:20:00+09:00",
        accel_delta=0.02,
        recent_count_10min=3,
    )
    result = classify_event(event)
    assert result.event_type == "repeated_vibration"
    assert result.severity in {"medium", "high"}
    assert result.is_meaningful is True


def test_repeated_vibration_not_triggered_by_raw_influx_without_meaningful_count() -> None:
    # Even if many raw samples exist upstream, classifier only accepts meaningful
    # recent_count_10min contract. Zero means no repeated-vibration trigger.
    event = _event(
        48.0,
        420,
        3000,
        "2026-05-04T23:20:00+09:00",
        accel_delta=0.02,
        recent_count_10min=0,
    )
    result = classify_event(event)
    assert result.event_type != "repeated_vibration"


def test_airborne_threshold_differs_between_day_and_night() -> None:
    # 41dB is below day airborne(45) but above night airborne(40).
    day_event = _event(41.0, 100, 1200, "2026-05-04T21:30:00+09:00")
    night_event = _event(41.0, 100, 3200, "2026-05-04T23:30:00+09:00")

    day_result = classify_event(day_event)
    night_result = classify_event(night_event)

    assert day_result.event_type == "background_noise"
    assert night_result.event_type == "daily_noise"


def test_impact_lmax_threshold_differs_between_day_and_night() -> None:
    # 53dB + strong vibration: night(>=52) can be impact, day(<57) should not.
    day_event = _event(53.0, 700, 6000, "2026-05-04T14:30:00+09:00", accel_delta=0.2)
    night_event = _event(53.0, 700, 6000, "2026-05-04T23:30:00+09:00", accel_delta=0.2)

    day_result = classify_event(day_event)
    night_result = classify_event(night_event)

    assert day_result.event_type != "impact_noise"
    assert night_result.event_type == "impact_noise"


def test_lightgbm_backend_falls_back_to_rule_on_runtime_fallback() -> None:
    event = _event(
        48.0,
        420,
        3000,
        "2026-05-04T23:20:00+09:00",
        accel_delta=0.02,
        recent_count_10min=3,
    )
    runtime = _FakeRuntime(
        LGBMInferenceResult(
            should_fallback=True,
            reason="missing LightGBM artifacts",
        )
    )
    result = classify_event(
        event,
        settings=_settings_with_backend("lightgbm"),
        runtime=runtime,
    )
    assert runtime.calls == 1
    assert result.event_type == "repeated_vibration"
    assert "lgbm_fallback" in result.rule_hits
    assert result.features["classifier_backend"] == "rule_fallback"
    assert "lgbm_reason" in result.features


def test_hybrid_backend_uses_lgbm_prediction_when_available() -> None:
    event = _event(
        52.0,
        120,
        4500,
        "2026-05-04T23:20:00+09:00",
        accel_delta=0.01,
        recent_count_10min=0,
    )
    runtime = _FakeRuntime(
        LGBMInferenceResult(
            should_fallback=False,
            reason=None,
            is_meaningful=True,
            meaningful_probability=0.94,
            event_type="repeated_vibration",
            event_type_probability=0.88,
            event_type_probabilities={"daily_noise": 0.12, "repeated_vibration": 0.88},
        )
    )
    result = classify_event(
        event,
        settings=_settings_with_backend("hybrid"),
        runtime=runtime,
    )
    assert runtime.calls == 1
    assert result.event_type == "repeated_vibration"
    assert result.features["classifier_backend"] == "hybrid/lgbm"
    assert result.confidence == 0.88
    assert result.is_meaningful is True


def test_impact_rule_backup_skips_lgbm_even_in_hybrid() -> None:
    event = _event(
        63.0,
        720,
        9000,
        "2026-05-04T23:10:00+09:00",
        accel_delta=0.2,
        recent_count_10min=0,
    )
    runtime = _FakeRuntime(
        LGBMInferenceResult(
            should_fallback=False,
            reason=None,
            is_meaningful=True,
            meaningful_probability=0.99,
            event_type="daily_noise",
            event_type_probability=0.97,
            event_type_probabilities={"daily_noise": 0.97, "repeated_vibration": 0.03},
        )
    )
    result = classify_event(
        event,
        settings=_settings_with_backend("hybrid"),
        runtime=runtime,
    )
    assert runtime.calls == 0
    assert result.event_type == "impact_noise"
    assert "impact_rule_backup" in result.rule_hits
    assert result.features["classifier_backend"] == "hybrid/rule_impact_backup"
