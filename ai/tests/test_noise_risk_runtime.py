from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from ai import dashboard_api
from ai.config import (
    AISettings,
    ClassifierSettings,
    NightTimeSettings,
    OpenAISettings,
    PatternSettings,
    ThresholdSettings,
)
from ai.event_classifier import classify_event
from ai.lgbm_runtime import LightGBMRuntime
from ai.ml_features import FEATURE_COLUMNS, FEATURE_SPEC_VERSION, feature_dict_from_event
from ai.schemas import EventFeatures

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_NAME = "lgbm_noise_risk_feature_spec_2_0_0_base"
MODEL_PATH = ARTIFACT_DIR / f"{MODEL_NAME}.pkl"
METADATA_PATH = ARTIFACT_DIR / f"{MODEL_NAME}.metadata.json"
LABEL_MAP_PATH = ARTIFACT_DIR / f"{MODEL_NAME}.label_map.json"


def _settings(model_dir: Path, *, backend: str = "lightgbm") -> AISettings:
    return AISettings(
        thresholds=ThresholdSettings(),
        nighttime=NightTimeSettings(),
        pattern=PatternSettings(),
        openai=OpenAISettings(enabled=False),
        classifier=ClassifierSettings(
            backend=backend,
            lgbm_model_dir=str(model_dir),
            lgbm_min_confidence=0.7,
            lgbm_shadow_mode=False,
        ),
    )


def _event(
    *,
    sound_level: float,
    vibration_value: float,
    duration_ms: int,
    timestamp: str,
    impact_count_in_window: int = 0,
) -> EventFeatures:
    return EventFeatures(
        device_id="SENSOR-A101-01",
        source="sensor",
        sound_level=sound_level,
        vibration_value=vibration_value,
        duration_ms=duration_ms,
        accel_delta=0.0,
        timestamp=datetime.fromisoformat(timestamp),
        impact_count_in_window=impact_count_in_window,
    )


def test_noise_risk_artifact_metadata_label_map_and_feature_names_match() -> None:
    assert MODEL_PATH.exists()
    assert METADATA_PATH.exists()
    assert LABEL_MAP_PATH.exists()

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))

    assert metadata["feature_spec_version"] == FEATURE_SPEC_VERSION
    assert metadata["feature_names"] == FEATURE_COLUMNS
    assert label_map["id_to_label"] == ["normal", "caution", "warning", "high"]

    runtime = LightGBMRuntime(settings=_settings(ARTIFACT_DIR / "models"))
    status = runtime.status()

    assert status["ready"] is True, status["error"]
    assert status["runtime_mode"] == "noise_risk"
    assert Path(status["model_path"]).resolve() == MODEL_PATH.resolve()
    assert Path(status["metadata_path"]).resolve() == METADATA_PATH.resolve()
    assert Path(status["label_map_path"]).resolve() == LABEL_MAP_PATH.resolve()
    assert status["feature_spec_version"] == FEATURE_SPEC_VERSION
    assert status["model_version"] == MODEL_NAME
    assert status["classes"] == ["normal", "caution", "warning", "high"]
    assert status["feature_names_match"] is True


def test_noise_risk_sample_inference_uses_lightgbm_path() -> None:
    settings = _settings(ARTIFACT_DIR / "models")
    runtime = LightGBMRuntime(settings=settings)
    cases = [
        {
            "event": _event(
                sound_level=35.0,
                vibration_value=8.0,
                duration_ms=2000,
                timestamp="2026-05-11T10:00:00+09:00",
            ),
            "expected_level": "normal",
            "expected_period": "daytime",
            "expected_acc": 0.005,
        },
        {
            "event": _event(
                sound_level=37.0,
                vibration_value=10.8,
                duration_ms=2000,
                timestamp="2026-05-11T10:00:00+09:00",
            ),
            "expected_level": "normal",
            "expected_period": "daytime",
            "expected_acc": None,
        },
        {
            "event": _event(
                sound_level=58.0,
                vibration_value=1007.0,
                duration_ms=2000,
                timestamp="2026-05-11T22:00:00+09:00",
                impact_count_in_window=3,
            ),
            "expected_level": "high",
            "expected_period": "nighttime",
            "expected_acc": 0.45,
        },
        {
            "event": _event(
                sound_level=66.0,
                vibration_value=1023.0,
                duration_ms=2000,
                timestamp="2026-05-11T22:10:00+09:00",
                impact_count_in_window=6,
            ),
            "expected_level": "high",
            "expected_period": "nighttime",
            "expected_acc": None,
        },
    ]

    for case in cases:
        event = case["event"]
        result = classify_event(event, settings=settings, runtime=runtime)
        feature_dict = feature_dict_from_event(event, settings=settings)

        assert result.features["prediction_path"] == "lightgbm"
        assert result.features["classifier_backend"] == "lgbm_noise_risk"
        assert result.features["feature_spec_version"] == FEATURE_SPEC_VERSION
        assert result.features["model_version"] == MODEL_NAME
        assert result.noise_risk_level == case["expected_level"]
        assert result.time_period == case["expected_period"]
        assert result.vibration_raw == event.vibration_value
        assert math.isfinite(result.vibration_acc_mps2)
        assert math.isfinite(result.vibration_dbv)
        assert feature_dict["vibration_raw"] == event.vibration_value
        if case["expected_acc"] is not None:
            assert math.isclose(result.vibration_acc_mps2, case["expected_acc"], abs_tol=1e-6)
        if event.vibration_value == 10.8:
            assert result.vibration_raw == 10.8
        if event.vibration_value == 1023.0:
            assert result.vibration_acc_mps2 > 0.45
        assert "lgbm_fallback" not in result.rule_hits


def test_noise_risk_runtime_falls_back_only_when_artifact_unavailable() -> None:
    missing_dir = Path(__file__).resolve().parent / ".tmp" / f"missing_noise_risk_{uuid4().hex}"
    missing_dir.mkdir(parents=True)
    settings = _settings(missing_dir)
    runtime = LightGBMRuntime(settings=settings)
    event = _event(
        sound_level=37.0,
        vibration_value=10.8,
        duration_ms=2000,
        timestamp="2026-05-11T10:00:00+09:00",
    )

    result = classify_event(event, settings=settings, runtime=runtime)

    assert result.features["prediction_path"] == "rule_fallback"
    assert result.features["classifier_backend"] == "rule_fallback"
    assert "lgbm_fallback" in result.rule_hits
    assert "missing LightGBM artifacts" in result.features["lgbm_reason"]


def test_classify_event_api_returns_noise_risk_model_fields(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_OPENAI", "false")
    monkeypatch.setenv("AI_CLASSIFIER_BACKEND", "lightgbm")
    monkeypatch.setenv("AI_LGBM_MODEL_DIR", str(ARTIFACT_DIR / "models"))
    dashboard_api.get_backend_settings.cache_clear()
    dashboard_api.get_engine.cache_clear()
    dashboard_api.get_schema.cache_clear()
    dashboard_api.get_settings.cache_clear()
    client = TestClient(dashboard_api.create_dashboard_app())

    payload = {
        "sensor_id": "SENSOR-A101-01",
        "household_id": 1,
        "source": "backend",
        "sensor_timestamp": "2026-05-11T22:00:00",
        "event_feature": {
            "sound_level": 58.0,
            "vibration_value": 1007.0,
            "duration_ms": 2000,
            "impact_count_in_window": 3,
        },
    }

    res = client.post("/api/v1/ai/classify-event", json=payload)

    assert res.status_code == 200
    classification = res.json()["classification"]
    for field in {
        "noise_risk_level",
        "vibration_raw",
        "vibration_acc_mps2",
        "vibration_dbv",
        "time_period",
        "impact_count",
        "reason",
        "is_daytime",
        "is_nighttime",
        "vibration_level_category",
        "repeated_impact_flag",
        "prediction_path",
        "model_version",
        "feature_spec_version",
        "timestamp_source",
    }:
        assert field in classification

    assert classification["prediction_path"] == "lightgbm"
    assert classification["model_version"] == MODEL_NAME
    assert classification["feature_spec_version"] == FEATURE_SPEC_VERSION
    assert classification["noise_risk_level"] == "high"
    assert classification["time_period"] == "nighttime"
    assert classification["is_nighttime"] is True
    assert classification["repeated_impact_flag"] is True
    assert classification["timestamp_source"] == "sensor_timestamp"
    forbidden_legal_terms = ["법정 기준 초과", "위반", "불법 층간소음"]
    assert all(term not in classification["reason"] for term in forbidden_legal_terms)
