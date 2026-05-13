from __future__ import annotations

import pytest

from ai.config import AISettings, get_settings


def _clear_settings_cache() -> None:
    get_settings.cache_clear()


def test_classifier_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("AI_CLASSIFIER_BACKEND", raising=False)
    monkeypatch.delenv("AI_LGBM_MODEL_DIR", raising=False)
    monkeypatch.delenv("AI_LGBM_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("AI_LGBM_SHADOW_MODE", raising=False)
    monkeypatch.delenv("AI_LGBM_SHADOW_LOG_PATH", raising=False)
    _clear_settings_cache()

    cfg = AISettings.from_env()
    assert cfg.classifier.backend == "rule"
    assert cfg.classifier.lgbm_model_dir == "./ai/artifacts/models"
    assert cfg.classifier.lgbm_min_confidence == 0.7
    assert cfg.classifier.lgbm_shadow_mode is False
    assert cfg.classifier.lgbm_shadow_log_path == "./ai/artifacts/logs/lgbm_shadow.jsonl"


def test_classifier_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AI_CLASSIFIER_BACKEND", "HYBRID")
    monkeypatch.setenv("AI_LGBM_MODEL_DIR", "./models/current")
    monkeypatch.setenv("AI_LGBM_MIN_CONFIDENCE", "0.82")
    monkeypatch.setenv("AI_LGBM_SHADOW_MODE", "true")
    monkeypatch.setenv("AI_LGBM_SHADOW_LOG_PATH", "./logs/shadow.jsonl")
    _clear_settings_cache()

    cfg = AISettings.from_env()
    assert cfg.classifier.backend == "hybrid"
    assert cfg.classifier.lgbm_model_dir == "./models/current"
    assert cfg.classifier.lgbm_min_confidence == pytest.approx(0.82)
    assert cfg.classifier.lgbm_shadow_mode is True
    assert cfg.classifier.lgbm_shadow_log_path == "./logs/shadow.jsonl"


def test_classifier_backend_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("AI_CLASSIFIER_BACKEND", "invalid_backend")
    _clear_settings_cache()

    with pytest.raises(ValueError):
        AISettings.from_env()
