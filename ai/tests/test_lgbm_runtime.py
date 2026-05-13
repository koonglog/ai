from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from ai.config import (
    AISettings,
    ClassifierSettings,
    NightTimeSettings,
    OpenAISettings,
    PatternSettings,
    ThresholdSettings,
)
from ai.lgbm_runtime import LightGBMRuntime
from ai.schemas import EventFeatures
from ai.training.train_event_type_lgbm import train_event_type_model
from ai.training.train_is_meaningful_lgbm import train_is_meaningful_model


def _case_dir() -> Path:
    tmp_root = Path(__file__).resolve().parent / ".tmp"
    tmp_root.mkdir(exist_ok=True)
    case_dir = tmp_root / f"lgbm_runtime_{uuid4().hex}"
    case_dir.mkdir()
    return case_dir


def _settings(model_dir: Path, *, min_confidence: float = 0.7) -> AISettings:
    return AISettings(
        thresholds=ThresholdSettings(),
        nighttime=NightTimeSettings(),
        pattern=PatternSettings(),
        openai=OpenAISettings(enabled=False),
        classifier=ClassifierSettings(
            backend="lightgbm",
            lgbm_model_dir=str(model_dir),
            lgbm_min_confidence=min_confidence,
            lgbm_shadow_mode=False,
        ),
    )


def _event(
    *,
    sound_level: float = 58.0,
    vibration_value: int = 520,
    duration_ms: int = 5000,
    accel_delta: float = 0.06,
    ts: str = "2026-05-13T23:20:00+09:00",
    recent_count_10min: int = 3,
) -> EventFeatures:
    return EventFeatures(
        device_id="SENSOR-A101-01",
        source="test",
        sound_level=sound_level,
        vibration_value=vibration_value,
        duration_ms=duration_ms,
        accel_delta=accel_delta,
        timestamp=datetime.fromisoformat(ts),
        recent_count_10min=recent_count_10min,
    )


def _write_demo_dataset(path: Path, rows: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "sensor_id",
        "household_id",
        "sound_level",
        "vibration_value",
        "duration_ms",
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
        "received_at",
        "sensor_timestamp",
        "is_meaningful_label",
        "event_type_label",
        "severity_label",
        "noise_event_id",
        "noise_event_started_at",
    ]
    base = datetime.fromisoformat("2026-05-11T00:00:00")
    payload_rows: list[dict[str, str]] = []
    for i in range(rows):
        ts = base + timedelta(minutes=5 * i)
        meaningful = 1 if (i % 3 != 0) else 0
        if meaningful:
            event_type = "daily_noise" if (i % 2 == 0) else "repeated_vibration"
            severity = "medium" if event_type == "daily_noise" else "high"
            noise_id = str(1000 + i)
            noise_started = ts.isoformat(sep=" ")
        else:
            event_type = ""
            severity = ""
            noise_id = ""
            noise_started = ""

        sound = 42.0 + (i % 15)
        vibration = 120 + (i % 9) * 60
        duration = 2000 + (i % 5) * 1000
        if meaningful and event_type == "repeated_vibration":
            vibration += 180

        payload_rows.append(
            {
                "id": str(i + 1),
                "sensor_id": "SENSOR-A101-01",
                "household_id": "1",
                "sound_level": f"{sound:.3f}",
                "vibration_value": str(vibration),
                "duration_ms": str(duration),
                "acceleration_x": "0.0",
                "acceleration_y": "0.0",
                "acceleration_z": "1.0",
                "received_at": ts.isoformat(sep=" "),
                "sensor_timestamp": ts.isoformat(sep=" "),
                "is_meaningful_label": str(meaningful),
                "event_type_label": event_type,
                "severity_label": severity,
                "noise_event_id": noise_id,
                "noise_event_started_at": noise_started,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload_rows)


def test_runtime_fallback_when_artifacts_missing() -> None:
    model_dir = _case_dir() / "missing_models"
    model_dir.mkdir()
    runtime = LightGBMRuntime(settings=_settings(model_dir))

    result = runtime.predict_event(_event())
    assert result.should_fallback is True
    assert result.reason is not None
    assert "missing LightGBM artifacts" in result.reason


def test_runtime_predict_success_with_trained_models() -> None:
    case_dir = _case_dir()
    dataset = case_dir / "dataset.csv"
    model_dir = case_dir / "models"
    _write_demo_dataset(dataset, rows=120)

    train_is_meaningful_model(
        dataset=dataset,
        out_dir=model_dir,
        train_ratio=0.8,
        seed=42,
        n_estimators=120,
        early_stopping_rounds=10,
    )
    train_event_type_model(
        dataset=dataset,
        out_dir=model_dir,
        classes=["daily_noise", "repeated_vibration"],
        train_ratio=0.8,
        seed=42,
        n_estimators=120,
        early_stopping_rounds=10,
    )

    runtime = LightGBMRuntime(settings=_settings(model_dir, min_confidence=0.2))
    result = runtime.predict_event(_event())

    assert result.should_fallback is False, result.reason
    assert result.is_meaningful is True
    assert result.meaningful_probability is not None
    assert result.event_type in {"daily_noise", "repeated_vibration"}
    assert result.event_type_probability is not None
    assert result.event_type_probabilities is not None
    assert set(result.event_type_probabilities.keys()) == {
        "daily_noise",
        "repeated_vibration",
    }
    assert runtime.status()["ready"] is True


class _FakeBooster:
    def __init__(self, payload):
        self.payload = payload

    def predict(self, _x):
        return self.payload


def test_runtime_fallback_when_meaningful_probability_too_low() -> None:
    runtime = LightGBMRuntime(settings=_settings(_case_dir() / "unused"))
    runtime._load_attempted = True
    runtime._ready = True
    runtime._meaningful_booster = _FakeBooster([0.2])  # type: ignore[assignment]
    runtime._event_type_booster = _FakeBooster([[0.7, 0.3]])  # type: ignore[assignment]
    runtime._id_to_label = ["daily_noise", "repeated_vibration"]

    result = runtime.predict_event(_event(), min_confidence=0.7)
    assert result.should_fallback is True
    assert result.reason is not None
    assert "meaningful probability below threshold" in result.reason


def test_runtime_fallback_when_event_type_probability_too_low() -> None:
    runtime = LightGBMRuntime(settings=_settings(_case_dir() / "unused"))
    runtime._load_attempted = True
    runtime._ready = True
    runtime._meaningful_booster = _FakeBooster([0.92])  # type: ignore[assignment]
    runtime._event_type_booster = _FakeBooster([[0.55, 0.45]])  # type: ignore[assignment]
    runtime._id_to_label = ["daily_noise", "repeated_vibration"]

    result = runtime.predict_event(_event(), min_confidence=0.8)
    assert result.should_fallback is True
    assert result.reason is not None
    assert "event_type probability below threshold" in result.reason
